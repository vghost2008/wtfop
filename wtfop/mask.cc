#include <stdio.h>
#include <cfloat>
#include <list>
#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <boost/algorithm/clamp.hpp>
#include <random>
#include<opencv2/opencv.hpp>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/work_sharder.h"
#include <boost/algorithm/clamp.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/at.hpp>
#include "bboxes.h"
#include <future>
#include "wtoolkit.h"

using namespace tensorflow;
using namespace std;
namespace ba=boost::algorithm;
namespace bm=boost::mpl;

typedef Eigen::ThreadPoolDevice CPUDevice;

/*
 * 
 * masks:[batch_size,Nr,h,w]
 * labels: [batch_size,Nr]
 * lens:[batch_size]
 * output_bboxes:[batch_size,nr,4] (ymin,xmin,ymax,xmax)
 * output_labels:[batch_size,nr]
 * output_lens:[batch_size]
 * output_ids:[batch_size,nr] 用于表示实例的编号，如第一个batch中的第二个实例所生成的所有的box的ids为3(id的编号从1开始)
 */
REGISTER_OP("MaskLineBboxes")
    .Attr("T: {int64,int32}")
	.Attr("max_output_nr:int")
    .Input("mask: uint8")
    .Input("labels: T")
    .Input("lens: int32")
	.Output("output_bboxes:float")
	.Output("output_labels:T")
	.Output("output_lens:int32")
	.Output("output_ids:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            int        nr = -1;

            c->GetAttr("max_output_nr",&nr);

            const auto batch_size = c->Value(c->Dim(c->input(0),0));
            const auto shape0     = c->MakeShape({batch_size,nr,4});
            const auto shape1     = c->Matrix(batch_size,nr);
            const auto shape2     = c->Vector(batch_size);

			c->set_output(0, shape0);
			c->set_output(1, shape1);
			c->set_output(2, shape2);
			c->set_output(3, shape1);
			return Status::OK();
			});

template <typename Device,typename T>
class MaskLineBboxesOp: public OpKernel {
    private:
        using bbox_t = tuple<float,float,float,float>;
	public:
		explicit MaskLineBboxesOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("max_output_nr", &max_output_nr_));
		}

		void Compute(OpKernelContext* context) override
		{
            TIME_THISV1("MaskLineBboxes");
			const Tensor &_mask= context->input(0);
			const Tensor &_labels= context->input(1);
			const Tensor &_lens = context->input(2);
			auto mask= _mask.template tensor<uint8_t,4>();
            auto labels = _labels.template tensor<T,2>();
            auto lens = _lens.template tensor<int32_t,1>();

			OP_REQUIRES(context, _mask.dims() == 4, errors::InvalidArgument("mask data must be 4-dimensional"));
			OP_REQUIRES(context, _labels.dims() == 2, errors::InvalidArgument("labels data must be 2-dimensional"));
			OP_REQUIRES(context, _lens.dims() == 1, errors::InvalidArgument("lens data must be 1-dimensional"));

			const auto     batch_size = _mask.dim_size(0);
			const auto     data_nr   = _mask.dim_size(1);
            list<vector<bbox_t>> out_bboxes;
            list<vector<int>> out_labels;
            list<vector<int>> out_ids;

            for(auto i=0; i<batch_size; ++i) {
                vector<bbox_t> res;
                vector<int> res_labels;
                vector<int> res_ids;
                res.reserve(1024);
                for(auto j=0; j<lens(i); ++j) {
                    const auto label = labels(i,j);
                    auto res0 = get_bboxes(mask.chip(i,0).chip(j,0));
                    if(!res0.empty()) {
                        res.insert(res.end(),res0.begin(),res0.end());
                        res_labels.insert(res_labels.end(),res0.size(),label);
                        res_ids.insert(res_ids.end(),res0.size(),j+1);
                    }
                }
			    OP_REQUIRES(context, res.size() == res_labels.size(), errors::InvalidArgument("size of bboxes should equal size of labels."));
                out_bboxes.push_back(std::move(res));
                out_labels.push_back(std::move(res_labels));
                out_ids.push_back(std::move(res_ids));
            }

            auto output_nr = max_output_nr_;

            if(output_nr<=0) {
                auto it = max_element(out_labels.begin(),out_labels.end(),[](const auto& v0,const auto& v1){ return v0.size()<v1.size();});
                output_nr = it->size();
            } 

			int dims_3d[3] = {batch_size,output_nr,4};
			int dims_2d[2] = {batch_size,output_nr};
			int dims_1d[1] = {batch_size};
			TensorShape  outshape0;
			TensorShape  outshape1;
			TensorShape  outshape2;
			Tensor      *output_bbox   = NULL;
			Tensor      *output_labels = NULL;
			Tensor      *output_lens   = NULL;
			Tensor      *output_ids    = NULL;

			TensorShapeUtils::MakeShape(dims_3d, 3, &outshape0);
			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape1);
			TensorShapeUtils::MakeShape(dims_1d, 1, &outshape2);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_bbox));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_labels));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape2, &output_lens));
			OP_REQUIRES_OK(context, context->allocate_output(3, outshape1, &output_ids));

			auto obbox   = output_bbox->template tensor<float,3>();
			auto olabels = output_labels->template tensor<T,2>();
			auto olens   = output_lens->template tensor<int32_t,1>();
			auto oids    = output_ids->template tensor<int32_t,2>();

            obbox.setZero();
            olabels.setZero();
            auto itb = out_bboxes.begin();
            auto itl = out_labels.begin();
            auto iti = out_ids.begin();

			for(int i=0; i<batch_size; ++i,++itb,++itl,++iti) {
                olens(i) = itl->size();
                for(auto j=0; j<olens(i); ++j) {
                    obbox(i,j,0) = std::get<0>((*itb)[j]);
                    obbox(i,j,1) = std::get<1>((*itb)[j]);
                    obbox(i,j,2) = std::get<2>((*itb)[j]);
                    obbox(i,j,3) = std::get<3>((*itb)[j]);
                    olabels(i,j) = (*itl)[j];
                    oids(i,j) = (*iti)[j];
                }
			}
		}
        /*
         * mask: [h,w]
         */
        vector<bbox_t> get_bboxes(const Eigen::Tensor<uint8_t,2,Eigen::RowMajor>& mask) {
            const auto h = mask.dimension(0);
            const auto w = mask.dimension(1);
            const auto y_delta = 1.0/h;
            const auto x_delta = 1.0/w;
            vector<bbox_t> res;
            res.reserve(256);

            for(auto i=0; i<h; ++i) {
                const auto ymin = i*y_delta;
                const auto ymax = (i+1)*y_delta;
                for(auto j=0; j<w; ++j) {
                    if(mask(i,j)<1) continue;
                    auto begin_j = j;
                    while((mask(i,j)>0) && (j<w))++j;
                    const auto xmin = begin_j*x_delta;
                    const auto xmax = (j==w)?1.0:j*x_delta;
                    res.emplace_back(ymin,xmin,ymax,xmax);
                }
            }
            return res;
        }
	private:
        int max_output_nr_ = 0;
};
REGISTER_KERNEL_BUILDER(Name("MaskLineBboxes").Device(DEVICE_CPU).TypeConstraint<int32_t>("T"), MaskLineBboxesOp<CPUDevice,int32_t>);
REGISTER_KERNEL_BUILDER(Name("MaskLineBboxes").Device(DEVICE_CPU).TypeConstraint<tensorflow::int64>("T"), MaskLineBboxesOp<CPUDevice,tensorflow::int64>);

/*
 * 
 * masks:[Nr,h,w]
 * bboxes:[Nr,4]
 * size:[2]={H,W}
 * output_masks:[Nr,H,W]
 */
REGISTER_OP("FullSizeMask")
    .Attr("T: {float32,uint8}")
    .Input("mask: T")
    .Input("bboxes: float32")
    .Input("size: int32")
	.Output("output_masks:T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            const auto nr = c->Value(c->Dim(c->input(0),0));
            const auto shape0     = c->MakeShape({nr,-1,-1});
			c->set_output(0, shape0);
			return Status::OK();
			});

template <typename Device,typename T>
class FullSizeMaskOp: public OpKernel {
    private:
        using bbox_t = tuple<float,float,float,float>;
        using type_to_int = bm::map<
              bm::pair<uint8_t,bm::int_<CV_8UC1>>
                  , bm::pair<float,bm::int_<CV_32FC1>>
             >;
        using tensor_t = Eigen::Tensor<T,2,Eigen::RowMajor>;
        using tensor_map_t = Eigen::TensorMap<tensor_t>;
	public:
		explicit FullSizeMaskOp(OpKernelConstruction* context) : OpKernel(context) {
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &_mask= context->input(0);
			const Tensor &_bboxes = context->input(1);
			const Tensor &_size= context->input(2);

			OP_REQUIRES(context, _mask.dims() == 3, errors::InvalidArgument("mask data must be 3-dimensional"));
			OP_REQUIRES(context, _bboxes.dims() == 2, errors::InvalidArgument("bboxes data must be 2-dimensional"));
			OP_REQUIRES(context, _size.dims() == 1, errors::InvalidArgument("size data must be 1-dimensional"));

			auto         mask        = _mask.template flat<T>();
			auto         bboxes      = _bboxes.template tensor<float,2>();
			auto         size        = _size.template tensor<int,1>();
			const auto   mh          = _mask.dim_size(1);
			const auto   mw          = _mask.dim_size(2);
			const auto   H           = size(0);
			const auto   W           = size(1);
			const auto   data_nr     = _mask.dim_size(0);
			int          dims_3d[3]  = {data_nr,H,W};
			Tensor      *output_mask = NULL;
			TensorShape  outshape0;

			TensorShapeUtils::MakeShape(dims_3d, 3, &outshape0);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_mask));

            auto o_tensor = output_mask->template tensor<T,3>();
            const auto H_max = H-1;
            const auto W_max = W-1;
            constexpr auto kMinSize = 1e-3;

            o_tensor.setZero();

            for(auto i=0; i<data_nr; ++i) {
                if((fabs(bboxes(i,3)-bboxes(i,1))<kMinSize)
                    || (fabs(bboxes(i,2)-bboxes(i,0))<kMinSize))
                    continue;

                long xmin = ba::clamp(bboxes(i,1)*W_max,0,W_max);
                long ymin = ba::clamp(bboxes(i,0)*H_max,0,H_max);
                long xmax = ba::clamp(bboxes(i,3)*W_max,0,W_max);
                long ymax = ba::clamp(bboxes(i,2)*H_max,0,H_max);
                const cv::Mat input_mask(mh,mw,bm::at<type_to_int,T>::type::value,(void*)(mask.data()+i*mh*mw));
                cv::Mat dst_mask(ymax-ymin+1,xmax-xmin+1,bm::at<type_to_int,T>::type::value);

                cv::resize(input_mask,dst_mask,cv::Size(xmax-xmin+1,ymax-ymin+1),0,0,CV_INTER_LINEAR);


                tensor_map_t src_map((T*)dst_mask.data,dst_mask.rows,dst_mask.cols);
                Eigen::array<long,2> offset = {ymin,xmin};
                Eigen::array<long,2> extents = {dst_mask.rows,dst_mask.cols};

                o_tensor.chip(i,0).slice(offset,extents) = src_map;

                /*if(((xmax-xmin>mw) || (ymax-ymin>mh)) && (xmax>xmin) && (ymax>ymin)) {
                    cv::Mat dst_mask(H,W,bm::at<type_to_int,T>::type::value,output_mask->template flat<T>().data()+H*W*i);
                    cv::Mat src_mask = dst_mask.clone();
                    const auto k = max<int>(3,sqrt((xmax-xmin)*(ymax-ymin)/(mh*mw))+1);
                    cv::medianBlur(src_mask,dst_mask,(k/2)*2+1);
                }*/
                
            }
		}
};
REGISTER_KERNEL_BUILDER(Name("FullSizeMask").Device(DEVICE_CPU).TypeConstraint<float>("T"), FullSizeMaskOp<CPUDevice,float>);
REGISTER_KERNEL_BUILDER(Name("FullSizeMask").Device(DEVICE_CPU).TypeConstraint<uint8_t>("T"), FullSizeMaskOp<CPUDevice,uint8_t>);

/*
 * 对mask [Nr,H,W] 旋转指定角度，同时返回相应instance的bbox
 * bbox [N,4],[ymin,xmin,ymax,xmax], 绝对坐标
 */
REGISTER_OP("MaskRotate")
    .Attr("T: {uint8,float}")
    .Input("image: T")
    .Input("angle: float")
	.Output("o_image:T")
	.Output("bbox:float")
    .SetShapeFn([](shape_inference::InferenceContext* c){
            auto input_shape0 = c->input(0);
            c->set_output(0, input_shape0);
            auto data_nr = c->Dim(input_shape0,0);
            auto output_shape1 = c->MakeShape({data_nr,4});
            c->set_output(1, output_shape1);
			return Status::OK();
            });

template <typename Device, typename T>
class MaskRotateOp: public OpKernel {
    private:
        using Tensor3D = Eigen::Tensor<T,3,Eigen::RowMajor>;
        using type_to_int = bm::map<
              bm::pair<uint8_t,bm::int_<CV_8UC1>>
                  , bm::pair<float,bm::int_<CV_32FC1>>
             >;
	public:
		explicit MaskRotateOp(OpKernelConstruction* context) : OpKernel(context) {
		}
		void Compute(OpKernelContext* context) override
		{
            TIME_THISV1("RotateMask");
			const Tensor &_input_img = context->input(0);
			const Tensor &_angle = context->input(1);

			OP_REQUIRES(context, _input_img.dims() == 3, errors::InvalidArgument("tensor must be a 3-dimensional tensor"));
			OP_REQUIRES(context, _angle.dims() == 0, errors::InvalidArgument("angle be a 0-dimensional tensor"));

            auto         input_img     = _input_img.template flat<T>().data();
            auto         angle         = _angle.template flat<float>().data()[0];
            const auto   img_channel   = _input_img.dim_size(0);
            const auto   img_height    = _input_img.dim_size(1);
            const auto   img_width     = _input_img.dim_size(2);
            const int    dim2d[]       = {img_channel,4};
            TensorShape  output_shape;
            Tensor      *output_tensor = nullptr;
            Tensor      *output_bbox   = nullptr;

            TensorShapeUtils::MakeShape(dim2d,2,&output_shape);

			OP_REQUIRES_OK(context, context->allocate_output(0, _input_img.shape(), &output_tensor));
			OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &output_bbox));

            auto          o_tensor     = output_tensor->template flat<T>().data();
            auto          o_bbox       = output_bbox->template flat<float>().data();
            const cv::Point2f cp(img_width/2,img_height/2);
            const cv::Mat r            = cv::getRotationMatrix2D(cp,angle,1.0);
            const auto    cv_type      = bm::at<type_to_int,T>::type::value;
            auto fn  = [img_height,img_width,cv_type,r](T* i_data,T* o_data,float* bbox) {
                cv::Mat i_img(img_height,img_width,cv_type,i_data);
                cv::Mat o_img(img_height,img_width,cv_type,o_data);

                cv::warpAffine(i_img,o_img,r,cv::Size(img_width,img_height));
                getBBox(o_img,bbox);
            };
            list<future<void>> futures;

            for(auto i=0; i<img_channel; ++i) {
                auto i_data = input_img+i *img_width *img_height;
                auto o_data = o_tensor+i *img_width *img_height;
                auto bbox   = o_bbox+i *4;
                futures.emplace_back(async(launch::async,fn,(T*)i_data,o_data,bbox));
                if(futures.size()>8)
                    futures.pop_front();
            }
            futures.clear();
        }

        static void getBBox(const cv::Mat& img,float* bbox)
        {
            vector<vector<cv::Point>> contours;
            vector<cv::Vec4i> hierarchy;
            vector<cv::Point> points;
            const auto    cv_type      = bm::at<type_to_int,T>::type::value;
            cv::Mat dst_img(img.rows,img.cols,cv_type);


            if(cv_type == CV_32FC1) {
                cv::threshold(img,dst_img,0.5,255,CV_THRESH_BINARY);
                cv::Mat dst_img1;
                dst_img.convertTo(dst_img1,CV_8UC1);
                cv::findContours(dst_img1, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0,0));
            } else {
                cv::threshold(img,dst_img,127,255,CV_THRESH_BINARY);
                cv::findContours(dst_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0,0));
            }

            for (auto &cont:contours) 
                points.insert(points.end(),cont.begin(),cont.end());

            if(points.size()<2) {
               memset(bbox,0,sizeof(float)*4);
               return;
            }

            const auto rect = cv::boundingRect(points);

            bbox[0] = rect.y;
            bbox[1] = rect.x;
            bbox[2] = rect.y+rect.height;
            bbox[3] = rect.x+rect.width;
        }
};
REGISTER_KERNEL_BUILDER(Name("MaskRotate").Device(DEVICE_CPU).TypeConstraint<uint8_t>("T"), MaskRotateOp<CPUDevice, uint8_t>);
REGISTER_KERNEL_BUILDER(Name("MaskRotate").Device(DEVICE_CPU).TypeConstraint<float>("T"), MaskRotateOp<CPUDevice, float>);

/*
 * 对mask [Nr,H,W] 旋转指定角度，同时返回相应instance的bbox
 * bbox [N,4],[ymin,xmin,ymax,xmax], 绝对坐标
 */
REGISTER_OP("GetBboxesFromMask")
    .Attr("T: {uint8,float}")
    .Input("image: T")
	.Output("bbox:float")
    .SetShapeFn([](shape_inference::InferenceContext* c){
            auto input_shape0 = c->input(0);
            auto data_nr = c->Dim(input_shape0,0);
            auto output_shape = c->MakeShape({data_nr,4});
            c->set_output(0, output_shape);
			return Status::OK();
            });

template <typename Device, typename T>
class GetBboxesFromMaskOp: public OpKernel {
    private:
        using Tensor3D = Eigen::Tensor<T,3,Eigen::RowMajor>;
        using type_to_int = bm::map<
              bm::pair<uint8_t,bm::int_<CV_8UC1>>
                  , bm::pair<float,bm::int_<CV_32FC1>>
             >;
	public:
		explicit GetBboxesFromMaskOp(OpKernelConstruction* context) : OpKernel(context) {
		}
		void Compute(OpKernelContext* context) override
		{
            TIME_THISV1("GetBboxesFromMask");
			const Tensor &_input_img = context->input(0);

			OP_REQUIRES(context, _input_img.dims() == 3, errors::InvalidArgument("tensor must be a 3-dimensional tensor"));

            auto         input_img     = _input_img.template flat<T>().data();
            const auto   img_channel   = _input_img.dim_size(0);
            const auto   img_height    = _input_img.dim_size(1);
            const auto   img_width     = _input_img.dim_size(2);
            const int    dim2d[]       = {img_channel,4};
            TensorShape  output_shape;
            Tensor      *output_bbox   = nullptr;

            TensorShapeUtils::MakeShape(dim2d,2,&output_shape);

			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_bbox));

            auto          o_bbox       = output_bbox->template flat<float>().data();
            const auto    cv_type      = bm::at<type_to_int,T>::type::value;
            auto fn  = [img_height,img_width,cv_type](T* i_data,float* bbox) {
                cv::Mat i_img(img_height,img_width,cv_type,i_data);
                getBBox(i_img,bbox);
            };
            list<future<void>> futures;

            for(auto i=0; i<img_channel; ++i) {
                auto i_data = input_img+i *img_width *img_height;
                auto bbox   = o_bbox+i *4;
                futures.emplace_back(async(launch::async,fn,(T*)i_data,bbox));
                if(futures.size()>8)
                    futures.pop_front();
            }
            futures.clear();
        }

        static void getBBox(const cv::Mat& img,float* bbox)
        {
            vector<vector<cv::Point>> contours;
            vector<cv::Vec4i> hierarchy;
            vector<cv::Point> points;
            const auto    cv_type      = bm::at<type_to_int,T>::type::value;
            cv::Mat dst_img(img.rows,img.cols,cv_type);


            if(cv_type == CV_32FC1) {
                cv::threshold(img,dst_img,0.5,255,CV_THRESH_BINARY);
                cv::Mat dst_img1;
                dst_img.convertTo(dst_img1,CV_8UC1);
                cv::findContours(dst_img1, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0,0));
            } else {
                cv::threshold(img,dst_img,127,255,CV_THRESH_BINARY);
                cv::findContours(dst_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0,0));
            }

            for (auto &cont:contours) 
                points.insert(points.end(),cont.begin(),cont.end());

            if(points.size()<2) {
               memset(bbox,0,sizeof(float)*4);
               return;
            }

            const auto rect = cv::boundingRect(points);

            bbox[0] = rect.y;
            bbox[1] = rect.x;
            bbox[2] = rect.y+rect.height;
            bbox[3] = rect.x+rect.width;
        }
};
REGISTER_KERNEL_BUILDER(Name("GetBboxesFromMask").Device(DEVICE_CPU).TypeConstraint<uint8_t>("T"), GetBboxesFromMaskOp<CPUDevice, uint8_t>);
REGISTER_KERNEL_BUILDER(Name("GetBboxesFromMask").Device(DEVICE_CPU).TypeConstraint<float>("T"), GetBboxesFromMaskOp<CPUDevice, float>);
