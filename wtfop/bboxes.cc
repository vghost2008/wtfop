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
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/work_sharder.h"
#include "bboxes.h"

using namespace tensorflow;
using namespace std;
typedef Eigen::ThreadPoolDevice CPUDevice;
/*
 * 
 * bottom_box:[batch_size,Nr,4](y,x,h,w)
 * bottom_pred:[batch_size,Nr,num_class]
 * output_box:[X,4]
 * output_classes:[X]
 * output_scores:[X]
 * output_batch_index:[X]
 */
REGISTER_OP("BoxesSelect")
    .Attr("T: {float, double,int32}")
	.Attr("threshold:float")
	.Attr("ignore_first:bool")
    .Input("bottom_box: T")
    .Input("bottom_pred: T")
	.Output("output_box:T")
	.Output("output_classes:T")
	.Output("output_scores:T")
	.Output("output_batch_index:T");

template <typename Device, typename T>
class BoxesSelectOp: public OpKernel {
	public:
		explicit BoxesSelectOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("ignore_first", &ignore_first));
			OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &bottom_box        = context->input(0);
			const Tensor &bottom_pred       = context->input(1);
			auto          bottom_box_flat   = bottom_box.flat<T>();
			auto          bottom_pred_flat  = bottom_pred.flat<T>();

			OP_REQUIRES(context, bottom_box.dims() == 3, errors::InvalidArgument("box data must be 3-dimensional"));
			OP_REQUIRES(context, bottom_pred.dims() == 3, errors::InvalidArgument("pred data must be 2-dimensional"));

			const auto     batch_nr  = bottom_box.dim_size(0);
			const auto     data_nr   = bottom_box.dim_size(1);
			const auto     class_nr  = bottom_pred.dim_size(2);
			using Outtype=tuple<const float*,int,float,int>; //box,class,scores,batch_index
			vector<Outtype>  tmp_outdata;
			auto type_func = ignore_first?type_without_first:type_with_first;
			auto shard = [this, &bottom_box_flat,&bottom_pred_flat,class_nr,data_nr,batch_nr,&tmp_outdata,type_func]
				(int64 start, int64 limit) {
					for (int64 b = start; b < limit; ++b) {
						int batch_ind = b;
						int data_ind  = batch_ind%data_nr;
						batch_ind /= data_nr;
						const auto   base_offset0 = batch_ind *data_nr *4+data_ind *4;
						const auto   base_offset1 = batch_ind *data_nr *class_nr+data_ind *class_nr;
						const float *box_data    = bottom_box_flat.data()+base_offset0;
						const float *pred        = bottom_pred_flat.data()+base_offset1;
						const auto   type        = type_func(pred,class_nr);
						const auto   scores      = pred[type];

						if((scores >= threshold) && (box_area(box_data)>1E-4)) 
							tmp_outdata.emplace_back(box_data,type,scores,batch_ind);
					}
				};

			const DeviceBase::CpuWorkerThreads& worker_threads =
			*(context->device()->tensorflow_cpu_worker_threads());
			const int64 total_cost= batch_nr*data_nr;
			//Shard(worker_threads.num_threads, worker_threads.workers,total_cost,total_cost, shard);
			shard(0,total_cost);
			int dims_2d[2] = {int(tmp_outdata.size()),4};
			int dims_1d[1] = {int(tmp_outdata.size())};
			TensorShape  outshape0;
			TensorShape  outshape1;
			Tensor      *output_box         = NULL;
			Tensor      *output_classes     = NULL;
			Tensor      *output_scores      = NULL;
			Tensor      *output_batch_index = NULL;

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape0);
			TensorShapeUtils::MakeShape(dims_1d, 1, &outshape1);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_box));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_classes));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_scores));
			OP_REQUIRES_OK(context, context->allocate_output(3, outshape1, &output_batch_index));

			auto obox         = output_box->template flat<T>();
			auto oclasses     = output_classes->template flat<T>();
			auto oscores      = output_scores->template flat<T>();
			auto obatch_index = output_batch_index->template flat<T>();

			for(int i=0; i<tmp_outdata.size(); ++i) {
				auto& data = tmp_outdata[i];
				auto box = get<0>(data);
				std::copy(box,box+4,obox.data()+4*i);
				oclasses(i) = get<1>(data);
				oscores(i) = get<2>(data);
				obatch_index(i) = get<3>(data);
			}
		}
		static int type_with_first(const float* data,size_t size) {
			auto it = max_element(data,data+size);
			return it-data;
		}
		static int type_without_first(const float* data,size_t size) {
			auto it = max_element(data+1,data+size);
			return it-data;
		}
	private:
		bool  ignore_first;
		float threshold;
};
REGISTER_KERNEL_BUILDER(Name("BoxesSelect").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesSelectOp<CPUDevice, float>);

/*
 * 要求数据已经按重要呈度从0到data_nr排好了序(从大到小）
 * bottom_box:[X,4](ymin,xmin,ymax,xmax)
 * classes:[X]
 * output_box:[Y,4]
 * output_classes:[Y]
 * output_index:[Y]
 * 输出时的相对位置不能改变
 */
REGISTER_OP("BoxesNms")
    .Attr("T: {float, double,int32}")
	.Attr("threshold:float")
	.Attr("classes_wise:bool")
    .Input("bottom_box: T")
    .Input("classes: int32")
	.Output("output_box:T")
	.Output("output_classes:int32")
	.Output("output_index:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			c->set_output(0, c->Matrix(-1, 4));
			c->set_output(1, c->Vector(-1));
			c->set_output(2, c->Vector(-1));
			return Status::OK();
			});

template <typename Device, typename T>
class BoxesNmsOp: public OpKernel {
	public:
		explicit BoxesNmsOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold));
			OP_REQUIRES_OK(context, context->GetAttr("classes_wise", &classes_wise));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &bottom_box          = context->input(0);
			const Tensor &bottom_classes      = context->input(1);
			auto          bottom_box_flat     = bottom_box.flat<T>();
			auto          bottom_classes_flat = bottom_classes.flat<int32>();

			OP_REQUIRES(context, bottom_box.dims() == 2, errors::InvalidArgument("box data must be 2-dimensional"));
			OP_REQUIRES(context, bottom_classes.dims() == 1, errors::InvalidArgument("classes data must be 1-dimensional"));

			const auto   data_nr           = bottom_box.dim_size(0);
			vector<bool> keep_mask(data_nr,true);
			const auto   loop_end          = data_nr-1;

			for(auto i=0; i<loop_end; ++i) {
				if(keep_mask[i]) {
					const auto iclass = bottom_classes_flat(i);
					for(auto j=i+1; j<data_nr; ++j) {
						if(classes_wise && (bottom_classes_flat(j) != iclass)) continue;
						if(bboxes_jaccard(bottom_box_flat.data()+i*4,bottom_box_flat.data()+j*4) < threshold) continue;
						keep_mask[j] = false;
					}
				}
			}

			const auto out_size = count(keep_mask.begin(),keep_mask.end(),true);
			int dims_2d[2] = {int(out_size),4};
			int dims_1d[1] = {int(out_size)};
			TensorShape  outshape0;
			TensorShape  outshape1;
			Tensor      *output_box         = NULL;
			Tensor      *output_classes     = NULL;
			Tensor      *output_index = NULL;

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape0);
			TensorShapeUtils::MakeShape(dims_1d, 1, &outshape1);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_box));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_classes));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_index));

			auto obox     = output_box->template flat<T>();
			auto oclasses = output_classes->template flat<int32>();
			auto oindex   = output_index->template flat<int32>();

			for(int i=0,j=0; i<data_nr; ++i) {
				if(!keep_mask[i]) continue;
				auto box = bottom_box_flat.data()+i*4;
				std::copy(box,box+4,obox.data()+4*j);
				oclasses(j) = bottom_classes_flat(i);
				oindex(j) = i;
				++j;
			}
		}
	private:
		float threshold    = 0.2;
		bool  classes_wise = true;
};
REGISTER_KERNEL_BUILDER(Name("BoxesNms").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesNmsOp<CPUDevice, float>);
/*
 * 要求数据已经按重要呈度从0到data_nr排好了序(从大到小）
 * bottom_box:[X,4](ymin,xmin,ymax,xmax)
 * classes:[X]
 * group:[Z,2]分组信息，分别为一个组里标签的开始与结束编号，不在分组信息的的默认为一个组
 * output_box:[Y,4]
 * output_classes:[Y]
 * output_index:[Y]
 * 输出时的相对位置不能改变
 */
REGISTER_OP("GroupBoxesNms")
    .Attr("T: {float, double,int32}")
	.Attr("threshold:float")
    .Input("bottom_box: T")
    .Input("classes: int32")
    .Input("group: int32")
	.Output("output_box:T")
	.Output("output_classes:int32")
	.Output("output_index:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			c->set_output(0, c->Matrix(-1, 4));
			c->set_output(1, c->Vector(-1));
			c->set_output(2, c->Vector(-1));
			return Status::OK();
			});

template <typename Device, typename T>
class GroupBoxesNmsOp: public OpKernel {
	public:
		explicit GroupBoxesNmsOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &bottom_box          = context->input(0);
			const Tensor &bottom_classes      = context->input(1);
			const Tensor &_group              = context->input(2);
			auto          bottom_box_flat     = bottom_box.flat<T>();
			auto          bottom_classes_flat = bottom_classes.flat<int32>();
			auto          group               = _group.template tensor<int,2>();

			OP_REQUIRES(context, bottom_box.dims() == 2, errors::InvalidArgument("box data must be 2-dimensional"));
			OP_REQUIRES(context, bottom_classes.dims() == 1, errors::InvalidArgument("classes data must be 1-dimensional"));
			OP_REQUIRES(context, _group.dims() == 2, errors::InvalidArgument("group data must be 2-dimensional"));

			const auto   data_nr           = bottom_box.dim_size(0);
			vector<bool> keep_mask(data_nr,true);
			const auto   loop_end          = data_nr-1;

			for(auto i=0; i<loop_end; ++i) {
				if(keep_mask[i]) {
					const auto iclass = bottom_classes_flat(i);
                    const auto igroup = get_group(group,iclass);
					for(auto j=i+1; j<data_nr; ++j) {
						if(igroup != get_group(group,bottom_classes_flat(j))) continue;
						if(bboxes_jaccard(bottom_box_flat.data()+i*4,bottom_box_flat.data()+j*4) < threshold) continue;
						keep_mask[j] = false;
					}
				}
			}

			const auto out_size = count(keep_mask.begin(),keep_mask.end(),true);
			int dims_2d[2] = {int(out_size),4};
			int dims_1d[1] = {int(out_size)};
			TensorShape  outshape0;
			TensorShape  outshape1;
			Tensor      *output_box         = NULL;
			Tensor      *output_classes     = NULL;
			Tensor      *output_index = NULL;

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape0);
			TensorShapeUtils::MakeShape(dims_1d, 1, &outshape1);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_box));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_classes));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_index));

			auto obox     = output_box->template flat<T>();
			auto oclasses = output_classes->template flat<int32>();
			auto oindex   = output_index->template flat<int32>();

			for(int i=0,j=0; i<data_nr; ++i) {
				if(!keep_mask[i]) continue;
				auto box = bottom_box_flat.data()+i*4;
				std::copy(box,box+4,obox.data()+4*j);
				oclasses(j) = bottom_classes_flat(i);
				oindex(j) = i;
				++j;
			}
		}
        template<typename TM>
        int get_group(const TM& group_data,int label)
        {
            for(auto i=0; i<group_data.dimension(0); ++i) {
                if((label>=group_data(i,0)) && (label<=group_data(i,1)))
                    return i;
            }
            return -1;
        }
	private:
		float threshold    = 0.2;
};
REGISTER_KERNEL_BUILDER(Name("GroupBoxesNms").Device(DEVICE_CPU).TypeConstraint<float>("T"), GroupBoxesNmsOp<CPUDevice, float>);
/*
 * 数据不需要排序
 * bottom_box:[X,4](ymin,xmin,ymax,xmax)
 * classes:[X]
 * confidence:[X]
 * output_box:[Y,4]
 * output_classes:[Y]
 * output_index:[Y]
 * 输出时的相对位置不能改变
 */
REGISTER_OP("BoxesSoftNms")
    .Attr("T: {float, double,int32}")
	.Attr("threshold:float")
	.Attr("classes_wise:bool")
	.Attr("delta:float")
    .Input("bottom_box: T")
    .Input("classes: int32")
    .Input("confidence:float")
	.Output("output_box:T")
	.Output("output_classes:int32")
	.Output("output_index:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			c->set_output(0, c->Matrix(-1, 4));
			c->set_output(1, c->Vector(-1));
			c->set_output(2, c->Vector(-1));
			return Status::OK();
			});

template <typename Device, typename T>
class BoxesSoftNmsOp: public OpKernel {
    public:
        struct InterData
        {
            int index;
            float score;
            bool operator<(const InterData& v)const {
                return score<v.score;
            }
        };
        explicit BoxesSoftNmsOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold));
            OP_REQUIRES_OK(context, context->GetAttr("classes_wise", &classes_wise));
            OP_REQUIRES_OK(context, context->GetAttr("delta", &delta));
        }

        void Compute(OpKernelContext* context) override
        {
            const Tensor &bottom_box          = context->input(0);
            const Tensor &bottom_classes      = context->input(1);
            const Tensor &confidence          = context->input(2);
            auto          bottom_box_flat     = bottom_box.flat<T>();
            auto          bottom_classes_flat = bottom_classes.flat<int32>();
            auto          confidence_flat     = confidence.flat<float>();

            OP_REQUIRES(context, bottom_box.dims() == 2, errors::InvalidArgument("box data must be 2-dimensional"));
            OP_REQUIRES(context, bottom_classes.dims() == 1, errors::InvalidArgument("classes data must be 1-dimensional"));
            OP_REQUIRES(context, confidence.dims() == 1, errors::InvalidArgument("confidence data must be 1-dimensional"));

            const auto   data_nr           = bottom_box.dim_size(0);
            vector<InterData> set_D(data_nr,InterData({0,0.0f}));
            vector<InterData> set_B;
            const auto   loop_end          = data_nr-1;

            for(auto i=0; i<data_nr; ++i) {
                set_D[i].index = i;
                set_D[i].score = confidence_flat.data()[i];
            }
            set_B.reserve(data_nr);

            for(auto i=0; i<data_nr; ++i) {
                auto it = max_element(set_D.begin(),set_D.end());
                if(it->score<threshold)
                    break;
                auto M = *it;
                set_D.erase(it);
                set_B.push_back(M);
                const auto index = M.index;
                const auto iclass = bottom_classes_flat(index);
                for(auto& data:set_D) {
                    const auto j = data.index;
                    if(classes_wise && (bottom_classes_flat(j) != iclass)) continue;
                    const auto iou = bboxes_jaccard(bottom_box_flat.data()+index*4,bottom_box_flat.data()+j*4);
                    if(iou>1e-2)
                        data.score *= exp(-iou*iou/delta);
                }
            }
            sort(set_B.begin(),set_B.end(),[](const InterData& lhv, const InterData& rhv){ return lhv.index<rhv.index;});
            const auto out_size = set_B.size();
            int dims_2d[2] = {int(out_size),4};
            int dims_1d[1] = {int(out_size)};
            TensorShape  outshape0;
            TensorShape  outshape1;
            Tensor      *output_box         = NULL;
            Tensor      *output_classes     = NULL;
            Tensor      *output_index = NULL;

            TensorShapeUtils::MakeShape(dims_2d, 2, &outshape0);
            TensorShapeUtils::MakeShape(dims_1d, 1, &outshape1);

            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_box));
            OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_classes));
            OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_index));

            auto obox     = output_box->template flat<T>();
            auto oclasses = output_classes->template flat<int32>();
            auto oindex   = output_index->template flat<int32>();

            for(auto j=0; j<out_size; ++j) {
                const auto i = set_B[j].index;
                auto box = bottom_box_flat.data()+i*4;
                std::copy(box,box+4,obox.data()+4*j);
                oclasses(j) = bottom_classes_flat(i);
                oindex(j) = i;
            }
        }
    private:
        float threshold    = 0.2;
        float delta        = 2.0;
        bool  classes_wise = true;
};
REGISTER_KERNEL_BUILDER(Name("BoxesSoftNms").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesSoftNmsOp<CPUDevice, float>);
/*
 * 与BoxesNms的主要区别为BoxesNmsNr使用输出人数来进行处理
 * 要求数据已经按重要呈度从0到data_nr排好了序(从大到小）
 * bottom_box:[X,4](ymin,xmin,ymax,xmax)
 * classes:[X]
 * output_box:[Y,4]
 * output_classes:[Y]
 * output_indices:[X]
 * 输出时的相对位置不能改变
 * 程序会自动改变threshold的方式来使输出box的数量为k个
 */
REGISTER_OP("BoxesNmsNr")
    .Attr("T: {float, double,int32}")
	.Attr("classes_wise:bool")
	.Attr("k:int")
	.Attr("max_loop:int")
    .Input("bottom_box: T")
    .Input("classes:int32")
	.Output("output_box:T")
	.Output("output_classes:int32")
	.Output("output_index:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			int k = 0;
			c->GetAttr("k",&k);
			c->set_output(0, c->Matrix(k, 4));
			c->set_output(1, c->Vector(k));
			c->set_output(2, c->Vector(k));
			return Status::OK();
			});

template <typename Device, typename T>
class BoxesNmsNrOp: public OpKernel {
	public:
		explicit BoxesNmsNrOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("classes_wise", &classes_wise_));
			OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
			OP_REQUIRES_OK(context, context->GetAttr("max_loop", &max_loop_));
            if(k_ <= 0) k_ = 8;
            if(max_loop_ <= 0) max_loop_ = 4;
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &bottom_box          = context->input(0);
			const Tensor &bottom_classes      = context->input(1);
			auto          bottom_box_flat     = bottom_box.flat<T>();
			auto          bottom_classes_flat = bottom_classes.flat<int32>();

			OP_REQUIRES(context, bottom_box.dims() == 2, errors::InvalidArgument("box data must be 2-dimensional"));
			OP_REQUIRES(context, bottom_classes.dims() == 1, errors::InvalidArgument("classes data must be 1-dimensional"));

			const auto   data_nr               = bottom_box.dim_size(0);
			vector<bool> keep_mask(data_nr,true);
			vector<bool> old_keep_mask(data_nr,true);
			const auto   loop_end              = data_nr-1;
			int          old_nr                = 0;

            auto loop_fn = [&](float threshold) {
                if(old_nr>k_)
				    std::swap(old_keep_mask,keep_mask);
				fill(keep_mask.begin(),keep_mask.end(),true);
                auto keep_nr = keep_mask.size();
				for(auto i=0; i<loop_end; ++i) {
					if(!keep_mask.at(i)) continue;
					const auto iclass = bottom_classes_flat(i);
					for(auto j=i+1; j<data_nr; ++j) {
						if(classes_wise_ && (bottom_classes_flat(j) != iclass)) continue;
						if(!keep_mask.at(j)) continue;
						if(bboxes_jaccard(bottom_box_flat.data()+i*4,bottom_box_flat.data()+j*4) < threshold) continue;
						keep_mask[j] = false;
						--keep_nr;
					}
				}
				//cout<<"keep nr:"<<keep_nr<<", threshold "<<threshold<<endl;
			   //keep_nr = count(keep_mask.begin(),keep_mask.end(),true);
               return keep_nr;
            };

            auto threshold_low   = 0.0;
            auto threshold_hight = 1.0;

            for(auto i=0; i<max_loop_; ++i) {
                auto threshold = (threshold_low+threshold_hight)/2.0;
                auto nr = loop_fn(threshold);
                old_nr = nr;
                if(nr == k_) break;
                if(nr>k_)
                    threshold_hight = threshold;
                else
                    threshold_low = threshold;
            }

			auto out_size = count(keep_mask.begin(),keep_mask.end(),true);

            if(k_>data_nr) k_ = data_nr;

            if(out_size<k_) {
                auto delta = k_-out_size;
                for(auto it=keep_mask.begin(); it!=keep_mask.end(); ++it) 
                    if((*it) == false) {
                        (*it) = true;
                        --delta;
                        if(0 == delta) break;
                    }
                out_size = count(keep_mask.begin(),keep_mask.end(),true);
            } else if(out_size>k_) {
                auto nr = out_size-k_;
                for(auto it = keep_mask.rbegin(); it!=keep_mask.rend(); ++it) {
                    if((*it) == true) {
                        *it = false;
                        --nr;
                        if(0 == nr) break;
                    }
                }
            }
            out_size = k_;

			int dims_2d[2] = {int(out_size),4};
			int dims_1d[1] = {int(out_size)};
			TensorShape  outshape0;
			TensorShape  outshape1;
			Tensor      *output_box         = NULL;
			Tensor      *output_classes     = NULL;
			Tensor      *output_index = NULL;

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape0);
			TensorShapeUtils::MakeShape(dims_1d, 1, &outshape1);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_box));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_classes));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_index));

			auto obox     = output_box->template flat<T>();
			auto oclasses = output_classes->template flat<int32>();
			auto oindex   = output_index->template flat<int32>();
			int  j        = 0;
			int  i        = 0;

			for(i=0,j=0; i<data_nr; ++i) {
				if(!keep_mask[i]) continue;
				auto box = bottom_box_flat.data()+i*4;
				std::copy(box,box+4,obox.data()+4*j);
				oclasses(j) = bottom_classes_flat(i);
				oindex(j) = i;
				++j;
				if(j>=out_size) break;
			}
            if(j<out_size) {
				cout<<"out size = "<<out_size<<", in size = "<<data_nr<<", j= "<<j<<std::endl;
                auto i = data_nr-1;
                for(;j<out_size; ++j) {
                    auto box = bottom_box_flat.data()+i*4;
                    std::copy(box,box+4,obox.data()+4*j);
                    oclasses(j) = bottom_classes_flat(i);
                    oindex(j) = i;
                }
            }
		}
	private:
		bool  classes_wise_ = true;
		int   k_            = 0;
		int   max_loop_     = 4;
};
REGISTER_KERNEL_BUILDER(Name("BoxesNmsNr").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesNmsNrOp<CPUDevice, float>);
/*
 * 与BoxesNmsNr的主要区别, 使用输入的theshold进行处理，选靠前的nr个boxes, 如果NMS后没有足够的boxes部分被删除的boxes会重新加入进来
 * 要求数据已经按重要呈度从0到data_nr排好了序(从大到小）
 * bottom_box:[X,4](ymin,xmin,ymax,xmax)
 * classes:[X]
 * output_box:[Y,4]
 * output_classes:[Y]
 * output_indices:[X]
 * 输出时的相对位置不能改变
 * 程序会自动改变threshold的方式来使输出box的数量为k个
 */
REGISTER_OP("BoxesNmsNr2")
    .Attr("T: {float, double,int32}")
	.Attr("classes_wise:bool")
	.Attr("k:int")
	.Attr("threshold:float")
    .Input("bottom_box: T")
    .Input("classes:int32")
	.Output("output_box:T")
	.Output("output_classes:int32")
	.Output("output_index:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			int k = 0;
			c->GetAttr("k",&k);
			c->set_output(0, c->Matrix(k, 4));
			c->set_output(1, c->Vector(k));
			c->set_output(2, c->Vector(k));
			return Status::OK();
			});

template <typename Device, typename T>
class BoxesNmsNr2Op: public OpKernel {
	public:
		explicit BoxesNmsNr2Op(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("classes_wise", &classes_wise_));
			OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold_));
			OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
            if(k_ <= 0) k_ = 1;
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &bottom_box          = context->input(0);
			const Tensor &bottom_classes      = context->input(1);
			auto          bottom_box_flat     = bottom_box.flat<T>();
			auto          bottom_classes_flat = bottom_classes.flat<int32>();

			OP_REQUIRES(context, bottom_box.dims() == 2, errors::InvalidArgument("box data must be 2-dimensional"));
			OP_REQUIRES(context, bottom_classes.dims() == 1, errors::InvalidArgument("classes data must be 1-dimensional"));

			const auto   data_nr               = bottom_box.dim_size(0);
			vector<bool> keep_mask(data_nr,true);
			const auto   loop_end              = data_nr-1;

            auto loop_fn = [&](float threshold) {
                auto keep_nr = keep_mask.size();
				for(auto i=0; i<loop_end; ++i) {
					if(!keep_mask.at(i)) continue;
					const auto iclass = bottom_classes_flat(i);
					for(auto j=i+1; j<data_nr; ++j) {
						if(classes_wise_ && (bottom_classes_flat(j) != iclass)) continue;
						if(!keep_mask.at(j)) continue;
						if(bboxes_jaccard(bottom_box_flat.data()+i*4,bottom_box_flat.data()+j*4) < threshold) continue;
						keep_mask[j] = false;
						--keep_nr;
					}
				}
               return keep_nr;
            };

            if(k_>data_nr) k_ = data_nr;

            auto nr = loop_fn(threshold_);
			auto out_size = count(keep_mask.begin(),keep_mask.end(),true);

            if(out_size<k_) {
                auto delta = k_-out_size;
                for(auto it=keep_mask.begin(); it!=keep_mask.end(); ++it) 
                    if((*it) == false) {
                        (*it) = true;
                        --delta;
                        if(0 == delta) break;
                    }
            } else if(out_size>k_) {
                auto nr = out_size-k_;
                for(auto it = keep_mask.rbegin(); it!=keep_mask.rend(); ++it) {
                    if((*it) == true) {
                        *it = false;
                        --nr;
                        if(0 == nr) break;
                    }
                }
            }
            out_size = k_;

			int dims_2d[2] = {int(out_size),4};
			int dims_1d[1] = {int(out_size)};
			TensorShape  outshape0;
			TensorShape  outshape1;
			Tensor      *output_box         = NULL;
			Tensor      *output_classes     = NULL;
			Tensor      *output_index = NULL;

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape0);
			TensorShapeUtils::MakeShape(dims_1d, 1, &outshape1);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_box));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_classes));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_index));

			auto obox     = output_box->template flat<T>();
			auto oclasses = output_classes->template flat<int32>();
			auto oindex   = output_index->template flat<int32>();
			int  j        = 0;
			int  i        = 0;

			for(i=0,j=0; i<data_nr; ++i) {
				if(!keep_mask[i]) continue;
				auto box = bottom_box_flat.data()+i*4;
				std::copy(box,box+4,obox.data()+4*j);
				oclasses(j) = bottom_classes_flat(i);
				oindex(j) = i;
				++j;
				if(j>=out_size) break;
			}
            if(j<out_size) {
				cout<<"out size = "<<out_size<<", in size = "<<data_nr<<", j= "<<j<<std::endl;
                auto i = data_nr-1;
                for(;j<out_size; ++j) {
                    auto box = bottom_box_flat.data()+i*4;
                    std::copy(box,box+4,obox.data()+4*j);
                    oclasses(j) = bottom_classes_flat(i);
                    oindex(j) = i;
                }
            }
		}
	private:
		bool  classes_wise_ = true;
		float threshold_    = 0.0;
		int   k_            = 0;
};
REGISTER_KERNEL_BUILDER(Name("BoxesNmsNr2").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesNmsNr2Op<CPUDevice, float>);
/*
 * 要求数据已经按重要呈度从0到data_nr排好了序(从大到小)
 * 通常数据先经过NMS处理过
 * 用于计算其余k个(或者所有)的Box与当前Box的交叉面积点当前Box的百分比，如果所占百分比大于threshold则删除当前box
 * 计算交叉面积时仅考虑同一个类别的box
 * k:需要考虑的box数目，如果为0表示考虑所有的box与当前box的百分比，否则仅考虑与当前box交叉面积最大的num个box
 * threshold:阀值
 * bottom_box:[X,4](ymin,xmin,ymax,xmax)
 * classes:[X]
 * output_box:[Y,4]
 * output_classes:[Y]
 * output_index:[Y]
 */
REGISTER_OP("EBoxesNms")
    .Attr("T: {float, double,int32}")
	.Attr("threshold:float")
	.Attr("k:int")
    .Input("bottom_box: T")
    .Input("classes: int32")
	.Output("output_box:T")
	.Output("output_classes:int32")
	.Output("output_index:int32")
    .SetShapeFn([](shape_inference::InferenceContext* c){
        auto shape0 = c->Matrix(c->UnknownDim(),4);
        auto shape1 = c->Vector(c->UnknownDim());
        c->set_output(0,shape0);
        c->set_output(1,shape1);
		return Status::OK();
    });

template <typename Device, typename T>
class EBoxesNmsOp: public OpKernel {
	public:
		explicit EBoxesNmsOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold));
			OP_REQUIRES_OK(context, context->GetAttr("k", &k));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &bottom_box          = context->input(0);
			const Tensor &bottom_classes      = context->input(1);
			auto          bottom_box_flat     = bottom_box.flat<T>();
			auto          bottom_classes_flat = bottom_classes.flat<int32>();

			OP_REQUIRES(context, bottom_box.dims() == 2, errors::InvalidArgument("box data must be 2-dimensional"));
			OP_REQUIRES(context, bottom_classes.dims() == 1, errors::InvalidArgument("classes data must be 1-dimensional"));

			const auto   data_nr           = bottom_box.dim_size(0);
			vector<bool> keep_mask(data_nr   ,true);
			const auto   loop_end          = data_nr-1;
			float        total_jaccard     = 0.0f;

			for(auto i=data_nr-1; i>0; --i) {
                vector<float> jaccards;
                jaccards.reserve(i);
				const auto iclass = bottom_classes_flat(i);
                for(auto j=0; j<i; ++j) {
                    if(bottom_classes_flat(j) != iclass) continue;
                    const auto jaccard = bboxes_jaccard_of_box0(bottom_box_flat.data()+i*4,bottom_box_flat.data()+j*4);
                    if(jaccard<1E-6) continue;
                    jaccards.push_back(jaccard);
                }
                if(k==0) {
                    total_jaccard = std::accumulate(jaccards.begin(),jaccards.end(),0.0);
                } else {
                    std::sort(jaccards.begin(),jaccards.end(),std::greater<float>());
                    auto num = std::min<int>(k,jaccards.size());
                    total_jaccard = std::accumulate(jaccards.begin(),std::next(jaccards.begin(),num),0.0);
                }
                if(total_jaccard>=threshold)
                    keep_mask[i] = false;
			}

			const auto   out_size       = count(keep_mask.begin(),keep_mask.end(),true);
			int          dims_2d[2]     = {int(out_size),4};
			int          dims_1d[1]     = {int(out_size)};
			TensorShape  outshape0;
			TensorShape  outshape1;
			Tensor      *output_box     = NULL;
			Tensor      *output_classes = NULL;
			Tensor      *output_index   = NULL;

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape0);
			TensorShapeUtils::MakeShape(dims_1d, 1, &outshape1);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_box));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_classes));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_index));

			auto obox     = output_box->template flat<T>();
			auto oclasses = output_classes->template flat<int32>();
			auto oindex   = output_index->template flat<int32>();

			for(int i=0,j=0; i<data_nr; ++i) {
				if(!keep_mask[i]) continue;
				auto box = bottom_box_flat.data()+i*4;
				copy_box(box,obox.data()+4*j);
				oclasses(j) = bottom_classes_flat(i);
				oindex(j) = i;
				++j;
			}
		}
	private:
		float threshold = 0.0f;
		int   k         = 0;
};
REGISTER_KERNEL_BUILDER(Name("EBoxesNms").Device(DEVICE_CPU).TypeConstraint<float>("T"), EBoxesNmsOp<CPUDevice, float>);
/*
 * prio_scaling:[4]
 * bottom_boxes:[1,X,4]/[batch_size,X,4](ymin,xmin,ymax,xmax) 候选box,相对坐标
 * bottom_gboxes:[batch_size,Y,4](ymin,xmin,ymax,xmax)ground truth box相对坐标
 * bottom_glabels:[batch_size,Y] 0为背景
 * bottom_glength:[batch_size] 为每一个batch中gboxes的有效数量
 * output_boxes:[batch_size,X,4] regs(cy,cx,h,w)
 * output_labels:[batch_size,X], 当前anchorbox的标签，背景为0,不为背景时为相应最大jaccard得分
 * output_scores:[batch_size,X], 当前anchorbox与groundtruthbox的jaccard得分，当jaccard得分高于threshold时就不为背影
 * output_remove_indict:[batch_size,X], anchorbox是否有效(一般为iou处理中间部分的无效)
 * output_indict:[batch_size,X], 当anchorbox有效时，与它对应的gboxes(从0开始)序号,无效时为-1
 */
REGISTER_OP("BoxesEncode")
    .Attr("T: {float,double,int32,int64}")
	.Attr("pos_threshold:float")
	.Attr("neg_threshold:float")
 	.Attr("prio_scaling: list(float)")
    .Input("bottom_boxes: T")
    .Input("bottom_gboxes: T")
    .Input("bottom_glabels: int32")
    .Input("bottom_glength: int32")
	.Output("output_boxes:T")
	.Output("output_labels:int32")
	.Output("output_scores:T")
	.Output("remove_indict:bool")
	.Output("indict:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            const auto input_shape0 = c->input(0);
            const auto input_shape1 = c->input(1);
            const auto batch_size = c->Dim(input_shape1,0);
            const auto boxes_nr  = c->Dim(input_shape0,1);
            auto shape0 = c->MakeShape({batch_size,boxes_nr,4});
            auto shape1 = c->MakeShape({batch_size,boxes_nr});

			c->set_output(0, shape0);
            for(auto i=1; i<5; ++i)
			    c->set_output(i, shape1);
			return Status::OK();
			});

template <typename Device, typename T>
class BoxesEncodeOp: public OpKernel {
	public:
		explicit BoxesEncodeOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("pos_threshold", &pos_threshold));
			OP_REQUIRES_OK(context, context->GetAttr("neg_threshold", &neg_threshold));
			OP_REQUIRES_OK(context, context->GetAttr("prio_scaling", &prio_scaling));
			OP_REQUIRES(context, prio_scaling.size() == 4, errors::InvalidArgument("prio scaling data must be shape[4]"));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &_bottom_boxes   = context->input(0);
			const Tensor &_bottom_gboxes  = context->input(1);
			const Tensor &_bottom_glabels = context->input(2);
			const Tensor &_bottom_gsize   = context->input(3);
			auto          bottom_boxes    = _bottom_boxes.template tensor<T,3>();
			auto          bottom_gboxes   = _bottom_gboxes.template tensor<T,3>();
			auto          bottom_glabels  = _bottom_glabels.template tensor<int,2>();
			auto          bottom_gsize    = _bottom_gsize.template tensor<int,1>();
			const auto    batch_size      = _bottom_gboxes.dim_size(0);
			const auto    data_nr         = _bottom_boxes.dim_size(1);

			OP_REQUIRES(context, _bottom_boxes.dims() == 3, errors::InvalidArgument("box data must be 3-dimensional"));
			OP_REQUIRES(context, _bottom_gboxes.dims() == 3, errors::InvalidArgument("box data must be 3-dimensional"));
			OP_REQUIRES(context, _bottom_glabels.dims() == 2, errors::InvalidArgument("labels data must be 2-dimensional"));
			OP_REQUIRES(context, _bottom_gsize.dims() == 1, errors::InvalidArgument("gsize data must be 1-dimensional"));

			int           dims_3d[3]            = {int(batch_size),int(data_nr),4};
			int           dims_2d[2]            = {int(batch_size),int(data_nr)};
			TensorShape   outshape0;
			TensorShape   outshape1;
			Tensor       *output_boxes          = NULL;
			Tensor       *output_labels         = NULL;
			Tensor       *output_scores         = NULL;
			Tensor       *output_remove_indict  = NULL;
			Tensor       *output_indict         = NULL;

			TensorShapeUtils::MakeShape(dims_3d, 3, &outshape0);
			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape1);


			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_boxes));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_labels));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_scores));
			OP_REQUIRES_OK(context, context->allocate_output(3, outshape1, &output_remove_indict));
			OP_REQUIRES_OK(context, context->allocate_output(4, outshape1, &output_indict));

			auto output_boxes_tensor          =  output_boxes->template tensor<T,3>();
			auto output_labels_tensor         =  output_labels->template tensor<int,2>();
			auto output_scores_tensor         =  output_scores->template tensor<T,2>();
			auto output_remove_indict_tensor  =  output_remove_indict->template tensor<bool,2>();
			auto output_indict_tensor         =  output_indict->template tensor<int,2>();

            BoxesEncodeUnit<T> encode_unit(pos_threshold,neg_threshold,prio_scaling);
            for(auto i=0; i<batch_size; ++i) {

                auto size     = bottom_gsize(i);
                auto boxes    = bottom_boxes.chip(bottom_boxes.dimension(0)==batch_size?i:0,0);
                auto _gboxes  = bottom_gboxes.chip(i,0);
                auto _glabels = bottom_glabels.chip(i,0);
                Eigen::array<long,2> offset={0,0};
                Eigen::array<long,2> extents={size,4};
                Eigen::array<long,1> offset1={0};
                Eigen::array<long,1> extents1={size};
                auto gboxes             = _gboxes.slice(offset,extents);
                auto glabels            = _glabels.slice(offset1,extents1);
                auto out_boxes          = output_boxes_tensor.chip(i,0);
                auto out_labels         = output_labels_tensor.chip(i,0);
                auto out_scores         = output_scores_tensor.chip(i,0);
                auto out_remove_indices = output_remove_indict_tensor.chip(i,0);
                auto out_indices        = output_indict_tensor.chip(i,0);
                auto res                = encode_unit(boxes,gboxes,glabels);

                out_boxes           =  std::get<0>(res);
                out_labels          =  std::get<1>(res);
                out_scores          =  std::get<2>(res);
                out_remove_indices  =  std::get<3>(res);
                out_indices         =  std::get<4>(res);
            }
		}
	private:
		float         pos_threshold;
		float         neg_threshold;
		vector<float> prio_scaling;
};
REGISTER_KERNEL_BUILDER(Name("BoxesEncode").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesEncodeOp<CPUDevice, float>);
/*
 * prio_scaling:[4]
 * bottom_boxes:[X,4](ymin,xmin,ymax,xmax) 候选box,相对坐标
 * bottom_gboxes:[Y,4](ymin,xmin,ymax,xmax)ground truth box相对坐标
 * bottom_glabels:[Y] 0为背景
 * output_boxes:[X,4] regs(cy,cx,h,w)
 * output_labels:[X], 当前anchorbox的标签，背景为0,不为背景时为相应最大jaccard得分
 * output_scores:[X], 当前anchorbox与groundtruthbox的jaccard得分，当jaccard得分高于threshold时就不为背影
 */
REGISTER_OP("BoxesEncode1")
    .Attr("T: {float,double,int32,int64}")
	.Attr("pos_threshold:float")
	.Attr("neg_threshold:float")
 	.Attr("prio_scaling: list(float)")
    .Input("bottom_boxes: T")
    .Input("bottom_gboxes: T")
    .Input("bottom_glabels: int32")
	.Output("output_boxes:T")
	.Output("output_labels:int32")
	.Output("output_scores:T")
	.Output("remove_indict:bool")
    .SetShapeFn([](shape_inference::InferenceContext* c){
            auto shape0 = c->input(0);
            auto shape1 = c->Vector(c->Dim(shape0,0));

            c->set_output(0,shape0);

            for(auto i=1; i<4; ++i) c->set_output(i,shape1);

            return Status::OK();
            });

template <typename Device, typename T>
class BoxesEncode1Op: public OpKernel {
	public:
        struct IOUIndex{
            int index;
            float iou;
        };
		explicit BoxesEncode1Op(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("pos_threshold", &pos_threshold));
			OP_REQUIRES_OK(context, context->GetAttr("neg_threshold", &neg_threshold));
			OP_REQUIRES_OK(context, context->GetAttr("prio_scaling", &prio_scaling));
			OP_REQUIRES(context, prio_scaling.size() == 4, errors::InvalidArgument("prio scaling data must be shape[4]"));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &_bottom_boxes   = context->input(0);
			const Tensor &_bottom_gboxes  = context->input(1);
			const Tensor &_bottom_glabels = context->input(2);
			auto          bottom_boxes    = _bottom_boxes.template tensor<T,2>();
			auto          bottom_gboxes   = _bottom_gboxes.template tensor<T,2>();
			auto          bottom_glabels  = _bottom_glabels.template tensor<int,1>();
			const auto    data_nr         = _bottom_boxes.dim_size(0);

			OP_REQUIRES(context, _bottom_boxes.dims() == 2, errors::InvalidArgument("box data must be 2-dimensional"));
			OP_REQUIRES(context, _bottom_gboxes.dims() == 2, errors::InvalidArgument("box data must be 2-dimensional"));
			OP_REQUIRES(context, _bottom_glabels.dims() == 1, errors::InvalidArgument("labels data must be 1-dimensional"));

			int           dims_2d[2]            = {int(data_nr),4};
			int           dims_1d[1]            = {int(data_nr)};
			TensorShape   outshape0;
			TensorShape   outshape1;
			Tensor       *output_boxes          = NULL;
			Tensor       *output_labels         = NULL;
			Tensor       *output_scores         = NULL;
			Tensor       *output_remove_indict  = NULL;
			vector<IOUIndex>   iou_indexs(data_nr,IOUIndex({-1,0.0})); //默认box不与任何ground truth box相交，iou为0

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape0);
			TensorShapeUtils::MakeShape(dims_1d, 1, &outshape1);


			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_boxes));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_labels));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_scores));
			OP_REQUIRES_OK(context, context->allocate_output(3, outshape1, &output_remove_indict));
			auto output_boxes_tensor         = output_boxes->template tensor<T,2>();
			auto output_labels_tensor        = output_labels->template tensor<int,1>();
			auto output_scores_tensor        = output_scores->template tensor<T,1>();
			auto output_remove_indict_tensor = output_remove_indict->template tensor<bool,1>();

			BoxesEncodeUnit<T> encode_unit(pos_threshold,neg_threshold,prio_scaling);
			auto &boxes              = bottom_boxes;
			auto &gboxes             = bottom_gboxes;
			auto &glabels            = bottom_glabels;
			auto &out_boxes          = output_boxes_tensor;
			auto &out_labels         = output_labels_tensor;
			auto &out_scores         = output_scores_tensor;
			auto &out_remove_indices = output_remove_indict_tensor;
			auto  res                = encode_unit(boxes,gboxes,glabels);

			out_boxes           =  std::get<0>(res);
			out_labels          =  std::get<1>(res);
			out_scores          =  std::get<2>(res);
			out_remove_indices  =  std::get<3>(res);
		}
	private:
		float         pos_threshold;
		float         neg_threshold;
		vector<float> prio_scaling;
};

REGISTER_OP("BoxesEncode1Grad")
    .Attr("T: {float,double,int32,int64}")
	.Attr("threshold:float")
 	.Attr("prio_scaling: list(float)")
    .Input("bottom_boxes: T")
    .Input("bottom_gboxes: T")
    .Input("bottom_glabels: int32")
    .Input("grad: T")
	.Output("output:T");
template <typename Device, typename T>
class BoxesEncode1GradOp: public OpKernel {
	public:
		explicit BoxesEncode1GradOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold));
			OP_REQUIRES_OK(context, context->GetAttr("prio_scaling", &prio_scaling));
			OP_REQUIRES(context, prio_scaling.size() == 4, errors::InvalidArgument("prio scaling data must be shape[4]"));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &bottom_boxes        = context->input(0);

			OP_REQUIRES(context, bottom_boxes.dims() == 2, errors::InvalidArgument("box data must be 2-dimensional"));

			TensorShape  outshape0   = bottom_boxes.shape();
			Tensor      *output_grad = nullptr;

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_grad));

			auto output_grad_flat = output_grad->template flat<T>();
			auto num_elements     = output_grad->NumElements();
			cout<<"num elements:"<<num_elements<<","<<bottom_boxes.dims()<<","<<bottom_boxes.dim_size(0)<<","<<bottom_boxes.dim_size(1)<<endl;

            for(auto i=0; i<num_elements; ++i)
                output_grad_flat(i) = 0.0f;
		}
	private:
		float threshold;
		vector<float> prio_scaling;
};
REGISTER_KERNEL_BUILDER(Name("BoxesEncode1").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesEncode1Op<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("BoxesEncode1Grad").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesEncode1GradOp<CPUDevice, float>);
/*
 * bottom_boxes:[Nr,4](ymin,xmin,ymax,xmax) proposal box,相对坐标
 * bottom_regs:[Nr,4],(y,x,h,w)
 * prio_scaling:[4]
 * output:[Nr,4] 相对坐标(ymin,xmin,ymax,xmax)
 */
REGISTER_OP("DecodeBoxes1")
    .Attr("T: {float, double}")
	.Attr("prio_scaling:list(float)")
    .Input("bottom_boxes: T")
    .Input("bottom_regs: T")
	.Output("output:T")
    .SetShapeFn(shape_inference::UnchangedShape);

template <typename Device, typename T>
class DecodeBoxes1Op: public OpKernel {
	public:
		explicit DecodeBoxes1Op(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context,
					context->GetAttr("prio_scaling", &prio_scaling));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &bottom_boxes       = context->input(0);
			const Tensor &bottom_regs        = context->input(1);
			auto          bottom_regs_flat   = bottom_regs.flat<T>();
			auto          bottom_boxes_flat  = bottom_boxes.flat<T>();

			OP_REQUIRES(context, bottom_regs.dims() == 2, errors::InvalidArgument("regs data must be 2-dimensional"));
			OP_REQUIRES(context, bottom_boxes.dims() == 2, errors::InvalidArgument("pos data must be 2-dimensional"));
			OP_REQUIRES(context, bottom_boxes.dim_size(0)==bottom_regs.dim_size(0), errors::InvalidArgument("First dim size must be equal."));
			OP_REQUIRES(context, bottom_boxes.dim_size(1)==4, errors::InvalidArgument("Boxes second dim size must be 4."));
			OP_REQUIRES(context, bottom_regs.dim_size(1)==4, errors::InvalidArgument("Regs second dim size must be 4."));
			const auto nr = bottom_regs.dim_size(0);

			TensorShape output_shape = bottom_regs.shape();
			// Create output tensors
			Tensor* output_tensor = NULL;
			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

			auto output = output_tensor->template flat<T>();
			for (auto b = 0; b < nr; ++b) {
				const auto base_offset = b *4;
				const auto regs_data   = bottom_regs_flat.data()+base_offset;
				const auto box_data    = bottom_boxes_flat.data()+base_offset;
				float      y;
				float      x;
				float      href;
				float      wref;

				std::tie(y,x,href,wref) = box_minmax_to_cxywh(box_data);

				auto cy          = boost::algorithm::clamp<T>(regs_data[0]*prio_scaling[0],-10.0f,10.0f)*href+y;
				auto cx          = boost::algorithm::clamp<T>(regs_data[1]*prio_scaling[1],-10.0f,10.0f)*wref+x;
				auto h           = href *exp(boost::algorithm::clamp<T>(regs_data[2]*prio_scaling[2],-10.0,10.0));
				auto w           = wref *exp(boost::algorithm::clamp<T>(regs_data[3]*prio_scaling[3],-10.0,10.0));
				auto output_data = output.data() + base_offset;

				std::tie(output_data[0],output_data[1],output_data[2],output_data[3]) = box_cxywh_to_minmax(cy,cx,h,w);
				transform(output_data,output_data+4,output_data,[](T& v) { return boost::algorithm::clamp<T>(v,0.0,1.0);});
				if(output_data[0]>output_data[2]) 
					output_data[2] = output_data[0];
				if(output_data[1]>output_data[3])
					output_data[3] = output_data[1];
			}
		}
	private:
		std::vector<float> prio_scaling;
};
REGISTER_KERNEL_BUILDER(Name("DecodeBoxes1").Device(DEVICE_CPU).TypeConstraint<float>("T"), DecodeBoxes1Op<CPUDevice, float>);
/*
 * bottom_boxes:[Nr,4](ymin,xmin,ymax,xmax) proposal box
 * width:float
 * height:float
 * output:[Nr,4] 相对坐标(ymin,xmin,ymax,xmax)
 */
REGISTER_OP("BoxesRelativeToAbsolute")
    .Attr("T: {float, double}")
	.Attr("width: int")
	.Attr("height: int")
    .Input("bottom_boxes: T")
	.Output("output:T")
    .SetShapeFn(shape_inference::UnchangedShape);

template <typename Device, typename T>
class BoxesRelativeToAbsoluteOp: public OpKernel {
	public:
		explicit BoxesRelativeToAbsoluteOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("width", &width_));
			OP_REQUIRES_OK(context, context->GetAttr("height", &height_));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &bottom_boxes      = context->input(0);
			auto          bottom_boxes_flat = bottom_boxes.flat<T>();

			OP_REQUIRES(context, bottom_boxes.dims() == 2, errors::InvalidArgument("boxes data must be 2-dimensional"));
			OP_REQUIRES(context, bottom_boxes.dim_size(1)==4, errors::InvalidArgument("Boxes second dim size must be 4."));
			const auto nr = bottom_boxes.dim_size(0);

			TensorShape output_shape = bottom_boxes.shape();
			// Create output tensors
			Tensor* output_tensor = NULL;
			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

			auto output = output_tensor->template flat<T>();
			auto shard = [this, &bottom_boxes_flat,&output]
					 (int64 start, int64 limit) {
						 for (int64 b = start; b < limit; ++b) {
							 const auto base_offset = b *4;
							 const auto box_data    = bottom_boxes_flat.data()+base_offset;
							 const auto output_data = output.data()+base_offset;
							 output_data[0] = box_data[0]*(height_-1)+0.5f;
							 output_data[1] = box_data[1]*(width_-1)+0.5f;
							 output_data[2] = box_data[2]*(height_-1)+0.5f;
							 output_data[3] = box_data[3]*(width_-1)+0.5f;
						 }
					 };

			const DeviceBase::CpuWorkerThreads& worker_threads =
				*(context->device()->tensorflow_cpu_worker_threads());
			Shard(worker_threads.num_threads, worker_threads.workers,
					nr, 1000, shard);
		}
	private:
		int width_;
		int height_;
};
REGISTER_KERNEL_BUILDER(Name("BoxesRelativeToAbsolute").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesRelativeToAbsoluteOp<CPUDevice, float>);
/*
 * 在image中剪切出一个ref_box定义的区域，同时对原来在image中的boxes进行处理，如果boxes与剪切区域jaccard得分小于threshold的会被删除
 * ref_box:shape=[4],[ymin,xmin,ymax,xmax] 参考box,相对坐标
 * boxes:[Nr,4],(ymin,xmin,ymax,xmax),需要处理的box
 * threshold:阀值
 * output:[Y,4]
 */
REGISTER_OP("CropBoxes")
    .Attr("T: {float, double}")
	.Attr("threshold:float")
    .Input("ref_box: T")
    .Input("boxes: T")
	.Output("output:T")
	.Output("mask:bool")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			c->set_output(0, c->Matrix(-1, 4));
			c->set_output(1, c->Vector(c->Dim(c->input(1), 0)));
			return Status::OK();
			});

template <typename Device, typename T>
class CropBoxesOp: public OpKernel {
	public:
		explicit CropBoxesOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context,
					context->GetAttr("threshold", &threshold));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &ref_box      = context->input(0);
			const Tensor &boxes        = context->input(1);
			auto          ref_box_flat = ref_box.flat<T>();
			auto          boxes_flat   = boxes.flat<T>();
			const auto    nr           = boxes.dim_size(0);
			vector<int>   good_index;
			vector<bool>  good_mask(nr,false);

			OP_REQUIRES(context, ref_box.dims() == 1, errors::InvalidArgument("ref box must be 1-dimensional"));
			OP_REQUIRES(context, boxes.dims() == 2, errors::InvalidArgument("boxes must be 2-dimensional"));

            good_index.reserve(nr);

            for(int i=0; i<nr; ++i) {
                auto cur_box = boxes_flat.data()+4*i;
                if(bboxes_jaccard_of_box0(cur_box,ref_box_flat.data()) < threshold) continue;
                good_index.push_back(i);
				good_mask[i] = true;
            }
            const int   out_nr       = good_index.size();
            const int   dims_2d[]    = {out_nr,4};
            const int   dims_1d[]    = {int(nr)};
            TensorShape output_shape;
            TensorShape output_shape1;

			TensorShapeUtils::MakeShape(dims_2d, 2, &output_shape);
			TensorShapeUtils::MakeShape(dims_1d, 1, &output_shape1);

			Tensor *output_tensor = NULL;
			Tensor *output_mask   = NULL;

			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
			OP_REQUIRES_OK(context, context->allocate_output(1, output_shape1, &output_mask));

			auto output           = output_tensor->template flat<T>();
			auto output_mask_flat = output_mask->template flat<bool>();

			for(auto i=0; i<nr; ++i)
				output_mask_flat.data()[i] = good_mask[i];

            for(int i=0; i<out_nr; ++i) {
                cut_box(ref_box_flat.data(),boxes_flat.data()+good_index[i]*4,output.data()+i*4);
            }
		}
	private:
		float threshold=1.0;
};
REGISTER_KERNEL_BUILDER(Name("CropBoxes").Device(DEVICE_CPU).TypeConstraint<float>("T"), CropBoxesOp<CPUDevice, float>);

/*
 * 删除与边框的jaccard得分大于threshold的box
 * size:shape=[2],[h,w] 相对坐标
 * boxes:[Nr,4],(ymin,xmin,ymax,xmax),需要处理的box
 * threshold:阀值
 * output:[Y,4]
 */
REGISTER_OP("RemoveBoundaryBoxes")
    .Attr("T: {float, double}")
	.Attr("threshold:float")
    .Input("size: T")
    .Input("boxes: T")
	.Output("output:T")
	.Output("output_index:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			c->set_output(0, c->Matrix(c->UnknownDim(), 4));
			c->set_output(1, c->Vector(c->UnknownDim()));
			return Status::OK();
			});

template <typename Device, typename T>
class RemoveBoundaryBoxesOp: public OpKernel {
	public:
		explicit RemoveBoundaryBoxesOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context,
					context->GetAttr("threshold", &threshold));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &size         = context->input(0);
			const Tensor &boxes        = context->input(1);
			auto          size_flat    = size.flat<T>();
			auto          boxes_flat   = boxes.flat<T>();
			vector<int>   good_index;
			T             ref_boxes0[] = {0.0f,0.0f,1.0f,size_flat.data()[1]};
			T             ref_boxes1[] = {0.0f,0.0f,size_flat.data()[0],1.0f};
			T             ref_boxes2[] = {0.0f,1.0f-size_flat.data()[1],1.0f,1.0f};
			T             ref_boxes3[] = {1.0f-size_flat.data()[0],0.0f,1.0f,1.0f};

			OP_REQUIRES(context, size.dims() == 1, errors::InvalidArgument("size must be 1-dimensional"));
			OP_REQUIRES(context, boxes.dims() == 2, errors::InvalidArgument("boxes must be 2-dimensional"));

			const auto nr = boxes.dim_size(0);
            good_index.reserve(nr);

            for(int i=0; i<nr; ++i) {
                auto cur_box = boxes_flat.data()+4*i;
                if(bboxes_jaccard_of_box0(cur_box,ref_boxes0) > threshold) continue;
                if(bboxes_jaccard_of_box0(cur_box,ref_boxes1) > threshold) continue;
                if(bboxes_jaccard_of_box0(cur_box,ref_boxes2) > threshold) continue;
                if(bboxes_jaccard_of_box0(cur_box,ref_boxes3) > threshold) continue;
                good_index.push_back(i);
            }
            const int   out_nr       = good_index.size();
            const int   dims_2d[]    = {out_nr,4};
            const int   dims_1d[]    = {out_nr};
            TensorShape output_shape;
            TensorShape output_shape1;

			TensorShapeUtils::MakeShape(dims_2d, 2, &output_shape);
			TensorShapeUtils::MakeShape(dims_1d, 1, &output_shape1);

			Tensor* output_tensor = NULL;
			Tensor* output_tensor_index = NULL;
			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
			OP_REQUIRES_OK(context, context->allocate_output(1, output_shape1, &output_tensor_index));

			auto output       = output_tensor->template flat<T>();
			auto output_index = output_tensor_index->template flat<int32_t>();

            for(int i=0; i<out_nr; ++i) {
                copy_box(boxes_flat.data()+good_index[i]*4,output.data()+i*4);
				output_index.data()[i] = good_index[i];
            }
		}
	private:
		float threshold=1.0;
};
REGISTER_KERNEL_BUILDER(Name("RemoveBoundaryBoxes").Device(DEVICE_CPU).TypeConstraint<float>("T"), RemoveBoundaryBoxesOp<CPUDevice, float>);

/*
 * 对牙片的box进行处理, 删除可能错误的box将box按上，下牙分别输出
 * class_wise:处理是否按每个类分别处理
 * boxes:[X,4], (ymin,xmin,ymax,xmax)
 * probs:[X]
 * labels:[X]
 * 输出的shape类似
 */
REGISTER_OP("TeethBoxesProc")
    .Attr("T: {float, double}")
    .Input("boxes: T")
    .Input("probs: T")
    .Input("labels: int32")
	.Output("tboxes:T")
	.Output("tprobs:T")
	.Output("tlabels:int32")
	.Output("bboxes:T")
	.Output("bprobs:T")
	.Output("blabels:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            auto shape0 = c->Matrix(c->UnknownDim(),4);
            auto shape1 = c->Vector(c->UnknownDim());
			c->set_output(0, shape0);
			c->set_output(1, shape1);
			c->set_output(2, shape1);
			c->set_output(3, shape0);
			c->set_output(4, shape1);
			c->set_output(5, shape1);
			return Status::OK();
			});

template <typename Device, typename T>
class TeethBoxesProcOp: public OpKernel {
    private:
        struct TeethInfo{
            int   index;
            float iou;
            float prob;
			/*
			 * 得分高的好
			 */
            inline float score()const {
				if(iou<0.05) 
					return (1.0-iou)+std::min<float>(prob*1.2,1.0)/2.0;
				else
                	return (1.0-iou)+prob/2.0;
            }
            inline bool operator<( const TeethInfo& rhv)const {
                return score()<rhv.score();
            }
            inline bool operator>( const TeethInfo& rhv)const {
                return score()>rhv.score();
            }
        };
    public:
        explicit TeethBoxesProcOp(OpKernelConstruction* context) : OpKernel(context) {
        }

        void Compute(OpKernelContext* context) override
        {
            const Tensor &boxes       = context->input(0);
            const Tensor &probs       = context->input(1);
            const Tensor &labels      = context->input(2);
            auto          boxes_flat  = boxes.flat<T>();
            auto          probs_flat  = probs.flat<T>();
            auto          labels_flat = labels.flat<int32_t>();
            const auto    nr          = boxes.dim_size(0);
            vector<bool>  is_good(nr,true);
            vector<bool>  is_top(nr,true);

            OP_REQUIRES(context, probs.dims() == 1, errors::InvalidArgument("probs must be 1-dimensional"));
            OP_REQUIRES(context, labels.dims() == 1, errors::InvalidArgument("labels must be 1-dimensional"));
            OP_REQUIRES(context, boxes.dims() == 2, errors::InvalidArgument("boxes must be 2-dimensional"));

            auto ymin = 1.0f;
            auto ymax = 0.0f;

            for(auto i = 0; i<nr; ++i) {
                auto box = boxes_flat.data()+i*4;
                if(ymin>box[0]) 
                    ymin = box[0];
                if(ymax<box[2])
                    ymax = box[2];
            }

            claster(boxes_flat.data(),nr,ymin,ymax,&is_top);
            procWisdomTeeth(boxes_flat.data(),probs_flat.data(),labels_flat.data(),is_top,true,&is_good);
            procWisdomTeeth(boxes_flat.data(),probs_flat.data(),labels_flat.data(),is_top,false,&is_good);
            procTeethNum(boxes_flat.data(),probs_flat.data(),labels_flat.data(),is_top,true,&is_good);
            procTeethNum(boxes_flat.data(),probs_flat.data(),labels_flat.data(),is_top,false,&is_good);
            getOutTensor(context,boxes_flat.data(),probs_flat.data(),labels_flat.data(),is_top,is_good,true);
            getOutTensor(context,boxes_flat.data(),probs_flat.data(),labels_flat.data(),is_top,is_good,false);
        }
        void getOutTensor(OpKernelContext* context,
                const float* boxes,const float* probs,const int32_t* labels,const vector<bool>& is_top,
                const vector<bool>& is_good,bool test_v) {
            TensorShape output_shape0;
            TensorShape output_shape1;
            auto        teeth_nr      = teethNr(is_top,is_good,test_v);
            int         dims_2d[]     = {teeth_nr,4};
            int         dims_1d[]     = {teeth_nr};
            const auto  data_nr       = is_top.size();

            TensorShapeUtils::MakeShape(dims_2d, 2, &output_shape0);
            TensorShapeUtils::MakeShape(dims_1d, 1, &output_shape1);

            Tensor     *output_boxes  = NULL;
            Tensor     *output_labels = NULL;
            Tensor     *output_probs  = NULL;
            const auto  base_index    = test_v?0:3;

            OP_REQUIRES_OK(context, context->allocate_output(0+base_index, output_shape0, &output_boxes));
            OP_REQUIRES_OK(context, context->allocate_output(1+base_index, output_shape1, &output_probs));
            OP_REQUIRES_OK(context, context->allocate_output(2+base_index, output_shape1, &output_labels));

            auto output_boxes_flat  = output_boxes->template flat<T>();
            auto output_probs_flat  = output_probs->template flat<T>();
            auto output_labels_flat = output_labels->template flat<int32_t>();
            int  j                  = 0;

            for(auto i=0; i<data_nr; ++i) {
                if((is_top[i] != test_v) || (!is_good[i])) continue;
                copy_box(boxes+i*4,output_boxes_flat.data()+j*4);
                output_probs_flat.data()[j] = probs[i];
                output_labels_flat.data()[j] = labels[i];
                ++j;
            }
        }
        int teethNr(const vector<bool>& is_top,const vector<bool>& is_good,bool test_v) {
            auto res = 0;
            for(int i=0; i<is_top.size(); ++i) {
                if((is_top[i] != test_v) || (!is_good[i])) continue;
                ++res;
            }
            return res;
        }
        void procTeethNum(const float* boxesdata,const float* probs,const int* labels,vector<bool> is_top,bool test_v,vector<bool>* is_good) 
        {
            auto       teeth_num = 0;
            const auto data_nr   = is_top.size();
            /*
             * 计算牙齿数
             */
            for(auto i=0; i<data_nr; ++i) {
                if((is_top[i] != test_v) || (!(*is_good)[i]) ||(labels[i] != nt_label)) continue;
                ++teeth_num;
            }

            if(teeth_num < kMaxHNormalTeethNr) return;

            vector<TeethInfo> teeth_info;

            for(auto i=0; i<data_nr; ++i) {
                if((is_top[i] != test_v) || (!(*is_good)[i]) ||(labels[i] != nt_label)) continue;
                teeth_info.push_back(TeethInfo({i,0.0f,probs[i]}));
            }
			/*
			 * 先生成一个初步的排序,每一个box都和所有的其它box算jaccard
			 */
            for(auto i=0; i<teeth_info.size(); ++i) {

                auto &info    = teeth_info[i];
                auto  cur_box = boxesdata+info.index *4;

				info.iou = 0.0;

                for(auto j=0; j<teeth_info.size(); ++j) {

                    if(j==i)continue;

                    auto       test_box_index = teeth_info[j].index;
                    auto       test_box       = boxesdata+test_box_index *4;
                    const auto jaccard        = bboxes_jaccard_of_box0(cur_box,test_box);

                    info.iou += jaccard;
                }
            }

            sort(teeth_info.begin(),teeth_info.end());
			/*
			 * 对数据再次排序，得分小的的要和所有得分比他高的算jaccard
			 */
            for(auto i=0; i<teeth_info.size(); ++i) {

                auto &info    = teeth_info[i];
                auto  cur_box = boxesdata+info.index *4;

				info.iou = 0.0;

                for(auto j=i+1; j<teeth_info.size(); ++j) {
                    auto       test_box_index = teeth_info[j].index;
                    auto       test_box       = boxesdata+test_box_index *4;
                    const auto jaccard        = bboxes_jaccard_of_box0(cur_box,test_box);

                    info.iou += jaccard;
                }
            }

            sort(teeth_info.begin(),teeth_info.end(),greater<TeethInfo>());
            for(auto i=kMaxHNormalTeethNr; i<teeth_info.size(); ++i) {
                auto index = teeth_info[i].index;
                (*is_good)[index] = false;
            }
        }
        void procWisdomTeeth(const float* boxesdata,const float* probs,const int* labels,const vector<bool>& is_top,
        bool test_v,
        vector<bool>* is_good) 
        {
            auto       nr           = count(is_top.begin(),is_top.end(),test_v);
            auto       pwt_nr        = 0;
            auto       nwt_nr        = 0;
            auto       data_nr      = is_top.size();
            auto       xmin         = 1.0;
            auto       xmax         = 0.0;
            const auto wt_threshold = 0.14;
            const auto wt_threshold2 = 0.03;

            /*
             * 删除靠近口腔中间的尽头牙
             */
            for(int i=0; i<data_nr; ++i) {
                const auto box = boxesdata+i *4;

                if((is_top[i] != test_v) || (!(*is_good)[i])) continue;

                const auto x   = (box[1]+box[3])/2;

                if(xmax<x)
                    xmax = x;
                if(xmin>x)
                    xmin = x;

                if(labels[i] == wt_label) {
                    if((x>=(0.5-wt_threshold)) && (x<=(0.5+wt_threshold))) {
                        (*is_good)[i] = false;
                    } 
                }
            }
            /*
             * 删除不是最靠边的尽头牙
             */
            for(auto i=0; i<data_nr; ++i) {
                const auto box = boxesdata+i *4;
                if((is_top[i] != test_v) || (!(*is_good)[i])) continue;
                const auto x   = (box[1]+box[3])/2;
                if(labels[i] == wt_label) {
                    if((x>0.5) && (x<xmax-wt_threshold2)) {
                        (*is_good)[i] = false;
                    } else if((x<0.5) && (x>xmin+wt_threshold2)) {
                        (*is_good)[i] = false;
                    } else if(x>0.5) {
                        ++pwt_nr;
                    } else if(x<0.5) {
                        ++nwt_nr;
                    }
                }
            }
            /*
             * 处理右边尽头牙太多的情况
             */
            if(pwt_nr>1) {
                /*
                 * pair中的内容依次为概率，牙齿序号
                 */
                vector<pair<float,int>> datas;
                for(auto i=0; i<data_nr; ++i) {
                    const auto box = boxesdata+i *4;
                    if((is_top[i] != test_v) || (!(*is_good)[i])) continue;
                    const auto x   = (box[1]+box[3])/2;
                    if((labels[i] == wt_label) &&
                            (x>0.5)) {
                        datas.push_back(make_pair(probs[i],i));
                    }
                }
                sort(datas.begin(),datas.end(),std::greater<pair<float,int>>());
                for(auto it = next(datas.begin()); it!=datas.end(); ++it)
                    (*is_good)[it->second] = false;
            }
            if(nwt_nr>1) {
                vector<pair<float,int>> datas;
                for(auto i=0; i<data_nr; ++i) {
                    const auto box = boxesdata+i *4;
                    if((is_top[i] != test_v) || (!(*is_good)[i])) continue;
                    const auto x   = (box[1]+box[3])/2;
                    if((labels[i] == wt_label) &&
                            (x<0.5)) {
                        datas.push_back(make_pair(probs[i],i));
                    }
                }
                sort(datas.begin(),datas.end(),std::greater<pair<float,int>>());
                for(auto it = next(datas.begin()); it!=datas.end(); ++it)
                    (*is_good)[it->second] = false;
            }
        }
        void claster(const float* boxesdata,int boxes_nr,float& init_ymin,float& init_ymax,vector<bool>* is_top) 
        {
            auto       old_is_top   = *is_top;
            auto       maxloop      = 32;
            const auto wt_threshold = 0.14;

            do {
                for(int i=0; i<boxes_nr; ++i) {
                    const auto box = boxesdata+i *4;
                    auto       y   = (box[0]+box[2])/2.0;
                    if(fabsf(y-init_ymin)<fabsf(y-init_ymax))
                        (*is_top)[i] = true;
                    else
                        (*is_top)[i] = false;
                }
                int    top_nr     = 0;
                double top_sum    = 0;
                double bottom_sum = 0;

                for(int i =0; i<boxes_nr; ++i) {
                    auto box = boxesdata+i*4;
                    auto y = (box[0]+box[2])/2.0;
                    auto x = (box[1]+box[3])/2.0;
                    if((*is_top)[i]) {
                        ++top_nr;
                        top_sum  += y;
                    } else {
                        bottom_sum += y;
                    }
                }
                if(top_nr>0) {
                    init_ymin = top_sum/top_nr;
                }
                if(top_nr != boxes_nr) {
                    init_ymax = bottom_sum/(boxes_nr-top_nr);
                }
                if(equal(old_is_top.begin(),old_is_top.end(),is_top->begin())) break;
                old_is_top = *is_top;
                --maxloop;
                if(0 == maxloop) {
                    cout<<"Cluster too many loop, stop cluster."<<endl;
                    break;
                }
            }while(true);

			for(auto i=0; i<boxes_nr; ++i) {
				if(!(*is_top)[i]) continue;

				auto cur_box = boxesdata+i *4;
				auto x       = (cur_box[1]+cur_box[3])/2.0;

				if((x>=0.5-wt_threshold) && (x<=0.5+wt_threshold)) continue;
				if(any_box_on_top(cur_box,boxesdata,boxes_nr)) {
					(*is_top)[i] = false;
				}
			}
        }
        bool any_box_on_top(const float* cur_box,const float* boxes,int box_nr) {

            for(int i=0; i<box_nr; ++i) {
                const auto test_box = boxes+i *4;
                const auto x        = (test_box[1]+test_box[3])/2.0;
                const auto delta    = (test_box[3]-test_box[1])/2.0;
                if((x<cur_box[1]-delta) || (x>cur_box[3]+delta)) continue;
                const auto y = (test_box[0]+test_box[2])/2.0;
                if(y<cur_box[0]) return true;
            }

            return false;
        }
    private:
        static constexpr auto wt_label           = 2;
        static constexpr auto nt_label           = 1;
        static constexpr auto kMaxHNormalTeethNr = 18;
};
REGISTER_KERNEL_BUILDER(Name("TeethBoxesProc").Device(DEVICE_CPU).TypeConstraint<float>("T"), TeethBoxesProcOp<CPUDevice, float>);

/*
 * 删除牙片中明显错误的box
 * 如面积过大的，位置过于靠边的，过长的，过高的，过窄的，过矮的等
 * boxes:[X,4], (ymin,xmin,ymax,xmax)
 * 输出的shape类似
 */
REGISTER_OP("CleanTeethBoxes")
    .Attr("T: {float, double}")
    .Input("boxes: T")
	.Output("output_boxes:T")
	.Output("output_indexs:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			c->set_output(0, c->Matrix(c->UnknownDim(), 4));
			c->set_output(1, c->Vector(c->UnknownDim()));
			return Status::OK();
			});

template <typename Device, typename T>
class CleanTeethBoxesOp: public OpKernel {
    private:
    public:
        explicit CleanTeethBoxesOp(OpKernelConstruction* context) : OpKernel(context) {
        }

        void Compute(OpKernelContext* context) override
        {
            const Tensor &boxes       = context->input(0);
            auto          boxes_flat  = boxes.flat<T>();
            const auto    nr          = boxes.dim_size(0);
            const auto    w_max       = 0.16f;
            const auto    w_min       = 0.006f;
            const auto    h_max       = 0.43f;
            const auto    h_min       = 0.01f;
            const auto    max_ratio   = 12.0f;
            const auto    size_max    = 0.036f;
            const auto    most_left   = 0.12f;
            const auto    most_right  = 1.0f-most_left;
            const auto    most_top    = 0.25f;
            const auto    most_bottom = 1.0f-0.11f;
            vector<bool>  is_good(nr    ,true);

            OP_REQUIRES(context, boxes.dims() == 2, errors::InvalidArgument("boxes must be 2-dimensional"));
			//cout<<"begin-------------------"<<endl<<endl;
            for(auto i = 0; i<nr; ++i) {
                const auto box  = boxes_flat.data()+i *4;
                const auto x    = (box[1]+box[3])/2.0f;
                const auto y    = (box[0]+box[2])/2.0f;
                const auto size = box_size(box);
                const auto w    = box[3]-box[1];
                const auto h    = box[2]-box[0];
				//cout<<w<<","<<h<<","<<size<<","<<x<<","<<y<<","<<(w/h)<<","<<(h/w)<<endl;

                if((w>w_max) || (w<w_min) || (h>h_max) || (h<h_min) || (size>size_max) 
                        || (x<most_left) || (x>most_right) || (y<most_top) || (y>most_bottom)) {
                    is_good[i] = false;
                } else if((w/h>max_ratio) || (h/w>max_ratio))  {
                    is_good[i] = false;
                }
				//cout<<is_good[i]<<endl;
            }
            const int   out_nr        = count(is_good.begin(),is_good.end(),true);
            const int   dims_2d[]     = {out_nr,4};
            const int   dims_1d[]     = {out_nr};
            TensorShape output_shape0;
            TensorShape output_shape1;

            TensorShapeUtils::MakeShape(dims_2d, 2, &output_shape0);
            TensorShapeUtils::MakeShape(dims_1d, 1, &output_shape1);

			Tensor *output_boxes  = NULL;
			Tensor *output_indexs = NULL;

			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape0, &output_boxes));
			OP_REQUIRES_OK(context, context->allocate_output(1, output_shape1, &output_indexs));

			auto outputboxes  = output_boxes->template flat<T>();
			auto outputindexs = output_indexs->template flat<int32_t>();

            for(int i=0,j=0; i<nr; ++i) {
                if(!is_good[i]) continue;
                copy_box(boxes_flat.data()+i*4,outputboxes.data()+j*4);
                outputindexs.data()[j] = i;
                ++j;
            }
        }
};
REGISTER_KERNEL_BUILDER(Name("CleanTeethBoxes").Device(DEVICE_CPU).TypeConstraint<float>("T"), CleanTeethBoxesOp<CPUDevice, float>);
Status distored_boxes_shape(shape_inference::InferenceContext* c) 
{
    auto shape = c->input(0);

    if(!c->FullyDefined(shape)) {
        c->set_output(0,c->Matrix(c->UnknownDim(),4));
        return Status::OK();
    }

    auto          data_nr  = c->Value(c->Dim(shape,0));
    bool          keep_org;
    int           res_nr   = 0;
    vector<float> xoffset;
    vector<float> yoffset;
    vector<float> scale;

    c->GetAttr("keep_org",&keep_org);
    c->GetAttr("xoffset",&xoffset);
    c->GetAttr("yoffset",&yoffset);
    c->GetAttr("scale",&scale);
    if(keep_org)
        res_nr = data_nr;
    else
        res_nr = 0;
    res_nr += (xoffset.size()+yoffset.size()+scale.size())*data_nr;
    c->set_output(0,c->Matrix(res_nr,4));
    return Status::OK();
}
/*
 * 对Boxes:[Nr,4]进行多样化处理
 * scale:对box进行缩放处理
 * offset:对box进行上下左右的平移处理
 */
REGISTER_OP("DistoredBoxes")
    .Attr("T: {float, double}")
	.Attr("scale:list(float)")
	.Attr("xoffset:list(float)")
	.Attr("yoffset:list(float)")
	.Attr("keep_org:bool")
    .Input("boxes: T")
	.Output("output_boxes:T")
	.SetShapeFn(distored_boxes_shape);

template <typename Device, typename T>
class DistoredBoxesOp: public OpKernel {
	private:
	public:
		explicit DistoredBoxesOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
			OP_REQUIRES_OK(context, context->GetAttr("xoffset", &xoffset_));
			OP_REQUIRES_OK(context, context->GetAttr("yoffset", &yoffset_));
			OP_REQUIRES_OK(context, context->GetAttr("keep_org", &keep_org_));
		}
        inline int get_output_nr(int nr)const {
            auto    output_nr  = 0; 
            if(keep_org_)
                output_nr += nr;
            output_nr += (xoffset_.size()+yoffset_.size())*nr;
            output_nr += scale_.size()*nr;
            return output_nr;
        }

		void Compute(OpKernelContext* context) override
        {
            const Tensor &boxes      = context->input(0);
            auto          boxes_flat = boxes.flat<T>();
            const auto    nr         = boxes.dim_size(0);
            const auto    output_nr  = get_output_nr(nr);

            OP_REQUIRES(context, boxes.dims() == 2, errors::InvalidArgument("boxes must be 2-dimensional"));

            const int   dims_2d[]     = {int(output_nr),4};
            TensorShape output_shape0;

            TensorShapeUtils::MakeShape(dims_2d, 2, &output_shape0);

            Tensor *output_boxes  = NULL;

            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape0, &output_boxes));

            auto outputboxes  = output_boxes->template flat<T>();
            auto output_index = 0;

            if(keep_org_) {
                copy_boxes(boxes_flat.data(),outputboxes.data(),nr);
                output_index = nr;
            }
            processOffsetX(boxes_flat.data(),outputboxes.data(),xoffset_,nr,output_index);
            processOffsetY(boxes_flat.data(),outputboxes.data(),yoffset_,nr,output_index);

            for(auto i=0; i<scale_.size(); ++i) {
                processScale(boxes_flat.data(),outputboxes.data(),scale_[i],nr,output_index);
            }
        }
		void processOffsetX(const T* src_boxes,T* out_boxes,const vector<float>& xoffset,const int nr,int& output_index) 
		{
			for(auto k=0; k<nr; ++k) {
				const auto src_box = src_boxes+4*k;
				for(auto i=0; i<xoffset.size(); ++i) {
					const auto os      = xoffset[i];
					const auto dx      = (src_box[3]-src_box[1]) *os;
					auto       cur_box = out_boxes+4 *output_index++;

					copy_box(src_box,cur_box);
					cur_box[1] += dx;
					cur_box[3] += dx;
				}
			}
		}
		void processOffsetY(const T* src_boxes,T* out_boxes,const vector<float>& yoffset,const int nr,int& output_index) 
		{
			for(auto k=0; k<nr; ++k) {
				const auto src_box = src_boxes+4*k;
				for(auto i=0; i<yoffset.size(); ++i) {
					const auto os      = yoffset[i];
					const auto dy      = (src_box[2]-src_box[0]) *os;
					auto       cur_box = out_boxes+4 *output_index++;

					copy_box(src_box,cur_box);
					cur_box[0] -= dy;
					cur_box[2] -= dy;
				}
			}
		}
		void processScale(const T* src_boxes,T* out_boxes,const float scale,const int nr,int& output_index) 
		{
			for(auto k=0; k<nr; ++k) {
				const auto src_box = src_boxes+4 *k;
				const auto dx      = (src_box[3]-src_box[1]) *(scale-1.0)/2.;
				const auto dy      = (src_box[2]-src_box[0]) *(scale-1.0)/2.;
				auto       cur_box = out_boxes+4 *output_index++;

				copy_box(src_box,cur_box);
				cur_box[0] -= dy;
				cur_box[1] -= dx;
				cur_box[2] += dy;
				cur_box[3] += dx;
			}
		}
	private:
		vector<float> scale_;
		vector<float> xoffset_;
		vector<float> yoffset_;
		bool          keep_org_;
};
REGISTER_KERNEL_BUILDER(Name("DistoredBoxes").Device(DEVICE_CPU).TypeConstraint<float>("T"), DistoredBoxesOp<CPUDevice, float>);

/*
 * 对Boxes的概率进行调整
 * 具体方法为：
 * 1，如果最大概率不在指定的类型中则不调整
 * 2，否则将指定的类型中非最大概率的一半值分配给最大概率
 * classes:指定需要调整的类别,如果为空则表示使用所有的非背景类别
 * probs:概率，[X,N]
 */
REGISTER_OP("ProbabilityAdjust")
    .Attr("T: {float, double}")
	.Attr("classes:list(int)")
    .Input("probs: T")
	.Output("output_probs:T")
	.SetShapeFn(shape_inference::UnchangedShape);

template <typename Device, typename T>
class ProbabilityAdjustOp: public OpKernel {
	public:
		explicit ProbabilityAdjustOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("classes", &classes_));
		}
		void Compute(OpKernelContext* context) override
		{
			const Tensor &probs = context->input(0);
			auto          probs_flat = probs.flat<T>();
			const auto    nr         = probs.dim_size(0);
			const auto    classes_nr = probs.dim_size(1);

			OP_REQUIRES(context, probs.dims() == 2, errors::InvalidArgument("probs must be 2-dimensional"));

			TensorShape output_shape = probs.shape();

			if(classes_.empty()) {
				for(auto i=1; i<classes_nr; ++i) {
					classes_.push_back(i);
				}
			}
			auto it = remove_if(classes_.begin(),classes_.end(),[classes_nr](int i){ return (i<0) || (i>=classes_nr);});

			classes_.erase(it,classes_.end());

			Tensor *output_probs = NULL;

			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_probs));

			output_probs->CopyFrom(probs,output_shape);

			auto output = output_probs->template flat<T>();

			for(int i=0; i<nr; ++i) {
				auto       v     = output.data()+i*classes_nr;
				const auto it    = max_element(v,v+classes_nr);
				const int  index = distance(v,it);
				auto       jt    = find(classes_.begin(),classes_.end(),index);
				auto       sum   = 0.;

                if(jt ==classes_.end()) continue;

                for(auto k:classes_) {
                    if(k==index)continue;
                    sum += v[k]/2.;
                    v[k] = v[k]/2.;
                }
                v[index] = v[index]+sum;
			}
		}
	private:
		vector<int> classes_;
};
REGISTER_KERNEL_BUILDER(Name("ProbabilityAdjust").Device(DEVICE_CPU).TypeConstraint<float>("T"), ProbabilityAdjustOp<CPUDevice, float>);

Status random_distored_boxes_shape(shape_inference::InferenceContext* c) 
{
    auto shape = c->input(0);

    if(!c->FullyDefined(shape)) {
        c->set_output(0,c->Matrix(c->UnknownDim(),4));
        return Status::OK();
    }

    auto data_nr  = c->Value(c->Dim(shape,0));
    bool keep_org;
    int  res_nr   = 0;
    int  size;

    c->GetAttr("keep_org",&keep_org);
    c->GetAttr("size",&size);
    if(keep_org)
        res_nr = data_nr;
    else
        res_nr = 0;
    res_nr += size*data_nr;
    c->set_output(0,c->Matrix(res_nr,4));
    return Status::OK();
}
/*
 * 对Boxes:[Nr,4]进行多样化处理
 * limits:[xoffset,yoffset,scale]大小限制
 * size:[产生xoffset,yoffset,scale的数量]
 */
REGISTER_OP("RandomDistoredBoxes")
    .Attr("T: {float, double}")
	.Attr("limits:list(float)")
	.Attr("size:int")
	.Attr("keep_org:bool")
    .Input("boxes: T")
	.Output("output_boxes:T")
	.SetShapeFn(random_distored_boxes_shape);

template <typename Device, typename T>
class RandomDistoredBoxesOp: public OpKernel {
	private:
	public:
		explicit RandomDistoredBoxesOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("limits", &limits_));
			OP_REQUIRES_OK(context, context->GetAttr("size", &size_));
			OP_REQUIRES_OK(context, context->GetAttr("keep_org", &keep_org_));
		}
        inline int get_output_nr(int nr)const {
            auto    output_nr  = 0; 
            if(keep_org_)
                output_nr += nr;
            output_nr += size_*nr;
            return output_nr;
        }

		void Compute(OpKernelContext* context) override
		{
			const Tensor &boxes      = context->input(0);
			auto          boxes_flat = boxes.flat<T>();
			const auto    nr         = boxes.dim_size(0);
			const auto    output_nr  = get_output_nr(nr);

			OP_REQUIRES(context, boxes.dims() == 2, errors::InvalidArgument("boxes must be 2-dimensional"));

			const int   dims_2d[]     = {int(output_nr),4};
			TensorShape output_shape0;

			TensorShapeUtils::MakeShape(dims_2d, 2, &output_shape0);

			Tensor *output_boxes  = NULL;

			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape0, &output_boxes));

			auto outputboxes  = output_boxes->template flat<T>();
			auto output_index = 0;

			if(keep_org_) {
				copy_boxes(boxes_flat.data(),outputboxes.data(),nr);
				output_index = nr;
			}
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<> xoffset(-limits_[0],limits_[0]);
			std::uniform_real_distribution<> yoffset(-limits_[1],limits_[1]);
			std::uniform_real_distribution<> scale(-limits_[2],limits_[2]);
			for(auto i=0; i<size_; ++i) {
				/*process(boxes_flat.data(),outputboxes.data(),
						nr,
						xoffset(gen),
						yoffset(gen),
						scale(gen),
						output_index);*/
				process(boxes_flat.data(),outputboxes.data(),
						nr,
						xoffset,
						yoffset,
						scale,
                        gen,
						output_index);
			}
		}
		void process(const T* src_boxes,T* out_boxes,int nr,float xoffset,float yoffset,float scale,int& output_index) 
		{
            for(auto k=0; k<nr; ++k) {
                const auto src_box = src_boxes+4*k;
				const auto sdx      = (src_box[3]-src_box[1]) *scale/2.;
				const auto sdy      = (src_box[2]-src_box[0]) *scale/2.;
                const auto dx      = (src_box[3]-src_box[1])*xoffset;
                const auto dy      = (src_box[2]-src_box[0])*yoffset;
                auto       cur_box = out_boxes+4 *output_index++;

                copy_box(src_box,cur_box);
				cur_box[0] += dy-sdy;
				cur_box[1] += dx-sdx;
				cur_box[2] += dy+sdy;
				cur_box[3] += dx+sdx;
            }
		}
        template<typename dis_t,typename gen_t>
		void process(const T* src_boxes,T* out_boxes,int nr,dis_t& xoffset,dis_t& yoffset,dis_t& scale,gen_t& gen,int& output_index) 
		{
            for(auto k=0; k<nr; ++k) {
                const auto src_box = src_boxes+4*k;
				const auto sdx     = (src_box[3]-src_box[1]) *scale(gen)/2.;
				const auto sdy     = (src_box[2]-src_box[0]) *scale(gen)/2.;
                const auto dx      = (src_box[3]-src_box[1])*xoffset(gen);
                const auto dy      = (src_box[2]-src_box[0])*yoffset(gen);
                auto       cur_box = out_boxes+4 *output_index++;

                copy_box(src_box,cur_box);
				cur_box[0] += dy-sdy;
				cur_box[1] += dx-sdx;
				cur_box[2] += dy+sdy;
				cur_box[3] += dx+sdx;
            }
		}
	private:
		int           size_;
		vector<float> limits_;
		bool          keep_org_;
};
REGISTER_KERNEL_BUILDER(Name("RandomDistoredBoxes").Device(DEVICE_CPU).TypeConstraint<float>("T"), RandomDistoredBoxesOp<CPUDevice, float>);

/*
 * 将boxes中与gboxes IOU最大且不小于threadshold的标记为相应的label, 即1个gboxes最多与一个boxes相对应
 * boxes:[batch_size,N,4](y,x,h,w)
 * gboxes:[batch_size,M,4]
 * glabels:[batch_size,M] 0表示背景
 * glens:[batch_size] 用于表明gboxes中的有效boxes数量
 * output_labels:[batch_size,N]
 * output_scores:[batch_size,N]
 * threashold: IOU threshold
 */
REGISTER_OP("BoxesMatch")
    .Attr("T: {float, double,int32}")
	.Attr("threshold:float")
    .Input("boxes: T")
    .Input("gboxes: T")
	.Input("glabels:int32")
	.Input("glens:int32")
	.Output("output_labels:int32")
	.Output("output_scores:T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
		    auto boxes_shape = c->input(0);
			shape_inference::ShapeHandle outshape;
			c->Subshape(boxes_shape,0,2,&outshape);
			c->set_output(0, outshape);
			c->set_output(1, outshape);
			return Status::OK();
			});

template <typename Device, typename T>
class BoxesMatchOp: public OpKernel {
	public:
		explicit BoxesMatchOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold_));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &_boxes   = context->input(0);
			const Tensor &_gboxes  = context->input(1);
			const Tensor &_glabels = context->input(2);
			const Tensor &_glens   = context->input(3);
			auto          boxes    = _boxes.tensor<T,3>();
			auto          gboxes   = _gboxes.tensor<T,3>();
			auto          glabels  = _glabels.tensor<int,2>();
			auto          glens    = _glens.tensor<int,1>();

			OP_REQUIRES(context, _boxes.dims() == 3, errors::InvalidArgument("box data must be 3-dimensional"));
			OP_REQUIRES(context, _gboxes.dims() == 3, errors::InvalidArgument("gboxes data must be 3-dimensional"));
			OP_REQUIRES(context, _glabels.dims() == 2, errors::InvalidArgument("glabels data must be 2-dimensional"));
			OP_REQUIRES(context, _glens.dims() == 1, errors::InvalidArgument("glens data must be 1-dimensional"));

			const int batch_nr  = _boxes.dim_size(0);
			const int boxes_nr  = _boxes.dim_size(1);
			const int gboxes_nr = _gboxes.dim_size(1);

			int dims_2d[2] = {batch_nr,boxes_nr};
			TensorShape  outshape;
			Tensor      *output_classes     = NULL;
			Tensor      *output_scores      = NULL;

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape, &output_classes));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape, &output_scores));

			auto oclasses     = output_classes->template tensor<int,2>();
			auto oscores      = output_scores->template tensor<T,2>();

            oclasses.setZero();
            oscores.setZero();

            for(auto i=0; i<batch_nr; ++i) {
                for(auto j=0; j<glens(i); ++j) {
                    auto max_scores = -1.0;
                    auto index      = -1;

                    for(auto k=0; k<boxes_nr; ++k) {
                        if(oclasses(i,k) != 0) continue;

                        Eigen::Tensor<T,1,Eigen::RowMajor> gbox = gboxes.chip(i,0).chip(j,0);
                        Eigen::Tensor<T,1,Eigen::RowMajor> box = boxes.chip(i,0).chip(k,0);
                        auto jaccard = bboxes_jaccardv1(gbox,box);

                        if((jaccard>threshold_) && (jaccard>max_scores)) {
                            max_scores = jaccard;
                            index = k;
                        }
                    }
                    if(index>0) {
                        oclasses(i,index) = glabels(i,j);
                        oscores(i,index) = max_scores;
                    }
                }
            }
		}
	private:
		float threshold_;
};
REGISTER_KERNEL_BUILDER(Name("BoxesMatch").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesMatchOp<CPUDevice, float>);

REGISTER_OP("AnchorGenerator")
    .Attr("scales: list(float)")
    .Attr("aspect_ratios:list(float)")
	.Input("shape:int32")
	.Input("size:int32")
	.Output("anchors:float")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			c->set_output(0, c->Matrix(-1, 4));
			return Status::OK();
			});

template <typename Device, typename T>
class AnchorGeneratorOp: public OpKernel {
	public:
		explicit AnchorGeneratorOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("scales", &scales_));
            vector<float> aspect_ratios;
			OP_REQUIRES_OK(context, context->GetAttr("aspect_ratios", &aspect_ratios));
            aspect_ratios_.assign(aspect_ratios.rbegin(),aspect_ratios.rend());
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &_in_shape= context->input(0);
			const Tensor &_in_size = context->input(1);
			auto          in_shape = _in_shape.tensor<int,1>();
			auto          in_size = _in_size.tensor<int,1>();

			OP_REQUIRES(context, _in_shape.dims() == 1, errors::InvalidArgument("shape data must be 1-dimensional"));
			OP_REQUIRES(context, _in_size.dims() == 1, errors::InvalidArgument("size data must be 1-dimensional"));
            const int out_nr = scales_.size()*aspect_ratios_.size()*in_shape(0)*in_shape(1);

			int dims_2d[2] = {out_nr,4};
			TensorShape  outshape;
			Tensor      *output_anchors = NULL;

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape, &output_anchors));

			auto oanchors = output_anchors->template tensor<T,2>();

            oanchors.setZero();
            auto index = 0;

            const auto x_delta = 1.0/(in_shape(1));
            const auto y_delta = 1.0/(in_shape(0));
            auto y_pos = y_delta/2.0;
            for(auto i=0; i<in_shape(0); ++i) {
                auto x_pos = x_delta/2.0;
                for(auto j=0; j<in_shape(1); ++j) {
                    for(auto& a:aspect_ratios_) {
                        for(auto& s:scales_) {
                            const auto sa = sqrt(a);
                            const auto hw = s/(2*in_size(1))/sa;
                            const auto hh = s/(2*in_size(0))*sa;
                            oanchors(index,0) = y_pos-hh;
                            oanchors(index,1) = x_pos-hw;
                            oanchors(index,2) = y_pos+hh;
                            oanchors(index,3) = x_pos+hw;
                            ++index;
                        }
                    }
                    x_pos += x_delta;
                }
                y_pos += y_delta;
            }
		}
	private:
        std::vector<float> scales_;
        std::vector<float> aspect_ratios_;
};
REGISTER_KERNEL_BUILDER(Name("AnchorGenerator").Device(DEVICE_CPU), AnchorGeneratorOp<CPUDevice, float>);
