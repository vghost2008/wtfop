#include <stdio.h>
#include <cfloat>
#include <list>
#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <boost/algorithm/clamp.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/work_sharder.h"
#include <boost/geometry/io/wkt/write.hpp>

using namespace tensorflow;
using namespace std;
typedef Eigen::ThreadPoolDevice CPUDevice;
namespace bg = boost::geometry;
using Point=bg::model::d2::point_xy<double>;
using MPoint = bg::model::multi_point<Point>;
using Box=bg::model::box<Point>;
using Polygon=boost::geometry::model::polygon<Point>;



/*
 * 对牙体想着疾病进行后处理，标签1的疾病为根尖疾病，在牙齿中心点圈的外围，标签1为牙体疾病，在牙齿圈的内部
 * teeth_boxes:[N,4] (ymin,xmin,ymax,xmax)
 * diseased_boxes:[M,4]
 * diseased_labels:[M]
 * diseased_probability:[M]
 * output:[M]
 */
REGISTER_OP("TeethDiseasedProc")
    .Attr("T: {float, double,int32}")
    .Input("teeth_boxes: T")
    .Input("diseased_boxes: T")
    .Input("diseased_labels: int32")
    .Input("diseased_probability: T")
	.Output("keep_mask:bool")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			c->set_output(0, c->Vector(c->Dim(c->input(1),0)));
			return Status::OK();
			});

template <typename Device, typename T>
class TeethDiseasedProcOp: public OpKernel {
	public:
		explicit TeethDiseasedProcOp(OpKernelConstruction* context) : OpKernel(context) {
		}

		void Compute(OpKernelContext* context) override
        {
            const Tensor &_teeth_boxes     = context->input(0);
            const Tensor &_diseased_boxes  = context->input(1);
            const Tensor &_diseased_labels = context->input(2);
            auto          teeth_boxes      = _teeth_boxes.template tensor<T,2>();
            auto          diseased_boxes   = _diseased_boxes.template tensor<T,2>();
            auto          diseased_labels  = _diseased_labels.template tensor<int,1>();

            OP_REQUIRES(context, _teeth_boxes.dims() == 2, errors::InvalidArgument("teeth boxes data must be 2-dimensional"));
            OP_REQUIRES(context, _diseased_boxes.dims() == 2, errors::InvalidArgument("diseased boxes data must be 2-dimensional"));
            OP_REQUIRES(context, _diseased_labels.dims() == 1, errors::InvalidArgument("diseased labels data must be 1-dimensional"));

            const auto teeth_nr    = _teeth_boxes.dim_size(0);
            const auto diseased_nr = _diseased_boxes.dim_size(0);
            vector<bool>  output_mask(diseased_nr,true);
            MPoint  points1;
            Polygon hull1;
            MPoint  points2;
            Polygon hull2;

            if(teeth_nr<3) {
                output(output_mask,context);
                return;
            }

			for(auto i=0; i<teeth_nr; ++i) {
				const Eigen::Tensor<T,1,Eigen::RowMajor> box_data = teeth_boxes.chip(i,0);
				Box box(Point(box_data(1),box_data(0)),Point(box_data(3),box_data(2)));
				auto min_x    = bg::get<bg::min_corner, 0>(box);
				auto min_y    = bg::get<bg::min_corner, 1>(box);
				auto max_x    = bg::get<bg::max_corner,0>(box);
				auto max_y    = bg::get<bg::max_corner,1>(box);
				auto centroid = bg::return_centroid<Point>(box);

				bg::append(points1,centroid);
				bg::append(points2,Point(min_x,min_y));
				bg::append(points2,Point(max_x,min_y));
				bg::append(points2,Point(max_x,max_y));
				bg::append(points2,Point(min_x,max_y));
			}

            boost::geometry::convex_hull(points1,hull1);
            boost::geometry::convex_hull(points2,hull2);

            for(auto i=0; i<diseased_nr; ++i) {
                const Eigen::Tensor<T,1,Eigen::RowMajor> box_data = diseased_boxes.chip(i,0);
                Box box(Point(box_data(1),box_data(0)),Point(box_data(3),box_data(2)));
                auto centroid = bg::return_centroid<Point>(box);
                if((diseased_labels(i)==1) && !bg::within(centroid,hull1))continue;
                if((diseased_labels(i)==2) && bg::within(centroid,hull2)) continue;
                output_mask[i] = false;
            }

            output(output_mask,context);
        }
        void output(const vector<bool>& mask,OpKernelContext* context) 
        {
			Tensor      *output_mask  = NULL;
			TensorShape  output_shape;
			const int    dims_1d[]    = {int(mask.size())};

			TensorShapeUtils::MakeShape(dims_1d, 1, &output_shape);
			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_mask));

			auto output           = output_mask->template flat<bool>();
            for(auto i=0; i<mask.size(); ++i)
                output.data()[i] = mask[i];
        }

};
REGISTER_KERNEL_BUILDER(Name("TeethDiseasedProc").Device(DEVICE_CPU).TypeConstraint<float>("T"), TeethDiseasedProcOp<CPUDevice, float>);
/*
 * 根据输入的boxes, labels生成一个邻接矩阵
 * min_nr: 每个节点至少与min_nr个节点相连接
 * min_dis:如果两个节点间的距离小于min_dis, 他们之间应该需要一个连接
 * teeth_boxes: [N,4], ymin,xmin,ymax,xmax相对坐标
 * teeth_labels:[N]
 * output:
 * matrix[N,N] 第i行,j列表示第i个点与第j个点之间的连接
 */
REGISTER_OP("TeethAdjacentMatrix")
    .Attr("T: {float, double,int32}")
	.Attr("min_nr:int")
	.Attr("min_dis:float")
    .Input("boxes: T")
    .Input("labels: int32")
	.Output("matrix:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
		    auto box_nr = c->Dim(c->input(0),0);
			c->set_output(0, c->Matrix(box_nr,box_nr));
			return Status::OK();
			});

template <typename Device, typename T>
class TeethAdjacentMatrixOp: public OpKernel {
    public:
        explicit TeethAdjacentMatrixOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("min_nr", &min_nr_));
            OP_REQUIRES_OK(context, context->GetAttr("min_dis", &min_dis_));
        }

        void Compute(OpKernelContext* context) override
        {
            const Tensor &_teeth_boxes  = context->input(0);
            const Tensor &_teeth_labels = context->input(1);
            auto          teeth_boxes   = _teeth_boxes.template tensor<T,2>();
            auto          teeth_labels  = _teeth_labels.template tensor<int,1>();

            OP_REQUIRES(context, _teeth_boxes.dims() == 2, errors::InvalidArgument("teeth boxes data must be 2-dimensional"));
            OP_REQUIRES(context, _teeth_labels.dims() == 1, errors::InvalidArgument("teeth labels data must be 1-dimensional"));
            const auto teeth_nr    = _teeth_boxes.dim_size(0);
            Eigen::Tensor<T,2,Eigen::RowMajor> dis_matrix(teeth_nr,teeth_nr);
			const auto yscale = 0.533;

            dis_matrix.setZero();

            for(auto i=0; i<teeth_nr-1; ++i) {
                const Eigen::Tensor<T,1,Eigen::RowMajor> box_data0 = teeth_boxes.chip(i,0);

                dis_matrix(i,i) = 0.;
                for(auto j=i+1; j<teeth_nr; ++j) {
                    const Eigen::Tensor<T,1,Eigen::RowMajor> box_data1 = teeth_boxes.chip(j,0);
                    Box box0(Point(box_data0(1),box_data0(0)*yscale),Point(box_data0(3),box_data0(2)*yscale));
                    Box box1(Point(box_data1(1),box_data1(0)*yscale),Point(box_data1(3),box_data1(2)*yscale));

                    auto dis = bg::distance(box0,box1);
                    dis_matrix(i,j) = dis;
                    dis_matrix(j,i) = dis;
                }
            }

            Tensor      *output_matrix = NULL;
            TensorShape  output_shape;
            const int    dims_2d[]     = {int(teeth_nr),int(teeth_nr)};
            auto         min_nr        = std::min<int>(min_nr_,teeth_nr);

            TensorShapeUtils::MakeShape(dims_2d, 2, &output_shape);
            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_matrix));

            auto output           = output_matrix->template tensor<int,2>();
            /*
             * dis<min_dis的类型为1
             * 自连接类型为2
             * 其它类型为3
             */
             output.setZero();
			 vector<int> indexs(teeth_nr);
			 int i = 0;
			 generate(indexs.begin(),indexs.end(),[&i](){
				 return i++;
			 });
			 /*
			  * 用于保证构建一个连通图
			  */
			 for(auto i=1; i<teeth_nr; ++i) {
				 auto cur = indexs.back();

				 indexs.pop_back();

				 vector<pair<float,int>> dis_data;

				 dis_data.reserve(indexs.size());
				 for(auto index:indexs) {
					 dis_data.push_back(make_pair(dis_matrix(cur,index),index));
				 }
				 auto it = min_element(dis_data.begin(),dis_data.end());
				 if(it->first<min_dis_)
					 output(cur,it->second) = 1;
				 else
					 output(cur,it->second) = 3;
			 }

             for(auto i=0; i<teeth_nr; ++i) {
                 auto conn_nr = 0;
                 for(auto j=0; j<teeth_nr; ++j) {
                     if(i==j) {
                         output(i,j) = 2;
                         ++conn_nr;
                         continue;
                     }
                     if(dis_matrix(i,j)<min_dis_) {
                         output(i,j) = 1;
                         ++conn_nr;
                         continue;
                     }
                 }
                 if(conn_nr>=min_nr)continue;
                 vector<pair<float,int>> dis_data;
				 dis_data.reserve(teeth_nr);
                 for(auto j=0; j<teeth_nr; ++j) {
                     dis_data.push_back(make_pair(dis_matrix(i,j),j));
                 }
                 sort(dis_data.begin(),dis_data.end());
                 for(auto j=0; j<min_nr; ++j) {
					 auto index = dis_data[j].second;
                     if(output(i,index) == 0)
                         output(i,index) = 3;
                 }
             }
			 auto total_nr = 0;
			 for(auto i=0; i<teeth_nr; ++i) {
			 for(auto j=0; j<teeth_nr; ++j) {
				 if(output(i,j)>0)++total_nr;
			 }
			 }
			 cout<<"Total edge number: "<<total_nr<<endl<<endl;
        }
    private:
        int   min_nr_  = 0;
        float min_dis_ = 0.0;

};
REGISTER_KERNEL_BUILDER(Name("TeethAdjacentMatrix").Device(DEVICE_CPU).TypeConstraint<float>("T"), TeethAdjacentMatrixOp<CPUDevice, float>);
