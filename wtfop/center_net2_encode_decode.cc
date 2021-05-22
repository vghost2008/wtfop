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
#include "unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h"
//#include "unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/work_sharder.h"
#include "bboxes.h"
#include "wtoolkit.h"
#include "wtoolkit_cuda.h"
#include <future>

using namespace tensorflow;
using namespace std;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
/*
 * num_classes: 类别数，不含背景
 * gaussian_iou: 一般为0.7
 * gbboxes: groundtruth bbox, [B,N,4] 相对坐标
 * glabels: 标签[B,N], 背景为0
 * glength: 有效的groundtruth bbox数量
 * output_size: 输出图的大小[2]=(OH,OW) 
 *
 * output:
 * output_heatmaps_c: center heatmaps [B,OH,OW,num_classes]
 * output_hw_offset: [B,OH,OW,4], (h,w,yoffset,xoffset)
 * output_mask: [B,OH,OW,2] (hw_mask,offset_mask)
 */
REGISTER_OP("Center2BoxesEncode")
    .Attr("T: {float,double,int32,int64}")
	.Attr("num_classes:int")
	.Attr("gaussian_iou:float=0.7")
    .Input("gbboxes: T")
    .Input("glabels: int32")
    .Input("glength: int32")
    .Input("output_size: int32")
	.Output("output_heatmaps_c:T")
	.Output("output_hw_offset:T")
	.Output("output_hw_offset_mask:T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            const auto input_shape0 = c->input(0);
            const auto batch_size = c->Dim(input_shape0,0);
            int num_classes;
            c->GetAttr("num_classes",&num_classes);
            auto shape0 = c->MakeShape({batch_size,-1,-1,num_classes});
            auto shape1 = c->MakeShape({batch_size,-1,-1,4});
            auto shape2 = c->MakeShape({batch_size,-1,-1,2});

			c->set_output(0, shape0);
			c->set_output(1, shape1);
			c->set_output(2, shape2);

			return Status::OK();
			});

template <typename Device,typename T>
class Center2BoxesEncodeOp: public OpKernel {
	public:
		explicit Center2BoxesEncodeOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("num_classes", &num_classes_));
			OP_REQUIRES_OK(context, context->GetAttr("gaussian_iou", &gaussian_iou_));
		}

		void Compute(OpKernelContext* context) override
        {
            TIME_THISV1("Center2BoxesEncode");
            const Tensor &_gbboxes    = context->input(0);
            const Tensor &_glabels    = context->input(1);
            const Tensor &_gsize      = context->input(2);
            auto          gbboxes     = _gbboxes.template tensor<T,3>();
            auto          glabels     = _glabels.template tensor<int,2>();
            auto          gsize       = _gsize.template tensor<int,1>();
            auto          output_size = context->input(3).template flat<int>().data();
            const auto    batch_size  = _gbboxes.dim_size(0);

            OP_REQUIRES(context, _gbboxes.dims() == 3, errors::InvalidArgument("box data must be 3-dimension"));
            OP_REQUIRES(context, _glabels.dims() == 2, errors::InvalidArgument("labels data must be 2-dimension"));
            OP_REQUIRES(context, _gsize.dims() == 1, errors::InvalidArgument("gsize data must be 1-dimension"));

            int           dims_4d0[4]            = {int(batch_size),output_size[0],output_size[1],num_classes_};
            int           dims_4d1[4]            = {int(batch_size),output_size[0],output_size[1],4};
            int           dims_4d2[4]            = {int(batch_size),output_size[0],output_size[1],2};
            TensorShape  outshape0;
            TensorShape  outshape1;
            TensorShape  outshape2;
            Tensor      *output_heatmaps_c  = NULL;
            Tensor      *output_hw_offset      = NULL;
            Tensor      *output_mask = NULL;

            TensorShapeUtils::MakeShape(dims_4d0, 4, &outshape0);
            TensorShapeUtils::MakeShape(dims_4d1, 4, &outshape1);
            TensorShapeUtils::MakeShape(dims_4d2, 4, &outshape2);


            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_heatmaps_c));
            OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_hw_offset));
            OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_mask));

            auto heatmaps_c  = output_heatmaps_c->template tensor<T,4>();
            auto hw_offsets     = output_hw_offset->template tensor<T,4>();
            auto o_mask = output_mask->template tensor<T,4>();
            Eigen::Tensor<float,2,Eigen::RowMajor> max_probs(output_size[0],output_size[1]);

            heatmaps_c.setZero();
            hw_offsets.setZero();
            o_mask.setZero();

            for(auto i=0; i<batch_size; ++i) {
                max_probs.setZero();
                for(auto j=0; j<gsize(i); ++j) {
                    const auto fytl = gbboxes(i,j,0)*(output_size[0]-1);
                    const auto fxtl = gbboxes(i,j,1)*(output_size[1]-1);
                    const auto fybr = gbboxes(i,j,2)*(output_size[0]-1);
                    const auto fxbr = gbboxes(i,j,3)*(output_size[1]-1);
                    const auto fyc = (fytl+fybr)/2;
                    const auto fxc = (fxtl+fxbr)/2;
                    const auto yc = int(fyc+0.5);
                    const auto xc = int(fxc+0.5);
                    const auto r0 = get_gaussian_radius(fybr-fytl,fxbr-fxtl,gaussian_iou_);
                    const auto label = glabels(i,j);
                    const auto h = fybr-fytl;
                    const auto w = fxbr-fxtl;

                    if(yc<0||xc<0||yc>=output_size[0]||xc>=output_size[1]) {
                        cout<<"ERROR bboxes data: "<<gbboxes(i,j,0)<<","<<gbboxes(i,j,1)<<","<<gbboxes(i,j,2)<<","<<gbboxes(i,j,3)<<endl;
                        continue;
                    }

                    draw_gaussian(heatmaps_c,fxc,fyc,r0,i,label,5);
                    draw_gaussianv2(max_probs,hw_offsets,fxc,fyc,r0,h,w,i,5);

                    hw_offsets(i,yc,xc,2) = fyc-yc;
                    hw_offsets(i,yc,xc,3) = fxc-xc;
                    o_mask(i,yc,xc,1) = 1.0;
                }
                o_mask.chip(i,0).chip(0,2) = max_probs;
            }
        }
        template<typename DT>
        static void draw_gaussian(DT& data,int cx,int cy,float radius,int batch_index,int class_index,float delta=6,float k=1.0)
        {
            const auto width   = data.dimension(2);
            const auto height  = data.dimension(1);
            const auto xtl     = max(0,int(cx-radius));
            const auto ytl     = max(0,int(cy-radius));
            const auto xbr     = min<int>(width,int(cx+radius+1));
            const auto ybr     = min<int>(height,int(cy+radius+1));
            const auto sigma   = (2*radius+1)/delta;
            const auto c_index = class_index-1;

            for(auto x=xtl; x<xbr; ++x) {
                for(auto y=ytl; y<ybr; ++y) {
                    auto dx = x-cx;
                    auto dy = y-cy;
                    auto v = exp(-(dx*dx+dy*dy)/(2*sigma*sigma))*k;
                    data(batch_index,y,x,c_index) = max(data(batch_index,y,x,c_index),v);
                }
            }
        }
        template<typename DT0,typename DT1>
        static void draw_gaussianv2(DT0& data0,DT1& data1,int cx,int cy,float radius,float h,float w,int batch_index,int class_index,float delta=6,float k=1.0)
        {
            const auto width   = data1.dimension(2);
            const auto height  = data1.dimension(1);
            const auto xtl     = max(0,int(cx-radius));
            const auto ytl     = max(0,int(cy-radius));
            const auto xbr     = min<int>(width,int(cx+radius+1));
            const auto ybr     = min<int>(height,int(cy+radius+1));
            const auto sigma   = (2*radius+1)/delta;
            const auto c_index = class_index-1;

            for(auto x=xtl; x<xbr; ++x) {
                for(auto y=ytl; y<ybr; ++y) {
                    auto dx = x-cx;
                    auto dy = y-cy;
                    auto v = exp(-(dx*dx+dy*dy)/(2*sigma*sigma))*k;
                    if(data0(y,x)<v) {
                        data0(y,x) = v;    
                        data1(batch_index,y,x,0) = h;
                        data1(batch_index,y,x,1) = w;
                    }
                }
            }
        }
	private:
        int   num_classes_  = 80;
        float gaussian_iou_ = 0.7f;
};
REGISTER_KERNEL_BUILDER(Name("Center2BoxesEncode").Device(DEVICE_CPU).TypeConstraint<float>("T"), Center2BoxesEncodeOp<CPUDevice, float>);
