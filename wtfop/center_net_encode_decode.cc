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
REGISTER_OP("CenterBoxesEncode")
    .Attr("T: {float,double,int32,int64}")
	.Attr("max_box_nr:int")
	.Attr("num_classes:int")
	.Attr("gaussian_iou:float")
    .Input("gbboxes: T")
    .Input("glabels: int32")
    .Input("glength: int32")
    .Input("output_size: int32")
	.Output("output_heatmaps_tl:T")
	.Output("output_heatmaps_br:T")
	.Output("output_heatmaps_c:T")
	.Output("output_offset:T")
	.Output("output_tags:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            const auto input_shape0 = c->input(0);
            const auto batch_size = c->Dim(input_shape0,0);
            int num_classes;
            int max_box_nr;
            c->GetAttr("num_classes",&num_classes);
            c->GetAttr("max_box_nr",&max_box_nr);
            auto shape0 = c->MakeShape({batch_size,-1,-1,num_classes});
            auto shape1 = c->MakeShape({batch_size,max_box_nr,6});
            auto shape2 = c->MakeShape({batch_size,max_box_nr,3});

            for(auto i=0; i<3; ++i)
			    c->set_output(i, shape0);
			c->set_output(3, shape1);
			c->set_output(4, shape2);
			return Status::OK();
			});

template <typename Device,typename T>
class CenterBoxesEncodeOp: public OpKernel {
	public:
		explicit CenterBoxesEncodeOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("max_box_nr", &max_box_nr_));
			OP_REQUIRES_OK(context, context->GetAttr("num_classes", &num_classes_));
			OP_REQUIRES_OK(context, context->GetAttr("gaussian_iou", &gaussian_iou_));
		}

		void Compute(OpKernelContext* context) override
        {
            TIME_THISV1("CenterBoxesEncode");
            const Tensor &_gbboxes    = context->input(0);
            const Tensor &_glabels    = context->input(1);
            const Tensor &_gsize      = context->input(2);
            auto          gbboxes     = _gbboxes.template tensor<T,3>();
            auto          glabels     = _glabels.template tensor<int,2>();
            auto          gsize       = _gsize.template tensor<int,1>();
            auto          output_size = context->input(3).template flat<int>().data();
            const auto    batch_size  = _gbboxes.dim_size(0);

            OP_REQUIRES(context, _gbboxes.dims() == 3, errors::InvalidArgument("box data must be 3-dimensional"));
            OP_REQUIRES(context, _glabels.dims() == 2, errors::InvalidArgument("labels data must be 2-dimensional"));
            OP_REQUIRES(context, _gsize.dims() == 1, errors::InvalidArgument("gsize data must be 1-dimensional"));

            int           dims_4d[4]            = {int(batch_size),output_size[0],output_size[1],num_classes_};
            int           dims_3d0[3]           = {int(batch_size),max_box_nr_,6};
            int           dims_3d1[3]           = {int(batch_size),max_box_nr_,3};
            TensorShape  outshape0;
            TensorShape  outshape1;
            TensorShape  outshape2;
            Tensor      *output_heatmaps_tl = NULL;
            Tensor      *output_heatmaps_br = NULL;
            Tensor      *output_heatmaps_c  = NULL;
            Tensor      *output_tags        = NULL;
            Tensor      *output_offset      = NULL;

            TensorShapeUtils::MakeShape(dims_4d, 4, &outshape0);
            TensorShapeUtils::MakeShape(dims_3d0, 3, &outshape1);
            TensorShapeUtils::MakeShape(dims_3d1, 3, &outshape2);


            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_heatmaps_tl));
            OP_REQUIRES_OK(context, context->allocate_output(1, outshape0, &output_heatmaps_br));
            OP_REQUIRES_OK(context, context->allocate_output(2, outshape0, &output_heatmaps_c));
            OP_REQUIRES_OK(context, context->allocate_output(3, outshape1, &output_offset));
            OP_REQUIRES_OK(context, context->allocate_output(4, outshape2, &output_tags));

            auto heatmaps_tl = output_heatmaps_tl->template tensor<T,4>();
            auto heatmaps_br = output_heatmaps_br->template tensor<T,4>();
            auto heatmaps_c  = output_heatmaps_c->template tensor<T,4>();
            auto offsets     = output_offset->template tensor<T,3>();
            auto tags        = output_tags->template tensor<int,3>();
            tags.setZero();
            offsets.setZero();
            heatmaps_tl.setZero();
            heatmaps_br.setZero();
            heatmaps_c.setZero();
            for(auto i=0; i<batch_size; ++i) {
                for(auto j=0; j<gsize(i); ++j) {
                    const auto fytl = gbboxes(i,j,0)*(output_size[0]-1);
                    const auto fxtl = gbboxes(i,j,1)*(output_size[1]-1);
                    const auto fybr = gbboxes(i,j,2)*(output_size[0]-1);
                    const auto fxbr = gbboxes(i,j,3)*(output_size[1]-1);
                    const auto fyc = (fytl+fybr)/2;
                    const auto fxc = (fxtl+fxbr)/2;
                    const auto ytl = int(fytl+0.5);
                    const auto xtl = int(fxtl+0.5);
                    const auto ybr = int(fybr+0.5);
                    const auto xbr = int(fxbr+0.5);
                    const auto yc = int(fyc+0.5);
                    const auto xc = int(fxc+0.5);
                    const auto r0 = get_gaussian_radius(fybr-fytl,fxbr-fxtl,gaussian_iou_);
                    const auto r1 = min<float>(r0,min(fybr-fytl,fxbr-fxtl)/2.0);
                    const auto label = glabels(i,j);
                    draw_gaussian(heatmaps_tl,xtl,ytl,r0,i,label);
                    draw_gaussian(heatmaps_br,xbr,ybr,r0,i,label);
                    draw_gaussian(heatmaps_c,fxc,fyc,r1,i,label);
                    offsets(i,j,0) = fytl-ytl;
                    offsets(i,j,1) = fxtl-xtl;
                    offsets(i,j,2) = fybr-ybr;
                    offsets(i,j,3) = fxbr-xbr;
                    offsets(i,j,4) = fyc-yc;
                    offsets(i,j,5) = fxc-xc;
                    tags(i,j,0) = ytl*output_size[1]+xtl;
                    tags(i,j,1) = ybr*output_size[1]+xbr;
                    tags(i,j,2) = yc*output_size[1]+xc;
                }
            }
        }
        template<typename DT>
        static void draw_gaussian(DT& data,int cx,int cy,float radius,int batch_index,int class_index,float k=1.0)
        {
            const auto width  = data.dimension(2);
            const auto height = data.dimension(1);
            const auto xtl    = max(0,int(cx-radius));
            const auto ytl    = max(0,int(cy-radius));
            const auto xbr    = min<int>(width,int(cx+radius+1));
            const auto ybr    = min<int>(height,int(cy+radius+1));
            const auto sigma  = radius/3;

            for(auto x=xtl; x<xbr; ++x) {
                for(auto y=ytl; y<ybr; ++y) {
                    auto dx = x-cx;
                    auto dy = y-cy;
                    auto v = exp(-(dx*dx+dy*dy)/(2*sigma*sigma))*k;
                    data(batch_index,y,x,class_index) = max(data(batch_index,y,x,class_index),v);
                }
            }
        }
	private:
        int max_box_nr_;
        int num_classes_;
        float gaussian_iou_ = 0.7f;
};
REGISTER_KERNEL_BUILDER(Name("CenterBoxesEncode").Device(DEVICE_CPU).TypeConstraint<float>("T"), CenterBoxesEncodeOp<CPUDevice, float>);
