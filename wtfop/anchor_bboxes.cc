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
#include "wtoolkit.h"

using namespace tensorflow;
using namespace std;

typedef Eigen::ThreadPoolDevice CPUDevice;
/*
 * scales:list of float
 * aspect_ratios: list of floats
 * shape:[H,W]
 * size:[h,w]: use the same metrics as scales does
 */
REGISTER_OP("AnchorGenerator")
    .Attr("scales: list(float)")
    .Attr("aspect_ratios:list(float)")
	.Input("shape:int32")
	.Input("size:float")
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
			auto          in_size = _in_size.tensor<float,1>();

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

/*
 * scales:list of float
 * aspect_ratios: list of floats
 * shape:[H,W]
 * size:[h,w]: use the same metrics as scales does
 */
REGISTER_OP("MultiAnchorGenerator")
    .Attr("scales: list(float)")
    .Attr("aspect_ratios:list(float)")
	.Input("shape:int32")
	.Input("size:float")
	.Output("anchors:float")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			c->set_output(0, c->Matrix(-1, 4));
			return Status::OK();
			});

template <typename Device, typename T>
class MultiAnchorGeneratorOp: public OpKernel {
	public:
        explicit MultiAnchorGeneratorOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("scales", &scales_));
            OP_REQUIRES_OK(context, context->GetAttr("aspect_ratios", &aspect_ratios_));
            const auto delta = 1e-3;
            for(auto i=0; i<aspect_ratios_.size()-1; ++i) {
                if((fabs(aspect_ratios_[i]*aspect_ratios_[i+1]-1.0)<delta)
                        && (fabs(scales_[i]-scales_[i+1])<delta)
                        && (aspect_ratios_[i]>aspect_ratios_[i+1]))
                    swap(aspect_ratios_[i],aspect_ratios_[i+1]);
            }
        }

		void Compute(OpKernelContext* context) override
		{
			const Tensor &_in_shape= context->input(0);
			const Tensor &_in_size = context->input(1);
			auto          in_shape = _in_shape.tensor<int,1>();
			auto          in_size = _in_size.tensor<float,1>();

			OP_REQUIRES(context, _in_shape.dims() == 1, errors::InvalidArgument("shape data must be 1-dimensional"));
			OP_REQUIRES(context, _in_size.dims() == 1, errors::InvalidArgument("size data must be 1-dimensional"));
            const int out_nr = scales_.size()*in_shape(0)*in_shape(1);

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
                    for(auto k=0; k<aspect_ratios_.size(); ++k) {
                        auto a = aspect_ratios_[k];
                        auto s = scales_[k];
                        const auto sa = sqrt(a);
                        const auto hw = s/(2*in_size(1))/sa;
                        const auto hh = s/(2*in_size(0))*sa;
                        oanchors(index,0) = y_pos-hh;
                        oanchors(index,1) = x_pos-hw;
                        oanchors(index,2) = y_pos+hh;
                        oanchors(index,3) = x_pos+hw;
                        ++index;
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
REGISTER_KERNEL_BUILDER(Name("MultiAnchorGenerator").Device(DEVICE_CPU), MultiAnchorGeneratorOp<CPUDevice, float>);
