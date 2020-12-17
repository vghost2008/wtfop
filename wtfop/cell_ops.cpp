#include <stdio.h>
#include <cfloat>
#include <list>
#include <cstdlib>
#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <future>
#include <assert.h>
#include <boost/algorithm/clamp.hpp>
#include "tensorflow/core/framework/common_shape_fns.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;
using namespace std;
typedef Eigen::ThreadPoolDevice CPUDevice;
REGISTER_OP("CellEncodeLabel")
    .Attr("T: {int32, int64}")
 	.Attr("num_classes: int")
    .Input("tensor: T")
	.Output("tensor_o:T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			const auto batch_size = c->Dim(c->input(0),0);
            int num_classes = 0;
            c->GetAttr("num_classes",&num_classes);
            auto shape = c->MakeShape({batch_size,num_classes});
            c->set_output(0,shape);
    return Status::OK();
    });

template <typename Device, typename T>
class CellEncodeLabelOp: public OpKernel {
	public:
		explicit CellEncodeLabelOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("num_classes", &num_classes_));
            //第一级分鳞，腺，TRI,CC,....
            //0-7表示第一级
            label_map_level0_[1] = 0;
            label_map_level0_[2] = 0;
            label_map_level0_[3] = 2;
            label_map_level0_[4] = 3;
            label_map_level0_[5] = 1;
            label_map_level0_[6] = 4;
            label_map_level0_[7] = 5;
            label_map_level0_[8] = 0;
            label_map_level0_[9] = 6;
            label_map_level0_[10] = 7;
            label_map_level0_[11] = 0;
            label_map_level0_[12] = 0;
            //第二级分低级别鳞状病变,高级别鳞状病变
            //8-9表示第二级
            label_map_level1_[1] = 9;
            label_map_level1_[11] = 9;
            label_map_level1_[2] = 8;
            label_map_level1_[8] = 8;
            label_map_level1_[12] = 8;
            //第三级分（ASCUS,LSIL0,(ASCH,HSIL,SCC)
            //10-14表示第三级
            label_map_level2_[1] = 11;
            label_map_level2_[11] = 10;
            label_map_level2_[2] = 13;
            label_map_level2_[8] = 14;
            label_map_level2_[12] = 12;
		}

		void Compute(OpKernelContext* context) override
        {
            const Tensor &_tensor     = context->input(0);
            auto          tensor_flat = _tensor.flat<T>().data();
            const auto    batch_size  = _tensor.dim_size(0);
            const auto    data_nr     = _tensor.NumElements();

			OP_REQUIRES(context, _tensor.dims() == 1, errors::InvalidArgument("input must be 1-dimensional"));

            int dims_2d[] = {batch_size,num_classes_};

            TensorShape outshape;
            TensorShapeUtils::MakeShape(dims_2d, 2, &outshape);

            Tensor      *output_tensor = nullptr;

            OP_REQUIRES_OK(context,context->allocate_output(0,outshape,&output_tensor));

            auto o_tensor = output_tensor->template tensor<T,2>();

            o_tensor.setZero();

            for(auto i=0; i<data_nr; ++i) {
                const auto l = tensor_flat[i];

                o_tensor(i,label_map_level0_[l]) = 1;

                auto it = label_map_level1_.find(l);

                if(label_map_level1_.end() == it) 
                    continue;
                o_tensor(i,it->second) = 1;
                auto jt = label_map_level2_.find(l);
                if(label_map_level2_.end() == jt) 
                    continue;
                o_tensor(i,jt->second) = 1;
            }
        }
	private:
        int num_classes_ = 0;
        unordered_map<int,int> label_map_level0_;
        unordered_map<int,int> label_map_level1_;
        unordered_map<int,int> label_map_level2_;
};
REGISTER_KERNEL_BUILDER(Name("CellEncodeLabel").Device(DEVICE_CPU).TypeConstraint<int>("T"), CellEncodeLabelOp<CPUDevice, int>);
REGISTER_KERNEL_BUILDER(Name("CellEncodeLabel").Device(DEVICE_CPU).TypeConstraint<tensorflow::int64>("T"), CellEncodeLabelOp<CPUDevice, tensorflow::int64>);

REGISTER_OP("CellDecodeLabel")
    .Attr("T: {float, double}")
    .Input("tensor: T") //probability
	.Output("tensor_o:T")
	.SetShapeFn(shape_inference::UnchangedShape);

template <typename Device, typename T>
class CellDecodeLabelOp: public OpKernel {
	public:
		explicit CellDecodeLabelOp(OpKernelConstruction* context) : OpKernel(context) {
            //第一级分鳞，腺，TRI,CC,....
            //0-7表示第一级
            label_map_level0_[1] = 5;
            label_map_level0_[2] = 3;
            label_map_level0_[3] = 4;
            label_map_level0_[4] = 6;
            label_map_level0_[5] = 7;
            label_map_level0_[6] = 9;
            label_map_level0_[7] = 10;
		}

		void Compute(OpKernelContext* context) override
        {
            const Tensor &_tensor    = context->input(0);

			OP_REQUIRES(context, _tensor.dims() == 2, errors::InvalidArgument("input must be 2-dimensional"));

            auto          tensor     = _tensor.template tensor<T,2>();
            const auto    batch_size = _tensor.dim_size(0);
            const auto    C          = _tensor.dim_size(1);
            const auto    kThreadNr  = 512;


            TensorShape  output_shape  = _tensor.shape();
            int dims_2d[] = {batch_size,C};

            TensorShape outshape;
            TensorShapeUtils::MakeShape(dims_2d, 2, &outshape);

            Tensor      *output_tensor = nullptr;

            OP_REQUIRES_OK(context,context->allocate_output(0,output_shape,&output_tensor));

            auto o_tensor = output_tensor->template tensor<T,2>();
            auto i_data = _tensor.template flat<T>().data();

            o_tensor.setZero();

            auto fn = [&](int begin,int end) {
                for(auto i=begin; i<end; ++i) {
                    for(auto it=label_map_level0_.begin(); it!=label_map_level0_.end(); ++it) {
                        o_tensor(i,it->second) = tensor(i,it->first);
                    }
                    if(tensor(i,8)>=tensor(i,9)) { //H
                        auto d = i_data+i*C+12;
                        auto it = max_element(d,d+3);
                        auto dis = it-d;
                        auto p = max<T>(d[dis],tensor(i,0));

                        if(dis==0){
                            o_tensor(i,12) = p;
                        } else if(dis==1) {
                            o_tensor(i,2) = p;
                        } else {
                            o_tensor(i,8) = p;
                        }
                    } else {
                        if(tensor(i,10)>=tensor(i,11)) {
                            o_tensor(i,11) = max<T>(tensor(i,0),tensor(i,10));
                        } else {
                            o_tensor(i,1) = max<T>(tensor(i,0),tensor(i,11));
                        }
                    }
                }
            };

            list<future<void>> furs;

            for(auto i=0; i<batch_size; i += kThreadNr) {
                furs.push_back(std::async(std::launch::async,fn,i,min<int>(i+kThreadNr,batch_size)));
            }
        }
	private:
        unordered_map<int,int> label_map_level0_;
};
REGISTER_KERNEL_BUILDER(Name("CellDecodeLabel").Device(DEVICE_CPU).TypeConstraint<float>("T"), CellDecodeLabelOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("CellDecodeLabel").Device(DEVICE_CPU).TypeConstraint<double>("T"), CellDecodeLabelOp<CPUDevice, double>);
