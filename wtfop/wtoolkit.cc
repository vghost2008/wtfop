#include <stdio.h>
#include <cfloat>
#include <list>
#include <cstdlib>
#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm>
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
REGISTER_OP("DrawPoints")
    .Attr("T: {float, double}")
	.Attr("color:list(float)")
	.Attr("point_size:int")
    .Input("image: T")
    .Input("points: T")
	.Output("output:T")
	.SetShapeFn(shape_inference::UnchangedShape);

template <typename Device, typename T>
class DrawPointsOp: public OpKernel {
	public:
		explicit DrawPointsOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("color", &color_));
			OP_REQUIRES_OK(context, context->GetAttr("point_size", &point_size_));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &_image      = context->input(0);
			const Tensor &_points     = context->input(1);
			auto          image_flat  = _image.flat<T>();
			auto          points_flat = _points.flat<T>();
			const auto    points_nr   = _points.dim_size(0);
			const auto    width       = _image.dim_size(1);
			const auto    height      = _image.dim_size(0);
			const auto    channels    = _image.dim_size(2);

			OP_REQUIRES(context, _image.dims() == 3, errors::InvalidArgument("images data must be 3-dimensional"));
			OP_REQUIRES(context, _points.dims() == 2, errors::InvalidArgument("points data must be 2-dimensional"));
			OP_REQUIRES(context, color_.size() > 0, errors::InvalidArgument("empty color"));

			TensorShape  output_shape  = _image.shape();
			Tensor      *output_tensor = nullptr;

			OP_REQUIRES_OK(context,context->allocate_output(0,output_shape,&output_tensor));

			if(!output_tensor->CopyFrom(_image,output_shape))
				return;

			auto       image  = output_tensor->tensor<T,3>();      
			const auto points = _points.tensor<T,2>();

			while(color_.size()<channels)
				color_.push_back(color_.back());

			auto shard = [&]
				(int64 start, int64 limit) {
					for(auto i=start; i<limit; ++i) {

						const auto x     = points(i,1);
						const auto y     = points(i,0);
						const auto beg_x = std::max<int>(x-point_size_,0);
						const auto end_x = std::min<int>(x+point_size_,width-1);
						const auto beg_y = std::max<int>(y-point_size_,0);
						const auto end_y = std::min<int>(y+point_size_,height-1);

						for(auto j=beg_x; j<=end_x; ++j) {
							for(auto k=beg_y; k<=end_y; ++k) {
								for(auto m=0; m<channels; ++m) {
									image(k,j,m) = color_.at(m);
								}
							}
						}
					}
				};
			shard(0,points_nr);
			/*const DeviceBase::CpuWorkerThreads& worker_threads =
			*(context->device()->tensorflow_cpu_worker_threads());
			const int64 total         = points_nr;
			const int64 cost_per_unit = 2;*/
			//Shard(worker_threads.num_threads, worker_threads.workers,total,cost_per_unit, shard);
		}
	private:
		std::vector<float> color_;
		int point_size_;
};
REGISTER_KERNEL_BUILDER(Name("DrawPoints").Device(DEVICE_CPU).TypeConstraint<float>("T"), DrawPointsOp<CPUDevice, float>);
/*
 * phy_max:返回的begin_index与end_index之间最多差phy_max
 * max：begin_index,end_index的最大值，
 * hint:提示值，生成的区间至少要包含hint中的一个值
 * 输出:
 * [begin_index,end_index)
 * hint:输入的hint中在[begin_index,end_index之间的部分
 */
REGISTER_OP("RandomRange")
	.Attr("phy_max:int")
    .Input("max: int32")
    .Input("hint: int32")
	.Output("oindex:int32")
	.Output("ohint:int32")
    .SetShapeFn([](shape_inference::InferenceContext* c){
        auto shape0 = c->Vector(2);
        auto shape1 = c->Vector(c->UnknownDim());
        c->set_output(0,shape0);
        c->set_output(1,shape1);
		return Status::OK();
    });

class RandomRange: public OpKernel {
	public:
		explicit RandomRange(OpKernelConstruction* context) : OpKernel(context) {
            std::srand(::time(nullptr));
			OP_REQUIRES_OK(context, context->GetAttr("phy_max", &phy_max_));
		}

		void Compute(OpKernelContext* context) override
        {
            const Tensor &_max      = context->input(0);
            const Tensor &_hint     = context->input(1);
            auto          max       = _max.flat<int>().data()[0];
            auto          hint_flat = _hint.flat<int>();
            const auto    hint_size = _hint.dim_size(0);
            int           index     = 0;
            int           beg_index = 0;
            int           end_index = max;

            OP_REQUIRES(context, _max.dims() <=1, errors::InvalidArgument("max data must be 1/0-dimensional"));
            OP_REQUIRES(context, _hint.dims() == 1, errors::InvalidArgument("hint data must be 1-dimensional"));

            TensorShape  output_shape0;
            Tensor      *output_tensor0 = nullptr;
            Tensor      *output_tensor1 = nullptr;
            const int    dim0[]           = {2};

            TensorShapeUtils::MakeShape(dim0,1,&output_shape0);

            OP_REQUIRES_OK(context,context->allocate_output(0,output_shape0,&output_tensor0));

            auto oindex = output_tensor0->flat<int>();

            if(max> phy_max_) {
                const auto index = std::rand()%hint_size;
                const auto base_index = hint_flat.data()[index];
                vector<int> outdata;

                beg_index = base_index-(phy_max_/2);
                end_index = beg_index+phy_max_;
                if(beg_index<0) {
                    beg_index = 0;
                    end_index = phy_max_;
                } else if (end_index>=max) {
                    end_index = max;
                    beg_index = max-phy_max_;
                }
                std::copy_if(hint_flat.data(),hint_flat.data()+hint_size,std::back_inserter(outdata),[beg_index,end_index](int v){ return (v>=beg_index)&& (v<end_index); });

                TensorShape output_shape1;
                const int   dim1[]        = {int(outdata.size())};
                TensorShapeUtils::MakeShape(dim1,1,&output_shape1);

                OP_REQUIRES_OK(context,context->allocate_output(1,output_shape1,&output_tensor1));
                auto ohint = output_tensor1->flat<int>();
                std::copy(outdata.begin(),outdata.end(),ohint.data());

            } else {
                TensorShape  output_shape1  = _hint.shape();
                OP_REQUIRES_OK(context,context->allocate_output(1,output_shape1,&output_tensor1));

                output_tensor1->CopyFrom(_hint,output_shape1);
            }
            oindex.data()[0] = beg_index;
            oindex.data()[1] = end_index;
        }
	private:
		int phy_max_;
};
REGISTER_KERNEL_BUILDER(Name("RandomRange").Device(DEVICE_CPU), RandomRange);

/*
 * 将输入的整数按指定的方式进行映射
 */
REGISTER_OP("IntHash")
	.Attr("T:{int32,int64}")
	.Attr("key:list(int)")
	.Attr("value:list(int)")
    .Input("input:T")
	.Output("output:T")
    .SetShapeFn(shape_inference::UnchangedShape);

template <typename Device, typename T>
class IntHash: public OpKernel {
	public:
		explicit IntHash(OpKernelConstruction* context) : OpKernel(context) {
            vector<int> key;
            vector<int> value;
			OP_REQUIRES_OK(context, context->GetAttr("key", &key));
			OP_REQUIRES_OK(context, context->GetAttr("value", &value));
            const auto nr = std::min(key.size(),value.size());
            for(auto i=0; i<nr; ++i)
                dict_[key[i]] = value[i];
		}

		void Compute(OpKernelContext* context) override
        {
            const Tensor &_input = context->input(0);
            Tensor      *output_tensor0 = nullptr;

            OP_REQUIRES_OK(context,context->allocate_output(0,_input.shape(),&output_tensor0));

            auto input  = _input.flat<T>();
            auto output = output_tensor0->flat<T>();

            for(auto i=0; i<input.size(); ++i) {
                const auto v = input.data()[i];
                const auto it = dict_.find(v);
                if(it != dict_.end()) 
                    output.data()[i] = it->second;
                else
                    output.data()[i] = 65536;
            }
        }
	private:
		map<int,int> dict_;
};
REGISTER_KERNEL_BUILDER(Name("IntHash").Device(DEVICE_CPU).TypeConstraint<int>("T"), IntHash<CPUDevice, int>);
