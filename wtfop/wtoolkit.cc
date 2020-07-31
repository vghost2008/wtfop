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
 * phy_max:返回的begin_index与end_index之间最多差phy_max(强制限制)
 * max：begin_index,end_index的最大值，
 * hint:提示值，生成的区间至少要包含hint中的一个值, 要求其值位于[0,max)之间
 * 输出:
 * oindex:[begin_index,end_index) shape=[2]的tensor用于表示一个范围
 * ohint:输入的hint中在[begin_index,end_index之间的部分
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
REGISTER_KERNEL_BUILDER(Name("IntHash").Device(DEVICE_CPU).TypeConstraint<tensorflow::int64>("T"), IntHash<CPUDevice, tensorflow::int64>);
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

/*
 * labels:[batch_size,nr]
 * ids:[batch_size,nr]
 * line_no[batch_size,nr]
 *return: 
 * output:[batch_size,sample_nr,3] (id0,id1_pos,id2_neg) 内容为相应的索引
 */
REGISTER_OP("SampleLabels")
    .Attr("T: {int32, int64}")
	.Attr("sample_nr:int")
    .Input("labels: T")
    .Input("ids: T")
    .Input("line_no: T")
	.Output("data:T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            int sample_nr = 0;
            const auto input_shape0 = c->input(0);
            const auto batch_size = c->Dim(input_shape0,0);
            c->GetAttr("sample_nr",&sample_nr);
            auto shape0 = c->MakeShape({batch_size,sample_nr,3});
			c->set_output(0, shape0);
			return Status::OK();
			});

template <typename Device, typename T>
class SampleLabelsOp: public OpKernel {
    public:
        explicit SampleLabelsOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("sample_nr", &sample_nr_));
        }
        void Compute(OpKernelContext* context) override
        {
            const Tensor &_labels     =  context->input(0);
            const Tensor &_ids        =  context->input(1);
            const Tensor &_line_no    =  context->input(2);

            OP_REQUIRES(context, _labels.dims() == 2, errors::InvalidArgument("labels must be 2-dimensional"));
            OP_REQUIRES(context, _line_no.dims() == 2, errors::InvalidArgument("line no must be 2-dimensional"));
            OP_REQUIRES(context, _ids.dims() == 2, errors::InvalidArgument("ids must be 2-dimensional"));


            auto          labels      =  _labels.tensor<T,2>();
            auto          ids         =  _ids.tensor<T,2>();
            auto          line_no     =  _line_no.tensor<T,2>();
            auto          batch_size  =  labels.dimension(0);
            const auto    line_no_br  =  line_no.dimension(0);
            int dims_3d[] = {batch_size,sample_nr_,3};
            TensorShape outshape0;
            Tensor *output_data = NULL;

            TensorShapeUtils::MakeShape(dims_3d, 3, &outshape0);

            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_data));

            auto out_tensor = output_data->tensor<T,3>();
            list<future<vector<tuple<T,T,T>>>> res;

            for(auto i=0; i<batch_size; ++i) {
                res.emplace_back(async(launch::async,&SampleLabelsOp<Device,T>::sample_one_batch,Eigen::Tensor<T,1,Eigen::RowMajor>(ids.chip(i,0)),
                Eigen::Tensor<T,1,Eigen::RowMajor>(labels.chip(i,0)),
                line_no.chip(line_no_br>1?i:0,0),
                sample_nr_));
            }

            for(auto i=0; i<batch_size; ++i) {
                auto data = next(res.begin(),i)->get();
                for(auto j=0; j<sample_nr_; ++j) {
                    out_tensor(i,j,0) = std::get<0>(data[j]);
                    out_tensor(i,j,1) = std::get<1>(data[j]);
                    out_tensor(i,j,2) = std::get<2>(data[j]);
                }
            }
        }
        static vector<tuple<T,T,T>> sample_one_batch(const Eigen::Tensor<T,1,Eigen::RowMajor>& ids,
        const Eigen::Tensor<T,1,Eigen::RowMajor>& labels,
        const Eigen::Tensor<T,1,Eigen::RowMajor>& line_no,
        int sample_nr) {
            //instance id->box index
            map<T,vector<int>> datas;
            map<T,int> id_to_label;
            map<int,vector<T>> label_to_id;
           const auto kDelta = 3;

            assert(ids.dimension(0)>0);
            const auto data_nr = ids.dimension(0);
            auto default_neg = data_nr-1;

            for(auto i=0; i<data_nr; ++i) {
                auto id = ids(i);
                if((id<1) || (labels(i)<1)) continue;
                auto it = datas.find(id);
                if(it == datas.end()) {
                    datas[id] = vector<int>({i});
                } else {
                    it->second.push_back(i);
                }
                const auto l = labels[i];
                id_to_label[id] = l;
            }
            for(auto it=id_to_label.begin(); it!=id_to_label.end(); ++it) {
                const auto id = it->first;
                const auto l = it->second;
                if(label_to_id.find(l) == label_to_id.end()) {
                    label_to_id[l] = vector<T>({id});
                } else {
                    label_to_id[l].push_back(id);
                }
            }
            /*
             * 用于简化采样时的操作
             */
            for(auto it=datas.begin(); it!=datas.end(); ++it) {
                if(it->second.size()==1) {
                    it->second.push_back(it->second[0]);
                }
            }

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, data_nr);

            if(datas.size() == 1) {
                auto it0 = datas.begin();
                auto ids = datas.begin()->second;
                int v0;
                int v1;
                std::tie(v0,v1) = sample_two_pos_int(ids,kDelta,line_no,[&dis,&gen]{return dis(gen);});
                auto neg_idx = (data_nr-1);
                if(find(ids.begin(),ids.end(),neg_idx) != ids.end()) {
                    neg_idx = 0;
                    if(find(ids.begin(),ids.end(),neg_idx) != ids.end()) {
                        cout<<"No neg idx find in sample_one_batch."<<endl;
                    }
                }
                vector<tuple<T,T,T>> res(sample_nr,make_tuple(v0,v1,neg_idx));
                return res;
            } else {
                /*
                 * 至少有两个以上的目标
                 */
                vector<tuple<T,T,T>> res(sample_nr);

                generate(res.begin(),res.end(),[&gen,&dis,&datas,&label_to_id,&id_to_label,&line_no](){
                        const auto id_index0 = dis(gen)%datas.size();
                        const auto id_index1 = sample_neg_data(datas,id_to_label,label_to_id,id_index0,[&dis,&gen]{return dis(gen);});
                        auto it0 = next(datas.begin(),id_index0);
                        auto it1 = next(datas.begin(),id_index1);
                        int v0;
                        int v1;
                        std::tie(v0,v1) = sample_two_pos_int(it0->second,kDelta,line_no,[&dis,&gen]{return dis(gen);});
                        auto id1_idx = dis(gen)%it1->second.size();
                        auto v2 = it1->second[id1_idx];
                        return make_tuple(v0,v1,v2);
                        });
                return res;
            } 
        }
        template<typename RFunc>
        static int sample_neg_data(const map<T,vector<int>>& id_to_index,const map<T,int>& id_to_label,const map<int,vector<T>>& label_to_id,int id_index,RFunc func) {
            /*
             * 尽量从具有相同label的实例中采样
             */
            auto id = next(id_to_index.begin(),id_index)->first;
            const auto label = id_to_label.at(id);
            auto ids = label_to_id.at(label);
            if(ids.size() == 1) {
                return sample_int_exclude(id_to_index.size(),id_index,func);
            } else {
                auto _index = distance(ids.begin(),find(ids.begin(),ids.end(),id));
                auto index = sample_int_exclude(ids.size(),_index,func);
                auto id1 = ids[index];
                assert(id_to_label.at(id1)==label);
                assert(id1!=id);
                auto it = id_to_index.find(id1);
                return distance(id_to_index.begin(),it);
            }

        }
        static pair<int,int> sample_two_int(int max_val,int delta,auto func) {
            const int v0 = func()%max_val;
            if(max_val<=delta) {  
                const auto v1 = sample_int_exclude(max_val,v0,func);
                return make_pair(v0,v1);
            }
            auto d_v1 = (func()%delta)+1;
            if(v0<max_val-1) {
                auto v1 = min(v0+d_v1,max_val-1);
                return make_pair(v0,v1);
            } else {
                return make_pair(v0,v0-d_v1);
            }
        };
        static pair<int,int> sample_two_pos_int(const vector<int>& indexs,int delta,
            const Eigen::Tensor<T,1,Eigen::RowMajor>& line_no,
            auto func) {
            /*
             * 尽量在不同的行采样
             */
            const int v0 = func()%indexs.size();
            const int index0 = indexs[v0];
            const int line_no0 = line_no(index0);
            vector<int> a_indexs;

            a_indexs.reserve(indexs.size());

            copy_if(indexs.begin(),indexs.end(),back_inserter(a_indexs),[line_no0,delta,&line_no](int v) {
                auto line_no1 = line_no(v);
                if(line_no1==line_no0) return false;
                return fabs(line_no1-line_no0)<=delta;
            });

            if(a_indexs.size()==0) {
                const auto v1 = sample_int_exclude(indexs.size(),v0,func);
                const int index1 = indexs[v1];
                return make_pair(index0,index1);
            }

            const auto v1 = func()%a_indexs.size();
            const auto index1 = a_indexs[v1];

            return make_pair(index0,index1);
        };
        template<typename RFunc>
        static int sample_int_exclude(int max_val,int exclude_v,RFunc func)
        {
            assert(max_val>0);
            auto res = func()%(max_val-1);
            return (res==exclude_v)?res+1:res;
        }
    private:
        int sample_nr_ = 0;
};
REGISTER_KERNEL_BUILDER(Name("SampleLabels").Device(DEVICE_CPU).TypeConstraint<int>("T"), SampleLabelsOp<CPUDevice, int>);
REGISTER_KERNEL_BUILDER(Name("SampleLabels").Device(DEVICE_CPU).TypeConstraint<tensorflow::int64>("T"), SampleLabelsOp<CPUDevice, tensorflow::int64>);

/*
 * data:[nr,nr] (i,j)表示i到j的距离
 * labels:[nr]
 * bboxes:[nr,4]
 * threshold:
 * dis_threshold:[2](x,y)
 * output:[nr]
 */
REGISTER_OP("MergeLineBoxes")
    .Attr("T: {int32, int64}")
	.Attr("threshold:float")
	.Attr("dis_threshold:list(float)")
    .Input("data: float")
    .Input("labels: T")
    .Input("bboxes:float")
	.Output("ids:T")
	.Output("unique_ids:T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            const auto input_shape0 = c->input(1);
			c->set_output(0, input_shape0);
			return Status::OK();
			});

template <typename Device, typename T>
class MergeLineBoxesOp: public OpKernel {
    public:
        explicit MergeLineBoxesOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("dis_threshold", &dis_threshold_));
            OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold_));
            OP_REQUIRES(context, dis_threshold_.size() == 2, errors::InvalidArgument("Threshold must be contains two elements."));
        }
        void Compute(OpKernelContext* context) override
        {
            const Tensor &_data      = context->input(0);
            const Tensor &_labels    = context->input(1);
            const Tensor &_bboxes    = context->input(2);

            OP_REQUIRES(context, _labels.dims() == 1, errors::InvalidArgument("labels must be 1-dimensional"));
            OP_REQUIRES(context, _data.dims() == 2, errors::InvalidArgument("data must be 2-dimensional"));
            OP_REQUIRES(context, _bboxes.dims() == 2, errors::InvalidArgument("bboxes must be 2-dimensional"));

            auto          data       = _data.tensor<float,2>();
            auto          bboxes     = _bboxes.tensor<float,2>();
            auto          labels     = _labels.tensor<T,1>();
            auto          batch_size = labels.dimension(0);
            const auto     data_nr = labels.dimension(0);
            list<future<vector<int>>> res;
            auto res_data = process(data,labels,bboxes,data_nr);
            vector<int> res_data1 = res_data;

            sort(res_data1.begin(),res_data1.end());

            auto last = unique(res_data1.begin(),res_data1.end());
            res_data1.erase(last,res_data1.end());

            int dims_1d[] = {data_nr};
            int dims_1d2[] = {res_data1.size()};
            TensorShape outshape0;
            TensorShape outshape1;

            TensorShapeUtils::MakeShape(dims_1d, 1, &outshape0);
            TensorShapeUtils::MakeShape(dims_1d2, 1, &outshape1);

            Tensor *output_data = NULL;
            Tensor *output_data1 = NULL;

            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_data));
            OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_data1));

            auto out_tensor = output_data->tensor<T,1>();
            auto out_tensor1 = output_data1->tensor<T,1>();

            out_tensor.setConstant(0);

            for(auto j=0; j<data_nr; ++j) {
                out_tensor(j) = res_data[j];
            }
            for(auto j=0; j<res_data1.size(); ++j) {
                out_tensor1(j) = res_data1[j];
            }
        }
        static auto get_distance(const Eigen::Tensor<float,1,Eigen::RowMajor>& box0,
        const Eigen::Tensor<float,1,Eigen::RowMajor>& box1
        ) {
            float xdis;
            const float ydis = fabs(box0(0)+box0(2)-box1(0)-box1(2))/2.0f;
            const float box_h = (box0(2)-box0(0));

            if(fabs(box_h-(box1(2)-box1(0)))>1e-2) {
                cout<<"Error box height "<<box_h<<", "<<(box1(2)-box1(0))<<endl;
            }

            if(ydis<0.8*box_h)
                return make_pair(1e8f,1e8f);

            if(box0(1)>=box1(3)) {
                xdis = box0(1)-box1(3);
            } else if(box0(3)<=box1(1)) {
                xdis = box1(1)-box0(3);
            } else {
                xdis = 0.0f;
            }

            return make_pair(xdis,ydis);
        }
        static auto get_distance_matrix(const Eigen::Tensor<T,1,Eigen::RowMajor>& labels,
        const Eigen::Tensor<float,2,Eigen::RowMajor>& bboxes,int data_nr) {

            const auto kMaxDis = 1e8;
            Eigen::Tensor<float,3,Eigen::RowMajor> dis(data_nr,data_nr,2); //dis(x,y)

            dis.setConstant(kMaxDis);

            for(auto i=0; i<data_nr; ++i) {
                dis(i,i,0) = 0;
                dis(i,i,1) = 0;
                for(auto j=i+1; j<data_nr; ++j) {
                    if(labels(i) != labels(j)) continue;
                    const auto b_dis = get_distance(bboxes.chip(i,0),bboxes.chip(j,0));
                    dis(i,j,0) = b_dis.first;
                    dis(i,j,1) = b_dis.second;
                    dis(j,i,0) = b_dis.first;
                    dis(j,i,1) = b_dis.second;
                }
            }
            return dis;
        }
        template<typename DT>
        void label_one(const DT& dis_matrix,
        int index,
        const Eigen::Tensor<float,2,Eigen::RowMajor>& data,
        vector<int>& ids,
        int data_nr) {
            for(auto j=0; j<data_nr; ++j) {
                if(ids[j]>0) continue;
                if((dis_matrix(index,j,0) < dis_threshold_[0])
                        &&(dis_matrix(index,j,1) < dis_threshold_[1])
                        && (data(index,j) <threshold_)){
                    ids[j] = ids[index];
                    label_one(dis_matrix,j,data,ids,data_nr);
                }
            }
        }
        vector<int> process(const Eigen::Tensor<float,2,Eigen::RowMajor>& data,
        const Eigen::Tensor<T,1,Eigen::RowMajor>& labels,
        const Eigen::Tensor<float,2,Eigen::RowMajor>& bboxes,int data_nr) {
            const auto dis_matrix = get_distance_matrix(labels,bboxes,data_nr);
            vector<int> ids(data_nr,0);
            int id = 0;

            for(auto i=0; i<data_nr; ++i) {
                if(ids[i] == 0) {
                    ids[i] = ++id;
                }
                const Eigen::Tensor<float,1,Eigen::RowMajor> data_i = data.chip(i,0);
                label_one(dis_matrix,i,data,ids,data_nr);
            }
            return ids;
        }
    private:
        vector<float> dis_threshold_;
        float threshold_ = 0.0f;
};
REGISTER_KERNEL_BUILDER(Name("MergeLineBoxes").Device(DEVICE_CPU).TypeConstraint<int>("T"), MergeLineBoxesOp<CPUDevice, int>);
REGISTER_KERNEL_BUILDER(Name("MergeLineBoxes").Device(DEVICE_CPU).TypeConstraint<tensorflow::int64>("T"), MergeLineBoxesOp<CPUDevice, tensorflow::int64>);

/*
 * limit:[min_size,max_size], satisfy max(out_size)<=max_size,min(out_size)>=min_size, if min_size/max_size is -1 or 1, means no limit
 * if both min_size and max_size return the input size
 * align:satisfy out_size[0]%align[0] == 0 and out_size[1]%align[1] == 0
 * Try to keep the ratio constant
 */
REGISTER_OP("GetImageResizeSize")
    .Input("size: int32")
	.Input("limit:int32") 
	.Input("align:int32")
	.Output("output_size:int32")
	.SetShapeFn(shape_inference::UnchangedShape);

class GetImageResizeSizeOp: public OpKernel {
        public:
		explicit GetImageResizeSizeOp(OpKernelConstruction* context) : OpKernel(context) {
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &_size = context->input(0);
			const Tensor &_limit= context->input(1);
			const Tensor &_align= context->input(2);

			OP_REQUIRES(context, _size.dims() == 1, errors::InvalidArgument("size must be 1-dimensional"));
			OP_REQUIRES(context, _limit.dims() == 1, errors::InvalidArgument("limit must be 1-dimensional"));
			OP_REQUIRES(context, _align.dims() == 1, errors::InvalidArgument("align must be 1-dimensional"));

			auto          size= _size.tensor<int,1>();
            auto          limit = _limit.flat<int>().data();
            auto          align = _align.flat<int>().data();
            int           out_size[2];
            auto scale = 1.0;
            if((limit[0]<1) && (limit[1]<1)) {
                out_size[0] = size(0);
                out_size[1] = size(1);
            } else if((limit[0]>0) && (limit[1]>0)) {
                if(size(0)<size(1))
                    scale = std::min(float(limit[0])/size(0),float(limit[1])/size(1));
                else
                    scale = std::min(float(limit[0])/size(1),float(limit[1])/size(0));
            } else if(limit[1]<1) {
                if(size(0)<size(1))
                    scale = float(limit[0])/size(0);
                else
                    scale = float(limit[0])/size(1);
            } else if(limit[0]<1) {
                if(size(0)<size(1))
                    scale = float(limit[1])/size(1);
                else
                    scale = float(limit[1])/size(0);
            }
            out_size[0] = size(0)*scale+0.5;
            out_size[1] = size(1)*scale+0.5;
            if(limit[0]>0) {
                if(out_size[0]<limit[0]) 
                    out_size[0] = limit[0];
                else if(out_size[1]<limit[0])
                    out_size[1] = limit[0];
            }

            if(align[1]>1)
                out_size[1] = ((out_size[1]+align[1]-1)/align[1])*align[1];
            if(align[0]>1)
                out_size[0] = ((out_size[0]+align[0]-1)/align[0])*align[0];

            TensorShape  outshape0;
            Tensor      *output_size = nullptr;
            int          dims_1d0[1]  = {2};
            TensorShapeUtils::MakeShape(dims_1d0, 1, &outshape0);
            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_size));
			auto       output_tensor = output_size->tensor<int,1>();      
            output_tensor(0) = out_size[0];
            output_tensor(1) = out_size[1];
        }
};
REGISTER_KERNEL_BUILDER(Name("GetImageResizeSize").Device(DEVICE_CPU), GetImageResizeSizeOp);

/*
 * image[H,W]
 * boxes[N,4], 绝对坐标
 */
REGISTER_OP("FillBBoxes")
    .Attr("T: {float, double}")
    .Attr("v: float")
    .Attr("include_last: bool=True")
    .Input("image: T")
    .Input("bboxes: T")
	.Output("output:T")
    .SetShapeFn([](shape_inference::InferenceContext* c){
        c->set_output(0,c->input(0));
		return Status::OK();
    });

template <typename Device, typename T>
class FillBoxesOp: public OpKernel {
	public:
		explicit FillBoxesOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("v", &v_));
			OP_REQUIRES_OK(context, context->GetAttr("include_last", &include_last_));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &_image      = context->input(0);
			const Tensor &_bboxes     = context->input(1);

			OP_REQUIRES(context, _image.dims() == 2, errors::InvalidArgument("images data must be 2-dimensional"));
			OP_REQUIRES(context, _bboxes.dims() == 2, errors::InvalidArgument("boxes data must be 2-dimensional"));

			auto          image       = _image.tensor<T,2>();
			const auto    bboxes      = _bboxes.tensor<T,2>();
            const auto    box_nr      = _bboxes.dim_size(0);

			TensorShape  output_shape  = _image.shape();
			Tensor      *output_tensor = nullptr;

			OP_REQUIRES_OK(context,context->allocate_output(0,output_shape,&output_tensor));

            auto out_tensor = output_tensor->tensor<T,2>();
            out_tensor = image;
            for(auto i=0; i<box_nr; ++i) 
                draw_a_box(out_tensor,bboxes.chip(i,0));

		}
        template<typename IT>
            void draw_a_box(IT& image,const Eigen::Tensor<T,1,Eigen::RowMajor>& box) {
                //使用float结束，结果更准确
                const auto xmin = max<int>(0,box(1));
                const auto xmax = min<float>(image.dimension(1),box(3));
                const auto ymin = max<int>(0,box(0));
                const auto ymax = min<float>(image.dimension(0),box(2));

                if(include_last_)
                    for(int x=xmin; x<=xmax; ++x) {
                        for(int y=ymin; y<=ymax; ++y) {
                            image(y,x) = v_;
                        }
                    }
                else
                    for(int x=xmin; x<xmax; ++x) {
                        for(int y=ymin; y<ymax; ++y) {
                            image(y,x) = v_;
                        }
                    }
            }
	private:
		float v_ = 1.0;
        bool include_last_ = true;

};
REGISTER_KERNEL_BUILDER(Name("FillBBoxes").Device(DEVICE_CPU).TypeConstraint<float>("T"), FillBoxesOp<CPUDevice, float>);

/*
 * 在data最后一维为True的位置随机选择出nr个,并将其它的设置为False
 * data: [D0,D1,...,Dn] a bool tensor
 * indices:[D0,D1,...,nr] 与返回值相对应的indices
 */
REGISTER_OP("RandomSelect")
    .Attr("nr: int")
    .Attr("sort_indices: bool = False")
    .Input("data: bool")
	.Output("output:bool")
	.Output("indices:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            auto input_shape0 = c->input(0);
            const auto dims = c->Rank(input_shape0);
            int nr;

            c->GetAttr("nr",&nr);

            shape_inference::ShapeHandle tmp_shape0;
            shape_inference::ShapeHandle tmp_shape1 = c->MakeShape({nr});
            shape_inference::ShapeHandle output_shape1;

            c->Subshape(input_shape0,0,-1,&tmp_shape0);
            c->Concatenate(tmp_shape0,tmp_shape1,&output_shape1);
			c->set_output(0, input_shape0);
			c->set_output(1, output_shape1);
			return Status::OK();
			});

template <typename Device>
class RandomSelectOp: public OpKernel {
	public:
		explicit RandomSelectOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("nr", &nr_));
			OP_REQUIRES_OK(context, context->GetAttr("sort_indices", &sort_indices_));
		}

		void Compute(OpKernelContext* context) override
		{
            const Tensor &_tensor        = context->input(0);
            auto          tensor         = _tensor.template flat<bool>().data();
            auto          dim_nr         = _tensor.dims();
            const auto    block_size     = _tensor.dim_size(dim_nr-1);
            const auto    total_nr       = _tensor.NumElements()/block_size;


            Tensor* output_data = NULL;
            Tensor* output_indices = NULL;
            TensorShape output_shape1 = _tensor.shape();

            output_shape1.set_dim(dim_nr-1,nr_);

			OP_REQUIRES(context, _tensor.dims() >= 1, errors::InvalidArgument("data must be at lest 1-dimensional"));
            OP_REQUIRES_OK(context, context->allocate_output(0, _tensor.shape(), &output_data));
            OP_REQUIRES_OK(context, context->allocate_output(1, output_shape1, &output_indices));


            auto       oq_tensor    = output_data->template flat<bool>().data();
            auto       oi_tensor    = output_indices->template flat<int>().data();
            const auto kMaxThreadNr = 100;
            std::vector<std::future<void>> res;
            const auto kDataNrPerThread = 20000;
            const auto kBatchSizePerThread = std::max<int>(1,kDataNrPerThread/block_size);

            output_indices->template flat<int>().setZero();
            copy(tensor,tensor+_tensor.NumElements(),oq_tensor);

            for(auto i=0; i<total_nr; i+=kBatchSizePerThread) {
                res.emplace_back(std::move(std::async(std::launch::async,
                                process_one_batch,oq_tensor+i*block_size,oi_tensor+i*nr_,
                                std::min<int>(kBatchSizePerThread,total_nr-i),
                                block_size,nr_,sort_indices_
                                )));
                if(res.size()>kMaxThreadNr)
                    res.clear();
            }
            res.clear();
		}
        static void process_one_batch(bool* data,int* o_indices,int batch_size,int size,int nr,bool sort_indices){
            for(auto i=0; i<batch_size; ++i) {
                 process_one_block(data+i*size,o_indices+i*nr,size,nr,sort_indices);
            }
        }
        static void process_one_block(bool* data,int* o_indices,int size,int nr,bool sort_indices){
            vector<int> indices;
            indices.reserve(nr*2);
            for(auto i=0; i<size; ++i){
                if(data[i])
                    indices.push_back(i);
            }
            if(indices.size()>=nr) {
                std::random_shuffle(indices.begin(),indices.end());
                for(auto i=nr; i<indices.size(); ++i) {
                    data[indices[i]] = false;
                }
            } 
            nr = std::min<int>(nr,indices.size());

            if(sort_indices)
                std::sort(indices.begin(),std::next(indices.begin(),nr));

            for(auto i=0; i<nr; ++i) {
                o_indices[i] = indices[i];
            }
        }
	private:
        int  nr_           = 1;
        bool sort_indices_ = false;

};
REGISTER_KERNEL_BUILDER(Name("RandomSelect").Device(DEVICE_CPU), RandomSelectOp<CPUDevice>);
