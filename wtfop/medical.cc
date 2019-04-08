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
#include <boost/algorithm/string/split.hpp>

using namespace tensorflow;
using namespace std;
namespace ba=boost::algorithm;
typedef Eigen::ThreadPoolDevice CPUDevice;
/*
 * 
 * bbox:[Nr,4](y,x,h,w)
 * labels:[Nr]
 * dlabels:[Nr]
 * type:[1]
 */
REGISTER_OP("MergeCharacter")
    .Attr("T: {float, double,int32}")
	.Attr("expand:float")
	.Attr("super_box_type:int")
    .Input("bboxes: T")
    .Input("labels: int32")
    .Input("dlabels: int32")
	.Output("text:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			c->set_output(0, c->Vector(-1));
			return Status::OK();
			});

template <typename Device, typename T>
class MergeCharacterOp: public OpKernel {
    public:
            using bbox_t = Eigen::Tensor<T,1,Eigen::RowMajor>;
            using bbox_info_t = tuple<bbox_t,int,int>;
	public:
		explicit MergeCharacterOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("expand", &expand_));
			OP_REQUIRES_OK(context, context->GetAttr("super_box_type", &super_box_type_));
		}

		void Compute(OpKernelContext* context) override
        {
            const Tensor &_bboxes  = context->input(0);
            const Tensor &_labels  = context->input(1);
            const Tensor &_dlabels = context->input(2);
            auto          bboxes   = _bboxes.template tensor<T,2>();
            auto          labels   = _labels.template tensor<int,1>();
            auto          dlabels  = _dlabels.template tensor<int,1>();

            OP_REQUIRES(context, _bboxes.dims() == 2, errors::InvalidArgument("box data must be 2-dimensional"));
            OP_REQUIRES(context, _labels.dims() == 1, errors::InvalidArgument("labels data must be 1-dimensional"));
            OP_REQUIRES(context, _dlabels.dims() == 1, errors::InvalidArgument("dlabels data must be 1-dimensional"));

            const auto     data_nr              = _bboxes.dim_size(0);
            vector<bbox_t> super_boxes;
            list<vector<int>> res_texts;
            for(auto i=0; i<data_nr; ++i) {
                if(labels(i) != super_box_type_)continue;
                super_boxes.emplace_back(expand_bbox(bboxes.chip(i,0)));
            }
            super_boxes = merge_super_bboxes(super_boxes);
            finetune_super_boxes(super_boxes,bboxes,labels);
            auto         is_h        = is_horizontal_text(super_boxes,bboxes);
            if(is_h) {
                sort(super_boxes.begin(),super_boxes.end(),[](const bbox_t& lhv,const bbox_t& rhv) { return lhv[0]<rhv[0];});
            } else {
                sort(super_boxes.begin(),super_boxes.end(),[](const bbox_t& lhv,const bbox_t& rhv) { return lhv[1]<rhv[1];});
            }


            if(data_nr-super_boxes.size() == 0) {
                make_default_return(context);
                return;
            }

            vector<float> bbox_iou;
            vector<int>    bboxes_type(size_t(data_nr),-1);

            bbox_iou.reserve(super_boxes.size());

            for(auto i=0; i<data_nr; ++i) {
                if(labels(i) == super_box_type_) continue;

                const bbox_t cur_bbox = bboxes.chip(i,0);

                bbox_iou.clear();
                for(auto sbbox:super_boxes) {
                    if(is_horizontal(sbbox)==is_h)
                        bbox_iou.emplace_back(iou(sbbox,cur_bbox,is_h));
                    else
                        bbox_iou.emplace_back(iou(sbbox,cur_bbox));
                }
                if(bbox_iou.empty()) continue;

                auto it = max_element(bbox_iou.begin(),bbox_iou.end());

                if(*it < 0.1)
                    bboxes_type[i] = -1;
                else
                    bboxes_type[i] = int(distance(bbox_iou.begin(),it));
            }

            for(auto i=0; i<super_boxes.size(); ++i) {
                const auto sbbox = super_boxes.at(i);
                vector<bbox_info_t> cur_bboxes;
                for(auto j=0; j<data_nr; ++j) {
                    if((bboxes_type[j] == i) && (labels(j) != super_box_type_)) {
                        cur_bboxes.emplace_back(bboxes.chip(j,0),labels(j),dlabels(j));
                    }
                }
                if(cur_bboxes.empty())continue;
                sort_text_info(cur_bboxes);
                auto ids = get_ids(cur_bboxes);
                res_texts.push_back(ids);
            }
            /*
               process type of -1
             */
            for(auto i=0; i<bboxes_type.size(); ++i) {
                if((bboxes_type[i] != -1) || (labels(i)==super_box_type_)) continue;
                vector<bbox_info_t> cur_bboxes;
                cur_bboxes.emplace_back(bboxes.chip(i,0),labels(i),dlabels(i));
                for(auto j=i+1; j<bboxes_type.size(); ++j) {
                    if((bboxes_type[j] != -1) || (labels(j)==super_box_type_)) continue;
                    auto scores = iou(bboxes.chip(i,0),bboxes.chip(j,0),is_h);
                    if(scores>0.3) {
                        bboxes_type[j] = 0;
                        cur_bboxes.emplace_back(bboxes.chip(j,0),labels(j),dlabels(j));
                    }
                }
                if(cur_bboxes.size()<1)continue;
                sort_text_info(cur_bboxes);
                auto ids = get_ids(cur_bboxes);
                res_texts.push_back(ids);
            }
            make_return(context,res_texts);
        }
        void make_return(OpKernelContext* context,const list<vector<int>>& texts) 
        {
            vector<int> datas;
            for(auto& t:texts) {
                datas.insert(datas.end(),t.begin(),t.end());
                datas.push_back(0);
            }

            TensorShape  outshape0;
            Tensor      *output_text = nullptr;
            int          dims_1d0[1]  = {datas.size()};
            TensorShapeUtils::MakeShape(dims_1d0, 1, &outshape0);
            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_text));

            auto text_data = output_text->flat<int>().data();

            for(auto i=0; i<datas.size(); ++i)
                text_data[i] = datas[i];
        }
        void make_default_return(OpKernelContext* context)
        {
            TensorShape  outshape;
            Tensor      *output_text = nullptr;
            int          dims_1d[1]  = {1};
            TensorShapeUtils::MakeShape(dims_1d, 1, &outshape);

            OP_REQUIRES_OK(context, context->allocate_output(0, outshape, &output_text));

            auto text_data = output_text->flat<int>().data();

            text_data[0] = 0;
        }
        void show_superbboxes(const vector<bbox_t>& bboxes,float h=256.0f,float w=256.0f) {
            cout<<"{";
            for(auto& box:bboxes) {
                cout<<bbox_to_str(box)<<endl;
            }
            cout<<"}"<<endl;
        }
        string bbox_to_str(const bbox_t& box) {
                stringstream ss;
                ss<<"("<<box(0)<<","<<box(1)<<","<<box(2)<<","<<box(3)<<")";
                return ss.str();
        }
        template<typename T0,typename T1>
        void finetune_super_boxes(vector<bbox_t>& super_bboxes,const T0& bboxes,const T1& labels)
        {
            vector<bbox_info_t> super_boxes_info;
            super_boxes_info.reserve(super_bboxes.size());

            for(auto& sbox:super_bboxes) {
                auto total_nr = 0;
                for(auto i=0; i<bboxes.dimension(0); ++i) {
                    if(labels(i) == super_box_type_) continue;
                    if(bboxes_jaccard_of_box0v1((bbox_t)bboxes.chip(i,0),sbox)>0.3) 
                        ++total_nr;
                }
                super_boxes_info.emplace_back(sbox,total_nr,0);
            }
            sort(super_boxes_info.begin(),super_boxes_info.end(),[](const auto& lhv,const auto& rhv) { return get<1>(lhv)>get<1>(rhv);});
            for(auto i=0; i<super_boxes_info.size(); ++i) {
                auto& cur_box = get<0>(super_boxes_info[i]);
                for(auto j=i+1; j<super_boxes_info.size(); ++j) {
                    clip_box_by(cur_box,get<0>(super_boxes_info[j]));
                }
            }
            super_bboxes.clear();
            for(auto& info:super_boxes_info) {
                if(box_sizev1(get<0>(info))<1E-4) continue;
                super_bboxes.push_back(get<0>(info));
            }
        }
        void clip_box_by(const bbox_t& ref_bbox,bbox_t& bbox) {
            if(bboxes_jaccardv1(ref_bbox,bbox)<1e-2) return;
            if(bboxes_jaccard_of_box0v1(bbox,ref_bbox)>=0.99) {
                bbox(0)=bbox(1)=bbox(2)=bbox(3) = -1.0;
                return;
            }
            if(bbox(1)<ref_bbox(1)) {
                bbox(3) = std::min(bbox(3),ref_bbox(1));
            } else if(bbox(3)>ref_bbox(3)) {
                bbox(1) = std::max(bbox(1),ref_bbox(3));
            } else if(bbox(0)<ref_bbox(0)) {
                bbox(2) = std::min(bbox(2),ref_bbox(0));
            } else if(bbox(2)>ref_bbox(2)) {
                bbox(0) = std::max(bbox(0),ref_bbox(2));
            }
        }
        float iou(const bbox_t& lhv,const bbox_t& rhv,bool is_h) {
            float union_v = 0.0;
            float int_v = 0.0;
            if(is_h) {
                union_v = min(rhv(2)-rhv(0),lhv(2)-lhv(0));
                int_v = min(rhv(2),lhv(2))-max(rhv(0),lhv(0));
            } else {
                union_v = min(rhv(3)-rhv(1),lhv(3)-lhv(1));
                int_v = min(rhv(3),lhv(3))-max(rhv(1),lhv(1));
            }
            if((int_v<0) || (union_v<0))
                return 0.0f;
            return int_v/union_v;
        }
        float iou(const bbox_t& super_box,const bbox_t& bbox) {
            return bboxes_jaccard_of_box0v1(bbox,super_box);
        }
        template<typename TT>
            inline bool is_horizontal_text(const vector<bbox_t>& boxes,const TT& bboxes) {
                if(!boxes.empty()) {
                    vector<int> is_hors(boxes.size());
                    transform(boxes.begin(),boxes.end(),is_hors.begin(),is_horizontal);
                    return accumulate(is_hors.begin(),is_hors.end(),0)>(boxes.size()/2);
                } else {
                    int h_words_nr = 0;
                    int v_words_nr = 0;
                    const auto data_nr = bboxes.dimension(0);
                    vector<bool> h_mask(data_nr,false);
                    vector<bool> v_mask(data_nr,false);
                    for(auto i=0; i<data_nr; ++i) {
                        if(!h_mask[i]) {
                            h_words_nr += 1;
                            for(auto j=i+1; j<data_nr; ++j) {
                                if((!h_mask[j]) && (iou(bboxes.chip(i,0),bboxes.chip(j,0),true)>0.5)) {
                                       h_mask[j] = true;
                                }
                            }
                        }
                        if(!v_mask[i]) {
                            v_words_nr += 1;
                            for(auto j=i+1; j<data_nr; ++j) {
                                if((!v_mask[j]) && (iou(bboxes.chip(i,0),bboxes.chip(j,0),true)>0.5)) {
                                    v_mask[j] = true;
                                }
                            }
                        }
                    }
                    cout<<__func__<<":"<<(h_words_nr<v_words_nr);
                    return h_words_nr<v_words_nr;
                }
            }
        static inline bool is_horizontal(const bbox_t& box) {
            return (box(3)-box(1))>(box(2)-box(0));
        }
        bbox_t merge_bbox(const bbox_t& lhv,const bbox_t& rhv) {
            auto ymin = std::min(lhv(0),rhv(0));
            auto xmin = std::min(lhv(1),rhv(1));
            auto ymax = std::max(lhv(2),rhv(2));
            auto xmax = std::max(lhv(3),rhv(3));
            float data[] = {ymin,xmin,ymax,xmax};
            cout<<"Merge:"<<bbox_to_str(lhv)<<","<<bbox_to_str(rhv)<<endl;
            return Eigen::TensorMap<bbox_t>(data,4);
        }
        vector<bbox_t> merge_super_bboxes(const vector<bbox_t>& bboxes) {
            return _merge_super_bboxes(_merge_super_bboxes(bboxes));
        }
        void sort_text_info(vector<bbox_info_t>& text_info) {
            array<int,4> counts = {0,0,0,0};
            for(auto& info:text_info)
                ++counts[get<2>(info)];
            cout<<"Count:{";
            for(auto x:counts)
                cout<<x<<",";
            cout<<"}"<<endl;
            const auto type = distance(counts.begin(),max_element(counts.begin(),counts.end()));
            switch(type) {
                case 0:
                    sort(text_info.begin(),text_info.end(),[](const bbox_info_t& lhv,const bbox_info_t& rhv) {
                            return get<0>(lhv)(1)<get<0>(rhv)(1);
                            });
                    break;
                case 1:
                    sort(text_info.begin(),text_info.end(),[](const bbox_info_t& lhv,const bbox_info_t& rhv) {
                            return get<0>(lhv)(0)<get<0>(rhv)(0);
                            });
                    break;
                case 2:
                    sort(text_info.begin(),text_info.end(),[](const bbox_info_t& lhv,const bbox_info_t& rhv) {
                            return get<0>(lhv)(1)>get<0>(rhv)(1);
                            });
                    break;
                case 3:
                    sort(text_info.begin(),text_info.end(),[](const bbox_info_t& lhv,const bbox_info_t& rhv) {
                            return get<0>(lhv)(0)>get<0>(rhv)(0);
                            });
                    break;
            }
            text_info = insert_space(text_info,type);
        }
        vector<bbox_info_t> insert_space(const vector<bbox_info_t>& text_info,int type) {
            if(text_info.empty()) return text_info;
            vector<bbox_info_t> res;
            switch(type) {
                case 0:
                    {
                        vector<float> values;
                        transform(text_info.begin(),text_info.end(),back_inserter(values),[](const bbox_info_t& v) {
                                return get<0>(v)[3]-get<0>(v)[1];
                                });
                        auto avg_width  = accumulate(values.begin(),values.end(),0.0f)/values.size();
                        res.push_back(text_info.front());
                        for(auto it=next(text_info.begin()); it!=text_info.end(); ++it) {
                            auto jt = prev(it);
                            const auto & b0 = get<0>(*it);
                            const auto & b1 = get<0>(*jt);
                            const auto count = int((b0(1)-b1(3))/avg_width+0.3f);
                            if(count>0) {
                                const auto dv = (b0(1)-b1(3))/count;
                                for(auto i=0; i<count; ++i) {
                                    T data[4] = {b1(0),b1(3)+dv*i,b1(2),b1(3)+dv*(i+1)};
                                    res.emplace_back(Eigen::TensorMap<bbox_t>(data,4),69,type);
                                }
                            }
                            res.push_back(*it);
                        }
                    }
                    break;
                case 2:
                    {
                        vector<float> values;
                        transform(text_info.begin(),text_info.end(),back_inserter(values),[](const bbox_info_t& v) {
                                return get<0>(v)[3]-get<0>(v)[1];
                                });
                        auto avg_width  = accumulate(values.begin(),values.end(),0.0f)/values.size();
                        res.push_back(text_info.front());
                        for(auto it=next(text_info.begin()); it!=text_info.end(); ++it) {
                            auto jt = prev(it);
                            const auto & b0 = get<0>(*it);
                            const auto & b1 = get<0>(*jt);
                            const auto count = int((b1(1)-b0(3))/avg_width+0.3f);
                            if(count>0) {
                                const auto dv = (b1(1)-b0(3))/count;
                                for(auto i=0; i<count; ++i) {
                                    T data[4] = {b0(0),b0(3)+dv*i,b0(2),b0(3)+dv*(i+1)};
                                    res.emplace_back(Eigen::TensorMap<bbox_t>(data,4),69,type);
                                }
                            }
                            res.push_back(*it);
                        }
                    }
                    break;
                 case 1:
                    {
                        vector<float> values;
                        transform(text_info.begin(),text_info.end(),back_inserter(values),[](const bbox_info_t& v) {
                                return get<0>(v)[2]-get<0>(v)[0];
                                });
                        auto avg_height = accumulate(values.begin(),values.end(),0.0f)/values.size();
                        res.push_back(text_info.front());
                        for(auto it=next(text_info.begin()); it!=text_info.end(); ++it) {
                            auto jt = prev(it);
                            const auto & b0 = get<0>(*it);
                            const auto & b1 = get<0>(*jt);
                            const auto count = int((b0(0)-b1(2))/avg_height+0.3f);
                            if(count>0) {
                                const auto dv = (b0(0)-b1(2))/count;
                                for(auto i=0; i<count; ++i) {
                                    T data[4] = {b1[2]+dv*i,b1[1],b1[2]+dv*(i+1),b1[3]};
                                    res.emplace_back(Eigen::TensorMap<bbox_t>(data,4),69,type);
                                }
                            }
                            res.push_back(*it);
                        }
                    }
                    break;
                 case 3:
                    {
                        vector<float> values;
                        transform(text_info.begin(),text_info.end(),back_inserter(values),[](const bbox_info_t& v) {
                                return get<0>(v)[2]-get<0>(v)[0];
                                });
                        auto avg_height = accumulate(values.begin(),values.end(),0.0f)/values.size();
                        res.push_back(text_info.front());
                        for(auto it=next(text_info.begin()); it!=text_info.end(); ++it) {
                            auto jt = prev(it);
                            const auto & b0 = get<0>(*it);
                            const auto & b1 = get<0>(*jt);
                            const auto count = int((b1(0)-b0(2))/avg_height+0.3f);
                            if(count>0) {
                                const auto dv = (b1(0)-b0(2))/count;
                                for(auto i=0; i<count; ++i) {
                                    T data[4] = {b0[2]+dv*i,b0[1],b0[2]+dv*(i+1),b0[3]};
                                    res.emplace_back(Eigen::TensorMap<bbox_t>(data,4),69,type);
                                }
                            }
                            res.push_back(*it);
                        }
                    }
                    break;
            }
            return res;
        }

        vector<bbox_t> _merge_super_bboxes(const vector<bbox_t>& bboxes) {
            if(bboxes.size() == 1) 
                return bboxes;
            auto pend = prev(bboxes.end());
            vector<bbox_t> res;
            vector<bool> mask(bboxes.size(),true);
            for(auto i=0; i<bboxes.size(); ++i) {
                if(mask[i] == false) continue;
                auto box = bboxes[i];
                auto is_h = is_horizontal(box);
                for(auto j=i+1; j<bboxes.size(); ++j) {
                    auto rbox = bboxes[j];
                    auto ris_h = is_horizontal(rbox);
                    if((is_h == ris_h) && (bboxes_jaccardv2(box,rbox,is_h)>0.143)) {
                        box = merge_bbox(box,rbox);
                        mask[j] = false;
                    }
                }
                res.push_back(box);
            }
            return res;
        }
        inline vector<int> get_ids(const vector<bbox_info_t>& box_info) {
            vector<int> ids(box_info.size());
            transform(box_info.begin(),box_info.end(),ids.begin(),[](const bbox_info_t& v){ return get<1>(v);});
            return ids;
        }

        bbox_t expand_bbox(const bbox_t& v) {
            auto h = v(2)-v(0);
            auto w = v(3)-v(1);
            const auto delta = expand_/2.0f;
            if(w>h) {
                float data[] = {v(0),v(1)-delta,v(2),v(3)+delta};
                return Eigen::TensorMap<bbox_t>(data,4);
            } else {
                float data[] = {v(0)-delta,v(1),v(2)+delta,v(3)};
                return Eigen::TensorMap<bbox_t>(data,4);
            }
        }
	private:
		float expand_         = 0.;
		int   super_box_type_ = 0;
};
REGISTER_KERNEL_BUILDER(Name("MergeCharacter").Device(DEVICE_CPU).TypeConstraint<float>("T"), MergeCharacterOp<CPUDevice, float>);
/*
 * id从1开始编号
 * targets:[Y]目标类型的集合，使用0为分隔符
 * texts:[X]待匹配的字符串集合，以0为分隔符
 * output:type:[1] 匹配的类型，或为-1表示无法匹配 
 */
REGISTER_OP("MachWords")
    .Attr("T: {int32,float, double,int32}")
	.Input("targets:T")
    .Input("texts: T")
	.Output("type:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			c->set_output(0, c->Vector(1));
			return Status::OK();
			});

template <typename Device, typename T>
class MachWordsOp: public OpKernel {
    public:
        using bbox_t = Eigen::Tensor<T,1,Eigen::RowMajor>;
        using bbox_info_t = pair<bbox_t,int>;
    public:
        explicit MachWordsOp(OpKernelConstruction* context) : OpKernel(context) {
            const string        trans_data       = "cijklopsuvwxz";
            transform(trans_data.begin(),trans_data.end(),back_inserter(trans_indexs_),[](char c) {
                return int(c)-int('a');
            });
        }
        template<typename TI,typename TO>
            static void split(TI& lhv,const TO& rhv) {
                vector<T> datas;
                datas.reserve(rhv.dimension(0));
                for(auto i=0; i<rhv.dimension(0); ++i)
                    datas.push_back(rhv(i));
                ba::split(lhv,datas,[](auto v) { return v==0; },ba::token_compress_on);
                auto it = remove_if(lhv.begin(),lhv.end(),[](auto v){return v.empty();});
                lhv.erase(it,lhv.end());
            }

        void Compute(OpKernelContext* context) override
        {
            const Tensor &_targets= context->input(0);
            const Tensor &_texts= context->input(1);
            auto          targets= _targets.template flat<T>();
            auto          texts= _texts.template flat<T>();
            vector<vector<int>> rtexts;

            OP_REQUIRES(context, _targets.dims() == 1, errors::InvalidArgument("targets data must be 1-dimensional"));
            OP_REQUIRES(context, _texts.dims() == 1, errors::InvalidArgument("texts data must be 1-dimensional"));
            split(target_type_data_,targets);
            for(auto& v:target_type_data_)
                tolower(v);
            split(rtexts,texts);

            auto min_score = 1;
            pair<int,int> type_info={-1,-1};
            for(auto& text:rtexts) {
                auto tmp_type_info = get_type(text);
                if(tmp_type_info.second>type_info.second)
                    type_info = tmp_type_info;
                else if(tmp_type_info.second == type_info.second)
                    min_score = type_info.second;
            }
            if(type_info.second>min_score) {
                make_return(context,type_info.first);
            } else {
                make_return(context,-1);
            }
        }
        void make_return(OpKernelContext* context,int type) 
        {
            TensorShape  outshape0;
            Tensor      *output_type = nullptr;
            int          dims_1d0[1]  = {1};
            TensorShapeUtils::MakeShape(dims_1d0, 1, &outshape0);
            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_type));

            auto type_data = output_type->flat<int>().data();

            type_data[0] = type;
        }
        pair<int,int> get_type(vector<int>& ids) {
            tolower(ids);
            pair<int,float> res{-1,-1.0};
            for(auto i=0; i<target_type_data_.size(); ++i) {
                auto& type_data = target_type_data_[i];
                auto score = match(ids,type_data);
                if(score>res.second) {
                    res = make_pair(i,score);
                }
            }
            return res;
        }

        float match(const vector<int>& source,const vector<int>& target) {
            auto d0 = split_ids(source,target.size());
            const auto max_score = 100;
            for(const auto& d:d0) {
                if(equal(d.begin(),d.end(),target.begin()))
                    return max_score+target.size();
            }
            if(target.size()<=2)
                return 0;
            if(d0.empty())
                d0.push_back(source);
            auto d1 = split_ids(source,target.size()+1);
            copy(d1.begin(),d1.end(),back_inserter(d0));
            float res_score = 0;
            for(auto& d:d0) {
                const auto cost = edit_distance(d.begin(),d.end(),target.begin(),target.end());
                if((cost>target.size()-2) || (cost>1))
                    continue;
                const auto score = max_score - cost;
                if(score>res_score)
                    res_score = score;


            }
            return res_score;
        }

        list<vector<int>> split_ids(const vector<int>& v,size_t size) {
            list<vector<int>> res;
            if(v.size()<size) {
                return res;
            }
            auto end=prev(v.end(),size-1);
            for(auto it = v.begin(); it!=end; ++it) {
                res.push_back(vector<int>(it,next(it,size)));
            }
            return res;
        }

        bbox_t expand_bbox(const bbox_t& v) {
            auto h = v(2)-v(0);
            auto w = v(3)-v(1);
            const auto delta = expand_/2.0f;
            if(w>h) {
                float data[] = {v(0),v(1)-delta,v(2),v(3)+delta};
                return Eigen::TensorMap<bbox_t>(data,4);
            } else {
                float data[] = {v(0)-delta,v(1),v(2)+delta,v(3)};
                return Eigen::TensorMap<bbox_t>(data,4);
            }
        }
        string ids_to_string(const vector<int>& ids) {
            string res;
            for(auto id:ids) {
                res.push_back(raw_text_data_[id-1]);
            }
            return res;
        }
        string get_raw_text_data() {
            string text_array;

            for(auto i=int('a'); i<=int('z'); ++i) {
                text_array.push_back(char(i));
            }
            for(auto i=int('A'); i<=int('Z'); ++i) {
                text_array.push_back(char(i));
            }
            for(auto i=int('0'); i<=int('9'); ++i) {
                text_array.push_back(char(i));
            }
            text_array.push_back('/');
            text_array.push_back('\\');
            text_array.push_back('-');
            text_array.push_back('+');
            text_array.push_back(':');

            return text_array;
        }
        void tolower(vector<int>& v) {
            auto need_trans = [this](int v) { return find(trans_indexs_.begin(),trans_indexs_.end(),v-27)!=trans_indexs_.end();};
            transform(v.begin(),v.end(),v.begin(),[&need_trans](int v) { 
                    if((v>=27)&&(v<53)&&need_trans(v))
                        return v-26; 
                    return v;
                    });
        }
        template<typename IT0,typename IT1>
            int edit_distance(IT0 fbegin,IT0 fend,IT1 sbegin, IT1 send) {
                auto fnr = distance(fbegin,fend);
                auto snr = distance(sbegin,send);
                const  auto cd = 1;
                const  auto ci = 1;
                const  auto cr = 1;
                if(min(fnr,snr)==0) {
                    return max(fnr,snr);
                }
                auto c0 = edit_distance(fbegin,prev(fend),sbegin,send)+cd;
                auto c1 = edit_distance(fbegin,fend,sbegin,prev(send))+ci;
                auto c2 = edit_distance(fbegin,prev(fend),sbegin,prev(send))+((*prev(fend)==*prev(send))?0:cr);
                return min({c0,c1,c2});
            }
    private:
        float               expand_           = 0.;
        int                 super_box_type_   = 0;
        vector<vector<int>> target_type_data_;
        string              raw_text_data_;
        vector<int>         trans_indexs_;
};
REGISTER_KERNEL_BUILDER(Name("MachWords").Device(DEVICE_CPU).TypeConstraint<int>("T"), MachWordsOp<CPUDevice, int>);
