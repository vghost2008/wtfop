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
 * bbox:[Nr,4](y,x,h,w)
 * labels:[Nr]
 * type:[1]
 */
REGISTER_OP("LabelType")
    .Attr("T: {float, double,int32}")
	.Attr("expand:float")
	.Attr("super_box_type:int")
    .Input("bboxes: T")
    .Input("labels: int32")
	.Output("type:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			c->set_output(0, c->Vector(1));
			return Status::OK();
			});

template <typename Device, typename T>
class LabelTypeOp: public OpKernel {
    public:
            using bbox_t = Eigen::Tensor<T,1,Eigen::RowMajor>;
            using bbox_info_t = pair<bbox_t,int>;
	public:
		explicit LabelTypeOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("expand", &expand_));
			OP_REQUIRES_OK(context, context->GetAttr("super_box_type", &super_box_type_));
            raw_text_data_ = get_raw_text_data();
            string target_type_string[] = {"Ki-67","ER","Her-2","PR","HP"};
            for(auto& str:target_type_string) {
                target_type_data_.emplace_back(string_to_ids(str));
                auto& data = target_type_data_.back();
                tolower(data);
            }
            assert(super_box_type_>=raw_text_data_.size());
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &_bboxes = context->input(0);
			const Tensor &_labels = context->input(1);
			auto          bboxes  = _bboxes.template tensor<T,2>();
			auto          labels  = _labels.template tensor<int,1>();

			OP_REQUIRES(context, _bboxes.dims() == 2, errors::InvalidArgument("box data must be 2-dimensional"));
			OP_REQUIRES(context, _labels.dims() == 1, errors::InvalidArgument("labels data must be 1-dimensional"));

			const auto     data_nr              = _bboxes.dim_size(0);
			vector<bbox_t> super_boxes;
            for(auto i=0; i<data_nr; ++i) {
                if(labels(i) != super_box_type_)continue;
                super_boxes.emplace_back(expand_bbox(bboxes.chip(i,0)));
            }
            show_superbboxes(super_boxes);
            super_boxes = merge_super_bboxes(super_boxes);
            show_superbboxes(super_boxes);
            auto         is_h        = is_horizontal_text(super_boxes);
            TensorShape  outshape;
            Tensor      *output_type = nullptr;
            int          dims_1d[1]  = {1};
			TensorShapeUtils::MakeShape(dims_1d, 1, &outshape);
            if(is_h) {
                cout<<"H:"<<endl;
                sort(super_boxes.begin(),super_boxes.end(),[](const bbox_t& lhv,const bbox_t& rhv) { return lhv[0]<rhv[0];});
            } else {
                cout<<"V:"<<endl;
                sort(super_boxes.begin(),super_boxes.end(),[](const bbox_t& lhv,const bbox_t& rhv) { return lhv[1]<rhv[1];});
            }

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape, &output_type));

            auto type_data = output_type->flat<int>().data();

            if(super_boxes.size() == 0) {
                type_data[0] = -1;
                return;
            }
            vector<float> bbox_iou;
			vector<int>    bboxes_type;
            bbox_iou.reserve(super_boxes.size());
            bboxes_type.reserve(data_nr);
           

            for(auto i=0; i<data_nr; ++i) {
                bbox_iou.clear();
                const bbox_t cur_bbox = bboxes.chip(i,0);
                for(auto sbbox:super_boxes) {
                    //bbox_iou.emplace_back(bboxes_jaccardv1(sbbox,cur_bbox));
                    bbox_iou.emplace_back(iou(sbbox,cur_bbox,is_h));
                }
                bboxes_type.emplace_back(distance(bbox_iou.begin(),max_element(bbox_iou.begin(),bbox_iou.end())));
            }
            pair<int,float> type_info={-1,-1.0};
            for(auto i=0; i<super_boxes.size(); ++i) {
                const auto sbbox = super_boxes.at(i);
                vector<bbox_info_t> cur_bboxes;
                for(auto j=0; j<data_nr; ++j) {
                    if((bboxes_type[j] == i) && (labels(j) != super_box_type_)) {
                        cur_bboxes.emplace_back(bboxes.chip(j,0),labels(j));
                    }
                }
                if(cur_bboxes.empty())continue;
                if(is_h) {
                    sort(cur_bboxes.begin(),cur_bboxes.end(),[](const bbox_info_t& lhv,const bbox_info_t& rhv) {
                        return lhv.first(1)<rhv.first(1);
                    });
                } else {
                    sort(cur_bboxes.begin(),cur_bboxes.end(),[](const bbox_info_t& lhv,const bbox_info_t& rhv) {
                        return lhv.first(0)<rhv.first(0);
                    });
                }
                auto tmp_type_info = get_type(cur_bboxes);
                if(tmp_type_info.second>type_info.second)
                    type_info = tmp_type_info;
            }

            if(type_info.second>1e-5) {
                type_data[0] = type_info.first;
            } else {
                type_data[0] = -1;
            }
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
        float iou(const bbox_t& lhv,const bbox_t& rhv,bool is_h) {
            float union_v = 0.0;
            float int_v = 0.0;
            if(is_h) {
                union_v = max(rhv(2),lhv(2))-min(rhv(0),lhv(0));
                int_v = min(rhv(2),lhv(2))-max(rhv(0),lhv(0));
            } else {
                union_v = max(rhv(3),lhv(3))-min(rhv(1),lhv(1));
                int_v = min(rhv(3),lhv(3))-max(rhv(1),lhv(1));
            }
            if((int_v<0) || (union_v<0))
                return 0.0f;
            return int_v/union_v;
        }
        inline bool is_horizontal_text(const vector<bbox_t>& boxes) {
            vector<int> is_hors(boxes.size());
            transform(boxes.begin(),boxes.end(),is_hors.begin(),is_horizontal);
            return accumulate(is_hors.begin(),is_hors.end(),0)>(boxes.size()/2);

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
                    if((is_h == ris_h) && (bboxes_jaccardv1(box,rbox)>1e-8)) {
                        box = merge_bbox(box,rbox);
                        mask[j] = false;
                    }
                }
                res.push_back(box);
            }
            return res;
        }
        pair<int,float> get_type(const vector<bbox_info_t>& box_info) {
            vector<int> ids(box_info.size());
            transform(box_info.begin(),box_info.end(),ids.begin(),[](const bbox_info_t& v){ return v.second;});
            cout<<"Text:"<<ids_to_string(ids)<<endl;
            tolower(ids);
            pair<int,float> res{-1,-1.0};
            for(auto i=0; i<target_type_data_.size(); ++i) {
                auto score = match(ids,target_type_data_[i]);
                if(score>res.second) {
                    res = make_pair(i,score);
                }
            }
            return res;
        }

        float match(const vector<int>& source,const vector<int>& target) {
            auto d0 = split_ids(source,target.size());
            for(const auto& d:d0) {
                if(equal(d.begin(),d.end(),target.begin()))
                    return 100.0+target.size();
            }
            if(d0.empty())
                d0.push_back(source);
            float res_score = 0.0f;
            auto d11 = split_ids(target,2);
            float match_nr = 0.0f;
            for(auto& d:d0) {
                auto d10 = split_ids(d,2);
                float tmp_match_nr = 0.0f;
                for(auto sd10:d10) {
                    for(auto&d1:d11) {
                        if(equal(sd10.begin(),sd10.end(),d1.begin())) 
                            tmp_match_nr += 1.0f;
                    }
                }
                if(tmp_match_nr>match_nr)
                    match_nr = tmp_match_nr;
            }
            if(d11.size()>0) {
                res_score = match_nr*2.0/d11.size();
            }

            auto d21 = split_ids(target,1);
            match_nr = 0.0f;
            for(auto& d:d0) {
                auto d10 = split_ids(d,1);
                float tmp_match_nr = 0.0f;
                for(auto sd10:d10) {
                    for(auto&d1:d21) {
                        if(equal(sd10.begin(),sd10.end(),d1.begin())) 
                            tmp_match_nr += 1.0f;
                    }
                }
                if(tmp_match_nr>match_nr)
                    match_nr = tmp_match_nr;
            }
            if(d21.size()>0) {
                auto tmp_res_score = match_nr*2.0/d21.size();
                if(tmp_res_score>res_score)
                    res_score = tmp_res_score;
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
        vector<int> string_to_ids(const string& str) {
            vector<int> res;
            res.reserve(str.size());
            for(auto c:str) {
                auto pos = raw_text_data_.find(c);
                if(pos == string::npos) {
                    res.push_back(-1);
                } else {
                    res.push_back(pos+1);
                }
            }
            return res;
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
            transform(v.begin(),v.end(),v.begin(),[](int v) {
                if((v>=26)&&(v<52)) return v-26;
                return v;
            });
        }
	private:
		float expand_         = 0.;
		int   super_box_type_ = 0;
        vector<vector<int>> target_type_data_;
        string raw_text_data_;
};
REGISTER_KERNEL_BUILDER(Name("LabelType").Device(DEVICE_CPU).TypeConstraint<float>("T"), LabelTypeOp<CPUDevice, float>);
