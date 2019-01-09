_Pragma("once")
#include <math.h>
#include <algorithm>
#include <memory>
#include <tuple>
#include <vector>
#include <boost/algorithm/clamp.hpp>
/*
 * box:(ymin,xmin,ymax,xmax)
 */
template<typename T0,typename T1>
auto bboxes_jaccard(const T0& box0, const T1& box1)
{
	const auto int_ymin  = std::max(box0[0],box1[0]);
	const auto int_xmin  = std::max(box0[1],box1[1]);
	const auto int_ymax  = std::min(box0[2],box1[2]);
	const auto int_xmax  = std::min(box0[3],box1[3]);
	const auto int_h     = std::max<float>(int_ymax-int_ymin,0.);
	const auto int_w     = std::max<float>(int_xmax-int_xmin,0.);
	const auto int_vol   = int_h *int_w;
	const auto vol1      = (box0[2]-box0[0]) *(box0[3]-box0[1]);
	const auto vol2      = (box1[2]-box1[0]) *(box1[3]-box1[1]);
	const auto union_vol = vol1+vol2-int_vol;

	if(union_vol<1E-6) return 0.0f;

	return int_vol/union_vol;
}
template<typename T0,typename T1>
auto bboxes_jaccardv1(const T0& box0, const T1& box1)
{
	const auto int_ymin  = std::max(box0(0),box1(0));
	const auto int_xmin  = std::max(box0(1),box1(1));
	const auto int_ymax  = std::min(box0(2),box1(2));
	const auto int_xmax  = std::min(box0(3),box1(3));
	const auto int_h     = std::max<float>(int_ymax-int_ymin,0.);
	const auto int_w     = std::max<float>(int_xmax-int_xmin,0.);
	const auto int_vol   = int_h *int_w;
	const auto vol1      = (box0(2)-box0(0)) *(box0(3)-box0(1));
	const auto vol2      = (box1(2)-box1(0)) *(box1(3)-box1(1));
	const auto union_vol = vol1+vol2-int_vol;

	if(union_vol<1E-6) return 0.0f;

	return int_vol/union_vol;
}
/*
 * box:(ymin,xmin,ymax,xmax)
 * 仅计算两个交叉的box的交叉面积占box0的百分比
 */
template<typename T>
T bboxes_jaccard_of_box0(const T* box0, const T* box1)
{
	const auto int_ymin = std::max(box0[0],box1[0]);
	const auto int_xmin = std::max(box0[1],box1[1]);
	const auto int_ymax = std::min(box0[2],box1[2]);
	const auto int_xmax = std::min(box0[3],box1[3]);
	const auto int_h    = std::max<T>(int_ymax-int_ymin,0.);
	const auto int_w    = std::max<T>(int_xmax-int_xmin,0.);
	const auto int_vol  = int_h *int_w;
	const auto box0_vol = (box0[2]-box0[0]) *(box0[3]-box0[1]);

	if(box0_vol<1E-6) return 0.0f;

	return int_vol/box0_vol;
}
/*
 * box:ymin,xmin,ymax,xmax
 * return:cy,cx,h,w
 */
template<typename T>
auto box_minmax_to_cxywh(const T& box)
{
	return std::make_tuple((box[0]+box[2])/2.0,(box[1]+box[3])/2.0,box[2]-box[0],box[3]-box[1]);
}
/*
 * box:cy,cx,h,w
 * return:ymin,xmin,ymax,xmax
 */
template<typename T>
std::tuple<T,T,T,T> box_cxywh_to_minmax(const T* box)
{
	return std::make_tuple(box[0]-box[2]/2.0,box[1]-box[3]/2.0,box[0]+box[2]/2.,box[1]+box[3]/2);
}
template<typename T>
std::tuple<T,T,T,T> box_cxywh_to_minmax(T cy, T cx, T h, T w)
{
	return std::make_tuple(cy-h/2.0,cx-w/2.,cy+h/2.0,cx+w/2.0);
}
/*
 * box:ymin,xmin,ymax,xmax
 */
template<typename T>
T box_area(const T* box) {
	const auto h = std::max<T>(0,box[2]-box[0]);
	const auto w = std::max<T>(0,box[3]-box[1]);
	return h*w;
}
/*
 * ref_box,target_box为在原图中的box
 * output为将原图按ref_box剪切后，target_box在新图中应该的大小与位置
 * output会截断到[0,1]
 */
template<typename T>
bool cut_box(const T* ref_box,const T* target_box,T* output) {

	bzero(output,sizeof(T)*4);
	if((target_box[0]>=ref_box[2])
			|| (target_box[2]<=ref_box[0])
			|| (target_box[1]>=ref_box[3])
			|| (target_box[3]<=ref_box[1]))
		return false;

	const auto w = ref_box[3]-ref_box[1];
	const auto h = ref_box[2]-ref_box[0];

	if((w<1E-5) 
			|| (h<1E-5))
		return false;

	output[0] = (target_box[0]-ref_box[0])/h;
	output[2] = (target_box[2]-ref_box[0])/h;
	output[1] = (target_box[1]-ref_box[1])/w;
	output[3] = (target_box[3]-ref_box[1])/w;

	for(auto i=0; i<4; ++i) {
		output[i] = boost::algorithm::clamp<float>(output[i],0.0f,1.0f);
	}
	return true;
}

template<typename T>
void copy_box(const T* input,T* output) {
	for(int i=0; i<4; ++i)
		output[i] = input[i];
}
template<typename T>
void copy_boxes(const T* input,T* output,int nr) {
	for(int i=0; i<nr; ++i) {
		copy_box(input,output);
		input += 4;
		output += 4;
	}
}
template<typename T>
T box_size(const T* input) {
	return (input[2]-input[0])*(input[3]-input[1]);
}
template <typename T>
class BoxesEncodeUnit {
	public:
		struct IOUIndex{
			int index;
			float iou;
		};
		explicit BoxesEncodeUnit(float pos_threshold,float neg_threshold,const std::vector<float>& prio_scaling) 
			:pos_threshold_(pos_threshold)
			 ,neg_threshold_(neg_threshold)
			 ,prio_scaling_(prio_scaling){
				 assert(prio_scaling_.size() == 4);
			 }
		/*
		template<typename DT0,typename DT1,typename DT2>
			auto operator()(
					const DT0& boxes,
					const DT1& gboxes,
					const DT2& glabels,int data_nr,int gdata_nr)
		 */
		   auto operator()(
		   const Eigen::Tensor<T,2,Eigen::RowMajor>& boxes,
		   const Eigen::Tensor<T,2,Eigen::RowMajor>& gboxes,
		   const Eigen::Tensor<int,1,Eigen::RowMajor>& glabels)
			{
                int                   data_nr              = boxes.dimension(0);
                int                   gdata_nr             = gboxes.dimension(0);
                auto                  out_boxes            = Eigen::Tensor<T,2,Eigen::RowMajor>(data_nr,4);
                auto                  out_labels           = Eigen::Tensor<int,1,Eigen::RowMajor>(data_nr);
                auto                  out_scores           = Eigen::Tensor<T,1,Eigen::RowMajor>(data_nr);
                auto                  out_remove_indices   = Eigen::Tensor<bool,1,Eigen::RowMajor>(data_nr);
                std::vector<int>      outindex(data_nr       ,-1);
                std::vector<bool>     is_max_score(data_nr   ,false);
                std::vector<IOUIndex> iou_indexs(data_nr     ,IOUIndex({-1,0.0}));                          //默认box不与任何ground truth box相交，iou为0

				for(auto i=0; i<data_nr; ++i) {
					out_labels(i) = 0;  //默认所有的都为背景
					out_scores(i) = 0;
				}
				/*
				 * 遍历每一个ground truth box
				 */
				for(auto i=0; i<gdata_nr; ++i) {
					const      Eigen::Tensor<T,1,Eigen::RowMajor> gbox= gboxes.chip(i,0);
					const auto glabel          = glabels(i);
					auto       max_index       = -1;
					auto       max_scores      = -1.0;

					/*
					 * 计算ground truth box与每一个候选box的jaccard得分
					 */
					for(auto j=0; j<data_nr; ++j) {
						const Eigen::Tensor<T,1,Eigen::RowMajor> box       = boxes.chip(j,0);
						auto        jaccard   = bboxes_jaccardv1(gbox,box);
						auto       &iou_index = iou_indexs[j];

						if(jaccard<1E-8) continue;

						if(jaccard>max_scores) {
							max_scores = jaccard;
							max_index = j;
						}

						if(jaccard>iou_index.iou) {
							iou_index.iou = jaccard;
							iou_index.index = i;
						}
						if((jaccard < pos_threshold_) 
								|| (jaccard<out_scores(j)) 
								|| is_max_score[j]) //不覆盖特殊情况
							continue;
						out_scores(j) = jaccard;
						out_labels(j) = glabel;
						outindex[j] = i;
					}
					if(max_scores<1E-8) continue;
					/*
					 * 面积交叉最大的给于标签
					 */
					auto j = max_index;

					if((out_scores(j) <= max_scores) || (!is_max_score[j])) {
						out_scores(j) = max_scores;
						out_labels(j) = glabel;
						outindex[j] = i;
						is_max_score[j] = true;
					}
				} //end for

				for(auto j=0; j<data_nr; ++j) {
					const auto& iou_index = iou_indexs[j];
					if((iou_index.iou>=pos_threshold_) || (iou_index.iou<neg_threshold_) || is_max_score[j])
						out_remove_indices(j) = false;
					else
						out_remove_indices(j) = true;
				}

				/*
				 * 计算所有正样本proposal box到所对应的ground truth box的回归参数
				 */
				for(auto j=0; j<data_nr; ++j) {
					auto  outbox  = out_boxes.chip(j,0);

					if((out_labels(j)<1) || (out_remove_indices(j))) {
                        outbox.setZero();
						continue;
					}
					auto &feat_cy = out_boxes(j,0);
					auto &feat_cx = out_boxes(j,1);
					auto &feat_h  = out_boxes(j,2);
					auto &feat_w  = out_boxes(j,3);
					Eigen::Tensor<T,1,Eigen::RowMajor> box     = boxes.chip(j,0);
					Eigen::Tensor<T,1,Eigen::RowMajor> gbox    = gboxes.chip(outindex[j],0);
					auto  yxhw    = box_minmax_to_cxywh(box.data());
					auto  yref    = std::get<0>(yxhw);
					auto  xref    = std::get<1>(yxhw);
					auto  href    = std::get<2>(yxhw);
					auto  wref    = std::get<3>(yxhw);

					if((href<1E-8) || (wref<1E-8)) {
						feat_cy = feat_cx = feat_h = feat_w = 0.0;
						continue;
					}

					auto gyxhw = box_minmax_to_cxywh(gbox.data());

					feat_cy  =  std::get<0>(gyxhw);
					feat_cx  =  std::get<1>(gyxhw);
					feat_h   =  std::get<2>(gyxhw);
					feat_w   =  std::get<3>(gyxhw);

					feat_cy  =  (feat_cy-yref)/(href*prio_scaling_[0]);
					feat_cx  =  (feat_cx-xref)/(wref*prio_scaling_[1]);
					feat_h   =  log(feat_h/href)/prio_scaling_[2];
					feat_w   =  log(feat_w/wref)/prio_scaling_[3];
				}
				return std::make_tuple(out_boxes,out_labels,out_scores,out_remove_indices);
			}
	private:
		const float              pos_threshold_;
		const float              neg_threshold_;
		const std::vector<float> prio_scaling_;
};
