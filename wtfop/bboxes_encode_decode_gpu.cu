#include <vector>
#include "wtoolkit_cuda.h"
#include <algorithm>

#ifdef GOOGLE_CUDA
template<typename T>
__device__ std::tuple<T,T,T,T> cuda_box_minmax_to_cxywh(const T* box)
{
	return std::make_tuple((box[0]+box[2])/2.0,(box[1]+box[3])/2.0,box[2]-box[0],box[3]-box[1]);
}
template<typename T0,typename T1>
__device__ float cuda_bboxes_jaccard(const T0* box0, const T1* box1)
{
	const auto int_ymin  = std::max(box0[0],box1[0]);
	const auto int_xmin  = std::max(box0[1],box1[1]);
	const auto int_ymax  = std::min(box0[2],box1[2]);
	const auto int_xmax  = std::min(box0[3],box1[3]);
	const float int_h     = std::max<float>(int_ymax-int_ymin,0.);
	const float int_w     = std::max<float>(int_xmax-int_xmin,0.);
	const auto int_vol   = int_h *int_w;
	const auto vol1      = (box0[2]-box0[0]) *(box0[3]-box0[1]);
	const auto vol2      = (box1[2]-box1[0]) *(box1[3]-box1[1]);
	const auto union_vol = vol1+vol2-int_vol;

	if(union_vol<1E-6) return 0.0f;

	return int_vol/union_vol;
}
template<typename T>
__device__ inline bool cuda_is_cross_boundaries(const T* box) {
    return (box[0]<0.0) || (box[1]<0.0) || (box[2]>1.0) ||(box[3]>1.0);
}
template<typename T>
__device__ inline T& get_sm_cell(T* score_matrix,int g_index, int a_index,size_t ab_size)
{
    return *(score_matrix+g_index*ab_size+a_index);
}
__global__ void get_jaccards(const float* gbboxes,const float* anchor_bboxes,float* score_matrix,size_t gb_size,size_t ab_size)
{
    const int g_index = threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
    const auto a_index = blockIdx.x;

    if(g_index>=gb_size) 
        return;
    if(cuda_is_cross_boundaries(anchor_bboxes+a_index)) {
        get_sm_cell(score_matrix,g_index,a_index,ab_size) = -1.0;
    } else {
        get_sm_cell(score_matrix,g_index,a_index,ab_size) = cuda_bboxes_jaccard(gbboxes+g_index,anchor_bboxes+a_index);
    }
}
__global__ void find_max_score_index(const float* score_matrix,float* scores,int* indexs,size_t gb_size,size_t ab_size)
{
    const auto g_index = blockIdx.x;
    auto pbegin = score_matrix+g_index*ab_size;
    auto pend = pbegin+ab_size;
    auto it = std::max_element(pbegin,pend);
    if((*it)<1e-8) 
        return;
    auto max_index = it-pbegin;
    indexs[g_index] = max_index;
    scores[g_index] = *it;
}
__global__ void get_scores_and_indexs(const float* score_matrix,float* scores,int* indexs,size_t gb_size,size_t ab_size)
{
    const auto a_index = blockIdx.x;
    auto max_i = -1;
    auto max_s = 1e-8;
    for(auto i=0; i<gb_size; ++i) {
        const auto cs = get_sm_cell(score_matrix,i,a_index,ab_size);
        if(cs>max_s) {
            max_i = i;
            max_s = cs;
        }
    }
    indexs[a_index] = max_i;
    scores[a_index] = max_s;
}
__global__ void update_indexs_and_scores_by_max_score(int* indexs,int* indexs0,float* scores,float* scores0,bool* is_max_score)
{
    auto g_index = blockIdx.x;
    if(indexs0[g_index]>0) {
        auto a_index = indexs0[g_index];
        indexs[a_index] = g_index;
        scores[a_index] = scores0[g_index];
        is_max_score[a_index] = true;
    }
}
__global__ void get_labels_and_remove_indices(const int* indexs,const float* scores,const bool* is_max_score,const int* glabels,int* out_labels,bool* remove_indices,float neg_threshold,float pos_threshold)
{
    auto a_index = blockIdx.x;
    const auto& index = indexs[a_index];
    const auto score = scores[a_index];
    if((score>=pos_threshold) || (score<neg_threshold) || is_max_score[a_index]) {
        remove_indices[a_index] = false;
        if((score>=pos_threshold) || is_max_score[a_index])
            out_labels[a_index] = glabels[index];
        else
            out_labels[a_index] = 0;
    } else {
        remove_indices[a_index] = true;
    }
}
__global__ void get_bboxes_regression(float* out_boxes,const float* anchor_bboxes,const float* gbboxes,const int* out_labels,const bool* out_remove_indices,const int* out_index,float* prio_scaling)
{
    auto j = blockIdx.x; //a_index

    auto  outbox  = out_boxes+j*4;
    if((out_labels[j]<1) || (out_remove_indices[j])) {
        return;
    }
    auto box     = anchor_bboxes+j*4;
    auto gbox    = gbboxes+out_index[j]*4;
    auto  yxhw    = cuda_box_minmax_to_cxywh(box);
    auto  yref    = std::get<0>(yxhw);
    auto  xref    = std::get<1>(yxhw);
    auto  href    = std::get<2>(yxhw);
    auto  wref    = std::get<3>(yxhw);

    if((href<1E-8) || (wref<1E-8)) {
        return;
    }

    auto gyxhw = cuda_box_minmax_to_cxywh(gbox);

    auto feat_cy  =  std::get<0>(gyxhw);
    auto feat_cx  =  std::get<1>(gyxhw);
    auto feat_h   =  std::get<2>(gyxhw);
    auto feat_w   =  std::get<3>(gyxhw);

    outbox[0] =  (feat_cy-yref)/(href*prio_scaling[0]);
    outbox[1] =  (feat_cx-xref)/(wref*prio_scaling[1]);
    outbox[2] =  log(feat_h/href)/prio_scaling[2];
    outbox[3] =  log(feat_w/wref)/prio_scaling[3];
}
__host__ void get_encodes(const float* gbboxes,const float* anchor_bboxes,const int* glabels,
float* out_boxes,float* out_scores,int* out_labels,bool* out_remove_indices,int* out_index,const float* prio_scaling,
size_t gb_size,size_t ab_size,float neg_threshold,float pos_threshold)
{
    float* d_score_matrix = nullptr;

    CHECK_OK(cudaMalloc((float**)&d_score_matrix,sizeof(float)*ab_size*gb_size));
    CHECK_OK(cudaMemset(out_boxes,0,sizeof(float)*4*ab_size));
    CHECK_OK(cudaMemset(out_scores,0,sizeof(float)*ab_size));
    CHECK_OK(cudaMemset(out_index,0xff,sizeof(int)*ab_size));
    CHECK_OK(cudaMemset(out_labels,0,sizeof(int)*ab_size));

    int block_limits[] = {1024,1024};
    dim3 block;
    dim3 grid(ab_size);

    if(gb_size>block_limits[0]) {
        block.x = block_limits[0];
        if(gb_size>(block_limits[0]*block_limits[1])) {
            block.y = block_limits[1];
            const auto bs = block.x*block.y;
            block.z = (gb_size+bs-1)/bs;
        } else {
            const auto bs = block.x;
            block.y = (gb_size+bs-1)/bs;
        }
    }
    get_jaccards<<<grid,block>>>(gbboxes,anchor_bboxes,d_score_matrix,gb_size,ab_size);
    show_cuda_data(d_score_matrix,gb_size*ab_size,ab_size,"d_score_matrix");

    float* d_scores = nullptr;
    int* d_indexs = nullptr;
    float* d_scores0 = nullptr;
    int* d_indexs0 = nullptr;
    dim3 grid1(gb_size);
    cuda_unique_ptr<float> d_prio_scaling = make_cuda_unique<float>(prio_scaling,4);
    cuda_unique_ptr<bool> d_is_max_score = make_cuda_unique<bool>((unsigned char)(0x00),ab_size);

    CHECK_OK(cudaMalloc((float**)&d_scores,sizeof(float)*ab_size));
    CHECK_OK(cudaMalloc((int**)&d_indexs,sizeof(int)*ab_size));
    CHECK_OK(cudaMalloc((float**)&d_scores0,sizeof(float)*gb_size));
    CHECK_OK(cudaMalloc((int**)&d_indexs0,sizeof(int)*gb_size));

    get_scores_and_indexs<<<grid,1>>>(d_score_matrix,d_scores,d_indexs,gb_size,ab_size);
    get_scores_and_indexs<<<grid1,1>>>(d_score_matrix,d_scores0,d_indexs,gb_size,ab_size);
    update_indexs_and_scores_by_max_score<<<grid1,1>>>(d_indexs,d_indexs0,d_scores,d_scores0,d_is_max_score.get());
    get_labels_and_remove_indices<<<grid,1>>>(d_indexs,d_scores,d_is_max_score.get(),glabels,out_labels,out_remove_indices,neg_threshold,pos_threshold);
    get_bboxes_regression<<<grid,1>>>(out_boxes,anchor_bboxes,gbboxes,out_labels,out_remove_indices,out_index,d_prio_scaling.get());

    cudaFree(d_score_matrix);
    cudaFree(d_scores);
    cudaFree(d_indexs);
    cudaFree(d_scores0);
    cudaFree(d_indexs0);
    printf("use gpu\n\n\n\n");
}
#endif
