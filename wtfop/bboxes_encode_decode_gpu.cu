#include <vector>
#include "wtoolkit_cuda.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
using namespace std;
constexpr auto kBlockSize = 64;

#ifdef GOOGLE_CUDA
template<typename T>
__device__ std::tuple<T,T,T,T> cuda_box_minmax_to_cxywh(const T* box)
{
	return std::make_tuple((box[0]+box[2])/2.0,(box[1]+box[3])/2.0,box[2]-box[0],box[3]-box[1]);
}
template<typename T>
__device__ std::tuple<T,T,T,T> cuda_box_cxywh_to_minmax(const T* box)
{
	return std::make_tuple(box[0]-box[2]/2.0,box[1]-box[3]/2.0,box[0]+box[2]/2.,box[1]+box[3]/2);
}
template<typename T>
__device__ std::tuple<T,T,T,T> cuda_box_cxywh_to_minmax(T cy, T cx, T h, T w)
{
	return std::make_tuple(cy-h/2.0,cx-w/2.,cy+h/2.0,cx+w/2.0);
}
template<typename T0,typename T1>
__device__ float cuda_bboxes_jaccard(const T0* box0, const T1* box1)
{
	const auto  int_ymin  = std::max(box0[0],box1[0]);
	const auto  int_xmin  = std::max(box0[1],box1[1]);
	const auto  int_ymax  = std::min(box0[2],box1[2]);
	const auto  int_xmax  = std::min(box0[3],box1[3]);
	const float int_h     = std::max<float>(int_ymax-int_ymin,0.);
	const float int_w     = std::max<float>(int_xmax-int_xmin,0.);
	const auto  int_vol   = int_h *int_w;
	const auto  vol1      = (box0[2]-box0[0]) *(box0[3]-box0[1]);
	const auto  vol2      = (box1[2]-box1[0]) *(box1[3]-box1[1]);
	const auto  union_vol = vol1+vol2-int_vol;

	if(union_vol<1E-6) return 0.0f;

	return int_vol/union_vol;
}
template<typename T>
__device__ T clamp(T v,T min, T max)
{
    if(v<min) return min;
    if(v>max) return max;
    return v;
}
template<typename T>
__device__ inline bool cuda_is_cross_boundaries(const T* box) {
    return (box[0]<0.0) || (box[1]<0.0) || (box[2]>1.0) ||(box[3]>1.0);
}
__global__ void get_scores_and_indexs(const float* gbboxes,const float* anchor_bboxes,float* scores,int* indexs,bool* is_boundary_box,size_t gb_size,size_t ab_size)
{
    const auto       a_index                = blockIdx.x;
    const auto       g_offset               = threadIdx.x;
    auto             max_i                  = -1;
    auto             max_s                  = 1e-8;
    float            abbox[4];
    float            gbbox[4];
    __shared__ short max_index[kBlockSize];
    __shared__ float max_scores[kBlockSize];

    for(auto i=0; i<4; ++i)
        abbox[i] = (anchor_bboxes+(a_index<<2))[i];

    if(cuda_is_cross_boundaries(abbox)) { 
        is_boundary_box[a_index] = true;
        return;
    }

    for(auto i=g_offset; i<gb_size; i += blockDim.x) {
        for(auto j=0; j<4; ++j)
            gbbox[j] = (gbboxes+(i<<2))[j];
        const auto cs = cuda_bboxes_jaccard(abbox,gbbox);
        //const auto cs = cuda_bboxes_jaccard(abbox,gbboxes+(i<<2));
        if(cs>max_s) {
            max_i = i;
            max_s = cs;
        }
    }
    max_index[g_offset] = max_i;
    max_scores[g_offset] = max_s;
    __syncthreads();
    if(g_offset != 0) return; 

    max_i = -1;
    max_s = 1e-8;
    for(auto i=0; i<blockDim.x; ++i) {
        const auto cs = max_scores[i];
        if(cs>max_s) {
            max_i = i;
            max_s = cs;
        }
    }
    if(max_i>=0) {
        indexs[a_index] = max_index[max_i];
        scores[a_index] = max_s;
    }
}
__global__ void find_max_score_index(const float* gbboxes,const float* anchor_bboxes,const bool* is_boundary_box,float* scores0,int* indexs0,size_t gb_size,size_t ab_size)
{
    const auto       g_index                = blockIdx.x;
    const auto       a_offset               = threadIdx.x;
    auto             max_i                  = -1;
    auto             max_s                  = 1e-8;
    float            gbbox[4];
    float            abbox[4];
    __shared__ short max_index[kBlockSize];
    __shared__ float max_scores[kBlockSize];

    for(auto i=0; i<4; ++i)
        gbbox[i] = (gbboxes+(g_index<<2))[i];

    for(auto i=a_offset; i<ab_size; i += blockDim.x) {
        for(auto j=0; j<4; ++j)
            abbox[j] = (anchor_bboxes+(i<<2))[j];
        if(is_boundary_box[i]) continue;
        const auto cs = cuda_bboxes_jaccard(gbbox,abbox);
        if(cs>max_s) {
            max_i = i;
            max_s = cs;
        }
    }
    max_index[a_offset] = max_i;
    max_scores[a_offset] = max_s;
    __syncthreads();
    if(a_offset != 0) return;
    max_i = -1;
    max_s = 1e-8;
    for(auto i=0; i<blockDim.x; ++i) {
        const auto cs = max_scores[i];
        if(cs>max_s) {
            max_i = i;
            max_s = cs;
        }
    }
    if(max_i>=0) {
        indexs0[g_index] = max_index[max_i];
        scores0[g_index] = max_s;
    }
}
__global__ void update_indexs_and_scores_by_max_score(int* indexs,int* indexs0,float* scores,float* scores0,bool* is_max_score,size_t gb_size,size_t ab_size)
{
    auto             a_index                = blockIdx.x;
    auto             g_offset               = threadIdx.x;
    __shared__ int   max_index[kBlockSize];
    __shared__ float max_scores[kBlockSize];

    max_index[g_offset] = -1;
    max_scores[g_offset] = 1e-8;

    for(auto i=g_offset; i<gb_size; i += blockDim.x) {
        if(indexs0[i]<0) continue;
        const auto la_index = indexs0[i];
        if(la_index != a_index) continue;
        if((max_index[g_offset]<0) || (max_scores[g_offset]<scores0[i])) {
            max_scores[g_offset]  =  scores0[i];
            max_index[g_offset]   =  i;
        }
    }
    __syncthreads();

    if(g_offset != 0) return;

    int   max_i = -1;
    float max_s = 1e-8;
    for(auto i=0; i<blockDim.x; ++i) {
        if((max_index[i]>=0) && (max_scores[i]>max_s)) {
            max_s = max_scores[i];
            max_i = i;
        }
    }
    if(max_i >= 0) {
        is_max_score[a_index] = true;
        scores[a_index] = max_s;
        indexs[a_index] = max_index[max_i];
    }
}
__global__ void get_labels_and_remove_indices(int* indexs,float* scores,const bool* is_max_score,const int* glabels,int* out_labels,bool* remove_indices,float neg_threshold,float pos_threshold)
{
    auto        a_index = blockIdx.x;
    const auto &index   = indexs[a_index];
    const auto  score   = scores[a_index];

    if((score>=pos_threshold) || (score<neg_threshold) || is_max_score[a_index]) {
        remove_indices[a_index] = false;
        if((score>=pos_threshold) || is_max_score[a_index]) {
            out_labels[a_index] = glabels[index];
        } else {
            out_labels[a_index]  =  0;
            indexs[a_index]      =  -1;
            scores[a_index]      =  0;
        }
    } else {
        remove_indices[a_index]  =  true;
        indexs[a_index]          =  -1;
        scores[a_index]          =  0.0f;
    }
}
__global__ void get_bboxes_regression(float* out_boxes,const float* anchor_bboxes,const float* gbboxes,const int* out_labels,const bool* out_remove_indices,const int* out_index,float* prio_scaling)
{
    auto j = blockIdx.x; //a_index

    auto  outbox  = out_boxes+j*4;
    if((out_labels[j]<1) || (out_remove_indices[j])) {
        return;
    }
    auto box  = anchor_bboxes+j *4;
    auto gbox = gbboxes+out_index[j] *4;
    auto yxhw = cuda_box_minmax_to_cxywh(box);
    auto yref = std::get<0>(yxhw);
    auto xref = std::get<1>(yxhw);
    auto href = std::get<2>(yxhw);
    auto wref = std::get<3>(yxhw);

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
__global__ void bboxes_decode_kernel(const float* anchor_bboxes,const float* regs,const float* prio_scaling,float* out_bboxes,size_t data_nr)
{
    const auto b           = threadIdx.x+blockIdx.x *blockDim.x;

    if(b>=data_nr) return;

    const auto base_offset = b *4;
    const auto regs_data   = regs+base_offset;
    const auto box_data    = anchor_bboxes+base_offset;
    float      y;
    float      x;
    float      href;
    float      wref;
    auto       xywh        = cuda_box_minmax_to_cxywh(box_data);

    y = std::get<0>(xywh);
    x = std::get<1>(xywh);
    href = std::get<2>(xywh);
    wref = std::get<3>(xywh);

    auto       cy          = clamp<float>(regs_data[0] *prio_scaling[0],-10.0f,10.0f) *href+y;
    auto       cx          = clamp<float>(regs_data[1] *prio_scaling[1],-10.0f,10.0f) *wref+x;
    auto       h           = href *exp(clamp<float>(regs_data[2] *prio_scaling[2],-10.0,10.0));
    auto       w           = wref *exp(clamp<float>(regs_data[3] *prio_scaling[3],-10.0,10.0));
    auto       output_data = out_bboxes + base_offset;
    const auto minmax      = cuda_box_cxywh_to_minmax(cy,cx,h,w);

    output_data[0] = clamp<float>(std::get<0>(minmax),0.0,1.0);
    output_data[1] = clamp<float>(std::get<1>(minmax),0.0,1.0);
    output_data[2] = clamp<float>(std::get<2>(minmax),0.0,1.0);
    output_data[3] = clamp<float>(std::get<3>(minmax),0.0,1.0); 

    if(output_data[0]>output_data[2]) 
        output_data[2] = output_data[0];
    if(output_data[1]>output_data[3])
        output_data[3] = output_data[1];
}
__host__ void get_encodes(const float* gbboxes,const float* anchor_bboxes,const int* glabels,
float* out_boxes,float* out_scores,int* out_labels,bool* out_remove_indices,int* out_index,const float* prio_scaling,
size_t gb_size,size_t ab_size,float neg_threshold,float pos_threshold,bool max_overlap_as_pos=true)
{
    cuda_unique_ptr<int> g_out_index;

    if(nullptr == out_index) {
        g_out_index = make_cuda_unique<int>(ab_size);
        out_index = g_out_index.get(); 
    }

    CHECK_OK(cudaMemset(out_boxes,0,sizeof(float)*4*ab_size));
    CHECK_OK(cudaMemset(out_scores,0,sizeof(float)*ab_size));
    CHECK_OK(cudaMemset(out_index,0xff,sizeof(int)*ab_size));
    CHECK_OK(cudaMemset(out_labels,0,sizeof(int)*ab_size));

    const int block_limits      = 1024;
    dim3      grid(ab_size);
    dim3      grid1(gb_size);
    auto      d_is_boundary_box = make_cuda_unique<bool>((unsigned char)(0x00),ab_size);

    get_scores_and_indexs<<<grid,std::min<size_t>(kBlockSize,gb_size)>>>(gbboxes,anchor_bboxes,out_scores,out_index,d_is_boundary_box.get(),gb_size,ab_size);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    cuda_unique_ptr<int> d_indexs0 = make_cuda_unique<int>((unsigned char)(0xff),gb_size);
    cuda_unique_ptr<float> d_scores0 = make_cuda_unique<float>((unsigned char)(0x00),gb_size);

    find_max_score_index<<<grid1,std::min<size_t>(kBlockSize,ab_size)>>>(gbboxes,anchor_bboxes,d_is_boundary_box.get(),d_scores0.get(),d_indexs0.get(),gb_size,ab_size);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    cuda_unique_ptr<bool> d_is_max_score = make_cuda_unique<bool>((unsigned char)(0x00),ab_size);

    if(max_overlap_as_pos)
        update_indexs_and_scores_by_max_score<<<grid,std::min<size_t>(kBlockSize,gb_size)>>>(out_index,d_indexs0.get(),out_scores,d_scores0.get(),d_is_max_score.get(),gb_size,ab_size);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());

    get_labels_and_remove_indices<<<grid,1>>>(out_index,out_scores,d_is_max_score.get(),glabels,out_labels,out_remove_indices,neg_threshold,pos_threshold);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    cuda_unique_ptr<float> d_prio_scaling = make_cuda_unique<float>(prio_scaling,4);

    get_bboxes_regression<<<grid,1>>>(out_boxes,anchor_bboxes,gbboxes,out_labels,out_remove_indices,out_index,d_prio_scaling.get());

    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}
void bboxes_decode_by_gpu(const float* anchor_bboxes,const float* regs,const float* prio_scaling,float* out_bboxes,size_t data_nr)
{
    if(0 == data_nr) 
        return;
    cuda_unique_ptr<float> d_prio_scaling = make_cuda_unique<float>(prio_scaling,4);
    const auto block_size = std::min<size_t>(data_nr,128);
    const auto grid_size = (data_nr+block_size-1)/block_size;

    bboxes_decode_kernel<<<grid_size,block_size>>>(anchor_bboxes,regs,d_prio_scaling.get(),out_bboxes,data_nr);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    cudaDeviceSynchronize(); 
}
#endif
