#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "knncuda.h"

extern THCState* state;

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using namespace std;

void hard_example_mining_launcher(
    at::Tensor feats,      // b,h,w,f
    at::Tensor feats_ref,  // b,n,f
    at::Tensor gt_loc,     // b,n,2
    at::Tensor hard_idxs,  // b,n,3
    int interval,
    float thresh_sq,
    bool cross_batch
);

void hard_example_mining(
    at::Tensor feats,      // b,h,w,f
    at::Tensor feats_ref,  // b,n,f
    at::Tensor gt_loc,     // b,n,2
    at::Tensor hard_idxs,  // b,n,3
    int interval,
    float thresh_sq,
    bool cross_batch
)
{
    CHECK_INPUT(feats);
    CHECK_INPUT(feats_ref);
    CHECK_INPUT(gt_loc);
    CHECK_INPUT(hard_idxs);

    hard_example_mining_launcher(feats,feats_ref,gt_loc,hard_idxs,interval,thresh_sq,cross_batch);
}

void knn_hard_example_mining_launcher(
    at::Tensor feats,      // b,h,w,f
    at::Tensor feats_ref,  // b,n,f
    at::Tensor gt_loc,     // b,n,2
    at::Tensor hard_idxs,  // b,n,k,3
    int interval,
    float thresh_sq
);

void knn_hard_example_mining(
    at::Tensor feats,      // b,h,w,f
    at::Tensor feats_ref,  // b,n,f
    at::Tensor gt_loc,     // b,n,2
    at::Tensor hard_idxs,  // b,n,3
    int interval,
    float thresh_sq
)
{
    CHECK_INPUT(feats);
    CHECK_INPUT(feats_ref);
    CHECK_INPUT(gt_loc);
    CHECK_INPUT(hard_idxs);

    knn_hard_example_mining_launcher(feats,feats_ref,gt_loc,hard_idxs,interval,thresh_sq);
}

void semi_hard_example_mining_launcher(
    at::Tensor feats,      // b,h,w,f
    at::Tensor feats_dist, // b,n
    at::Tensor feats_ref,  // b,n,f
    at::Tensor gt_loc,     // b,n,2
    at::Tensor hard_idxs,  // b,n,3
    int interval,
    float thresh_sq,
    float margin,
    bool cross_batch
);

void semi_hard_example_mining(
    at::Tensor feats,      // b,h,w,f
    at::Tensor feats_dist, // b,n
    at::Tensor feats_ref,  // b,n,f
    at::Tensor gt_loc,     // b,n,2
    at::Tensor hard_idxs,  // b,n,3
    int interval,
    float thresh_sq,
    float margin,
    bool cross_batch
)
{
    CHECK_INPUT(feats);
    CHECK_INPUT(feats_ref);
    CHECK_INPUT(gt_loc);
    CHECK_INPUT(hard_idxs);

    semi_hard_example_mining_launcher(
        feats,feats_dist,feats_ref,gt_loc,
        hard_idxs,interval,thresh_sq,margin,cross_batch);
}

void knn_semi_hard_example_mining_launcher(
    at::Tensor feats,      // b,h,w,f
    at::Tensor feats_dist, // b,n
    at::Tensor feats_ref,  // b,n,f
    at::Tensor gt_loc,     // b,n,2
    at::Tensor hard_idxs,  // b,n,k,3
    int interval,
    float thresh_sq,
    float margin
);

void knn_semi_hard_example_mining(
    at::Tensor feats,      // b,h,w,f
    at::Tensor feats_dist, // b,n
    at::Tensor feats_ref,  // b,n,f
    at::Tensor gt_loc,     // b,n,2
    at::Tensor hard_idxs,  // b,n,k,3
    int interval,
    float thresh_sq,
    float margin
)
{
    CHECK_INPUT(feats);
    CHECK_INPUT(feats_ref);
    CHECK_INPUT(gt_loc);
    CHECK_INPUT(hard_idxs);

    knn_semi_hard_example_mining_launcher(
        feats,feats_dist,feats_ref,gt_loc,
        hard_idxs,interval,thresh_sq,margin);
}

bool knn_cublas_dev(float *       ref_dev,
                    int           ref_nb,
                    float *       query_dev,
                    int           query_nb,
                    int           dim,
                    int           k,
                    float *       knn_dist,
                    int *         knn_index,
                    float*        dist_dev,         // que_nb,
                    int*          index_dev,
                    float*        ref_norm_dev,
                    float*        query_norm_dev
);

void knn_search(
    at::Tensor ref_feats,   // dim,rn
    at::Tensor que_feats,   // dim,qn
    at::Tensor knn_dist,    // k,qn
    at::Tensor knn_index    // k,qn
)
{
    CHECK_INPUT(ref_feats);
    CHECK_INPUT(que_feats);
    CHECK_INPUT(knn_dist);
    CHECK_INPUT(knn_index);

    int que_nb=que_feats.size(1);
    int ref_nb=ref_feats.size(1);
    int dim=que_feats.size(0);
    int k=knn_index.size(0);
    assert(dim==ref_feats.size(0));
    assert(que_nb==knn_dist.size(1));
    assert(que_nb==knn_index.size(1));
    assert(k==knn_index.size(0));

    auto dist_dev=at::zeros({que_nb,ref_nb},ref_feats.type());
    auto ref_norm_dev=at::zeros({ref_nb},ref_feats.type());
    auto que_norm_dev=at::zeros({que_nb},ref_feats.type());
    auto index_dev=at::zeros({que_nb,k},knn_index.type());

    knn_cublas_dev(
        ref_feats.data<float>(), ref_nb,
        que_feats.data<float>(), que_nb,
        dim, k,
        knn_dist.data<float>(),
        knn_index.data<int>(),
        dist_dev.data<float>(),
        index_dev.data<int>(),
        ref_norm_dev.data<float>(),
        que_norm_dev.data<float>()
    );


}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hard_example_mining", &hard_example_mining, "hard example mining");
    m.def("knn_hard_example_mining", &knn_hard_example_mining, "knn hard example mining");
    m.def("semi_hard_example_mining", &semi_hard_example_mining, "semi-hard example mining");
    m.def("knn_semi_hard_example_mining", &knn_semi_hard_example_mining, "knn semi-hard example mining");
    m.def("knn_search", &knn_search, "knn_search");
}