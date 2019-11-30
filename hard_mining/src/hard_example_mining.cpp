#include <torch/torch.h>
#include <iostream>
#include <vector>

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


void sq_semi_hard_example_mining_launcher(
    at::Tensor feats,         // b,h,w,f
    at::Tensor feats_dist_sq, // b,n
    at::Tensor feats_ref,     // b,n,f
    at::Tensor gt_loc,        // b,n,2
    at::Tensor hard_idxs,     // b,n,3
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

void sq_semi_hard_example_mining(
    at::Tensor feats,      // b,h,w,f
    at::Tensor feats_dist_sq, // b,n
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

    sq_semi_hard_example_mining_launcher(
        feats,feats_dist_sq,feats_ref,gt_loc,
        hard_idxs,interval,thresh_sq,margin,cross_batch);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hard_example_mining", &hard_example_mining, "hard example mining");
    m.def("semi_hard_example_mining", &semi_hard_example_mining, "semi-hard example mining");
    m.def("sq_semi_hard_example_mining", &sq_semi_hard_example_mining, "squared semi-hard example mining");
}