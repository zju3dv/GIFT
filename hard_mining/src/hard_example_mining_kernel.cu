#include "cuda_common.h"
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
//#include <ATen/Error.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <curand_kernel.h>

__global__ void hard_example_mining_kernel(
    float* feats,       // b,h,w,f
    float* feats_ref,   // b,n,f
    float* gt_loc,      // b,n,2
    int* hard_idxs,     // b,n,3
    int interval,       // 16 or 32
    float thresh_sq,    // 8*8
    int hbeg,
    int wbeg,
    int b,int h, int w, int f, int n
)
{
    int bni=threadIdx.x+blockIdx.x*blockDim.x;
    if(bni>=b*n) return;
    int bi=bni/n;
    int ni=bni-bi*n;

    float* feats_cur=&feats_ref[bi*n*f+ni*f];
    float gtx=gt_loc[bi*n*2+ni*2];
    float gty=gt_loc[bi*n*2+ni*2+1];

    float min_dist=FLT_MAX;
    int minh=-1,minw=-1;
    for(int hi=hbeg;hi<h;hi+=interval)
    for(int wi=wbeg;wi<w;wi+=interval)
    {
        if(DIST2D(wi,hi,gtx,gty)<thresh_sq)  continue;
        float* feats_ref=&feats[bi*h*w*f+hi*w*f+wi*f];
        float dist=0.f;
        for(int fi=0;fi<f;fi++)
            dist+=(feats_ref[fi]-feats_cur[fi])*(feats_ref[fi]-feats_cur[fi]);
        if(dist<min_dist)
        {
            min_dist=dist;minh=hi;minw=wi;
        }
    }
    hard_idxs[bi*n*3+ni*3]=bi;
    hard_idxs[bi*n*3+ni*3+1]=minw;
    hard_idxs[bi*n*3+ni*3+2]=minh;
}

__global__ void hard_example_mining_cross_batch_kernel(
    float* feats,       // b,h,w,f
    float* feats_ref,   // b,n,f
    float* gt_loc,      // b,n,2
    int* hard_idxs,     // b,n,3
    int interval,       // 16 or 32
    float thresh_sq,    // 8*8
    int hbeg,
    int wbeg,
    int b,int h, int w, int f, int n
)
{
    int bni=threadIdx.x+blockIdx.x*blockDim.x;
    if(bni>=b*n) return;
    int bi=bni/n;
    int ni=bni-bi*n;

    float* feats_cur=&feats_ref[bi*n*f+ni*f];
    float gtx=gt_loc[bi*n*2+ni*2];
    float gty=gt_loc[bi*n*2+ni*2+1];

    float min_dist=FLT_MAX;
    int minb=-1,minh=-1,minw=-1;
    for(int sbi=0;sbi<b;sbi++)
    for(int hi=hbeg;hi<h;hi+=interval)
    for(int wi=wbeg;wi<w;wi+=interval)
    {
        if(sbi==bi&&DIST2D(wi,hi,gtx,gty)<thresh_sq)  continue;
        float* feats_ref=&feats[sbi*h*w*f+hi*w*f+wi*f];
        float dist=0.f;
        for(int fi=0;fi<f;fi++) dist+=(feats_ref[fi]-feats_cur[fi])*(feats_ref[fi]-feats_cur[fi]);
        if(dist<min_dist)
        {
            minb=bi;min_dist=dist;minh=hi;minw=wi;
        }
    }
    hard_idxs[bi*n*3+ni*3]=minb;
    hard_idxs[bi*n*3+ni*3+1]=minw;
    hard_idxs[bi*n*3+ni*3+2]=minh;
}


__global__ void semi_hard_example_mining_cross_batch_kernel(
    float* feats,       // b,h,w,f
    float* feats_dist,  // b,n
    float* feats_ref,   // b,n,f
    float* gt_loc,      // b,n,2
    int* hard_idxs,     // b,n,3
    int interval,       // 16 or 32
    float thresh_sq,    // 8*8
    float margin,
    int hbeg,
    int wbeg,
    int b,int h, int w, int f, int n
)
{
    int bni=threadIdx.x+blockIdx.x*blockDim.x;
    if(bni>=b*n) return;
    int bi=bni/n;
    int ni=bni-bi*n;

    float* feats_cur=&feats_ref[bi*n*f+ni*f];
    float gtx=gt_loc[bi*n*2+ni*2];
    float gty=gt_loc[bi*n*2+ni*2+1];
    float dist_pos=feats_dist[bi*n+ni];

    int curh=0, curw=0, curb=0;
    for(int sbi=0;sbi<b;sbi++)
    for(int hi=hbeg;hi<h;hi+=interval)
    for(int wi=wbeg;wi<w;wi+=interval)
    {
        if(sbi==bi&&DIST2D(wi,hi,gtx,gty)<thresh_sq)  continue;
        float* feats_ref=&feats[sbi*h*w*f+hi*w*f+wi*f];
        float dist=0.f;
        for(int fi=0;fi<f;fi++)
            dist+=(feats_ref[fi]-feats_cur[fi])*(feats_ref[fi]-feats_cur[fi]);

        if(sqrt(dist)<(dist_pos+margin))
        {
            curh=hi;
            curw=wi;
            curb=sbi;
        }
    }

    hard_idxs[bi*n*3+ni*3]=curb;
    hard_idxs[bi*n*3+ni*3+1]=curw;
    hard_idxs[bi*n*3+ni*3+2]=curh;
}

__global__ void semi_hard_example_mining_kernel(
    float* feats,       // b,h,w,f
    float* feats_dist,  // b,n
    float* feats_ref,   // b,n,f
    float* gt_loc,      // b,n,2
    int* hard_idxs,     // b,n,3
    int interval,       // 16 or 32
    float thresh_sq,    // 8*8
    float margin,
    int hbeg,
    int wbeg,
    int b,int h, int w, int f, int n
)
{
    int bni=threadIdx.x+blockIdx.x*blockDim.x;
    if(bni>=b*n) return;
    int bi=bni/n;
    int ni=bni-bi*n;

    float* feats_cur=&feats_ref[bi*n*f+ni*f];
    float gtx=gt_loc[bi*n*2+ni*2];
    float gty=gt_loc[bi*n*2+ni*2+1];
    float dist_pos=feats_dist[bi*n+ni];

    int curh=0, curw=0;
    for(int hi=hbeg;hi<h;hi+=interval)
    for(int wi=wbeg;wi<w;wi+=interval)
    {
        if(DIST2D(wi,hi,gtx,gty)<thresh_sq)  continue;
        float* feats_ref=&feats[bi*h*w*f+hi*w*f+wi*f];
        float dist=0.f;
        for(int fi=0;fi<f;fi++)
            dist+=(feats_ref[fi]-feats_cur[fi])*(feats_ref[fi]-feats_cur[fi]);

        if(sqrt(dist)<(dist_pos+margin))
        {
            curh=hi;
            curw=wi;
        }
    }

    hard_idxs[bi*n*3+ni*3]=bi;
    hard_idxs[bi*n*3+ni*3+1]=curw;
    hard_idxs[bi*n*3+ni*3+2]=curh;
}

__global__ void setup_kernel(
    curandState * state,
    unsigned long seed,
    int b,
    int n
)
{
    int bni=threadIdx.x+blockIdx.x*blockDim.x;
    if(bni>=b*n) return;
    curand_init(seed,bni,0,&state[bni]);
}

void hard_example_mining_launcher(
    at::Tensor feats,      // b,h,w,f
    at::Tensor feats_ref,  // b,n,f
    at::Tensor gt_loc,     // b,n,2
    at::Tensor hard_idxs,  // b,n,3
    int interval,
    float thresh_sq,
    bool cross_batch
)
{
    int b=feats.size(0);
    int h=feats.size(1);
    int w=feats.size(2);
    int f=feats.size(3);
    int n=feats_ref.size(1);

    assert(feats_ref.size(0)==b);
    assert(feats_ref.size(2)==f);
    assert(gt_loc.size(0)==b);
    assert(gt_loc.size(1)==n);
    assert(gt_loc.size(2)==2);
    assert(hard_idxs.size(0)==b);
    assert(hard_idxs.size(1)==n);
    assert(hard_idxs.size(2)==3);

    int bdim0,bdim1,bdim2;
    int tdim0,tdim1,tdim2;

    getGPULayout(b*n,1,1,&bdim0,&bdim1,&bdim2,&tdim0,&tdim1,&tdim2);

    srand(time(0));
    int hbeg=rand()%interval;
    int wbeg=rand()%interval;

    dim3 bdim(bdim0,bdim1,bdim2);
    dim3 tdim(tdim0,tdim1,tdim2);

    if(cross_batch)
        hard_example_mining_cross_batch_kernel<<<bdim,tdim>>>(
            feats.data<float>(),
            feats_ref.data<float>(),
            gt_loc.data<float>(),
            hard_idxs.data<int>(),
            interval,thresh_sq,hbeg,wbeg,b,h,w,f,n
        );
    else
        hard_example_mining_kernel<<<bdim,tdim>>>(
            feats.data<float>(),
            feats_ref.data<float>(),
            gt_loc.data<float>(),
            hard_idxs.data<int>(),
            interval,thresh_sq,hbeg,wbeg,b,h,w,f,n
        );
    gpuErrchk(cudaGetLastError())
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
)
{
    int b=feats.size(0);
    int h=feats.size(1);
    int w=feats.size(2);
    int f=feats.size(3);
    int n=feats_ref.size(1);

    assert(feats_ref.size(0)==b);
    assert(feats_ref.size(2)==f);
    assert(feats_dist.size(0)==b);
    assert(feats_dist.size(1)==n);
    assert(gt_loc.size(0)==b);
    assert(gt_loc.size(1)==n);
    assert(gt_loc.size(2)==2);
    assert(hard_idxs.size(0)==b);
    assert(hard_idxs.size(1)==n);
    assert(hard_idxs.size(2)==3);

    int bdim0,bdim1,bdim2;
    int tdim0,tdim1,tdim2;

    getGPULayout(b*n,1,1,&bdim0,&bdim1,&bdim2,&tdim0,&tdim1,&tdim2);

    dim3 bdim(bdim0,bdim1,bdim2);
    dim3 tdim(tdim0,tdim1,tdim2);

    srand(time(0));
    int hbeg=rand()%interval;
    int wbeg=rand()%interval;

    if(cross_batch)
        semi_hard_example_mining_cross_batch_kernel<<<bdim,tdim>>>(
            feats.data<float>(),
            feats_dist.data<float>(),
            feats_ref.data<float>(),
            gt_loc.data<float>(),
            hard_idxs.data<int>(),
            interval,thresh_sq,margin,
            hbeg,wbeg,b,h,w,f,n
        );

    else
        semi_hard_example_mining_kernel<<<bdim,tdim>>>(
            feats.data<float>(),
            feats_dist.data<float>(),
            feats_ref.data<float>(),
            gt_loc.data<float>(),
            hard_idxs.data<int>(),
            interval,thresh_sq,margin,
            hbeg,wbeg,b,h,w,f,n
        );
    gpuErrchk(cudaGetLastError())
}

/////////////////////////////////////////////////
__global__ void knn_hard_example_mining_kernel(
    float* feats,       // b,h,w,f
    float* feats_ref,   // b,n,f
    float* gt_loc,      // b,n,2
    int* hard_idxs,     // b,n,k,3
    int interval,       // 16 or 32
    float thresh_sq,    // 8 * 8
    int ki, int k,
    int hbeg, int wbeg,
    int b,int h, int w, int f, int n
)
{
    int bni=threadIdx.x+blockIdx.x*blockDim.x;
    if(bni>=b*n) return;
    int bi=bni/n;
    int ni=bni-bi*n;

    float* feats_cur=&feats_ref[bi*n*f+ni*f];
    float gtx=gt_loc[bi*n*2+ni*2];
    float gty=gt_loc[bi*n*2+ni*2+1];
    int* cur_hard_idxs=&hard_idxs[bi*n*k*3+ni*k*3];

    float min_dist=FLT_MAX;
    int minh=-1,minw=-1;
    for(int hi=hbeg;hi<h;hi+=interval)
    for(int wi=wbeg;wi<w;wi+=interval)
    {
        if(DIST2D(wi,hi,gtx,gty)<thresh_sq)  continue;
        bool continue_flag=false;
        for(int ski=0;ski<ki;ski++)
        {
            int pw=cur_hard_idxs[ski*3+1];
            int ph=cur_hard_idxs[ski*3+2];
            if(DIST2D(wi,hi,float(pw),float(ph))<thresh_sq)
            {
                continue_flag=true;
                break;
            }
        }
        if(continue_flag) continue;

        float* feats_que=&feats[bi*h*w*f+hi*w*f+wi*f];
        float dist=0.f;
        for(int fi=0;fi<f;fi++)
            dist+=(feats_que[fi]-feats_cur[fi])*(feats_que[fi]-feats_cur[fi]);
        if(dist<min_dist)
        {
            min_dist=dist;minh=hi;minw=wi;
        }
    }
    hard_idxs[bi*n*k*3+ni*k*3+ki*3]=bi;
    hard_idxs[bi*n*k*3+ni*k*3+ki*3+1]=minw;
    hard_idxs[bi*n*k*3+ni*k*3+ki*3+2]=minh;
}


void knn_hard_example_mining_launcher(
    at::Tensor feats,      // b,h,w,f
    at::Tensor feats_ref,  // b,n,f
    at::Tensor gt_loc,     // b,n,2
    at::Tensor hard_idxs,  // b,n,k,3
    int interval,
    float thresh_sq
)
{
    int b=feats.size(0);
    int h=feats.size(1);
    int w=feats.size(2);
    int f=feats.size(3);
    int n=feats_ref.size(1);
    int k=hard_idxs.size(2);

    assert(feats_ref.size(0)==b);
    assert(feats_ref.size(2)==f);
    assert(gt_loc.size(0)==b);
    assert(gt_loc.size(1)==n);
    assert(gt_loc.size(2)==2);
    assert(hard_idxs.size(0)==b);
    assert(hard_idxs.size(1)==n);
    assert(hard_idxs.size(3)==3);

    int bdim0,bdim1,bdim2;
    int tdim0,tdim1,tdim2;

    getGPULayout(b*n,1,1,&bdim0,&bdim1,&bdim2,&tdim0,&tdim1,&tdim2);

    srand(time(0));
    int hbeg=rand()%interval;
    int wbeg=rand()%interval;

    dim3 bdim(bdim0,bdim1,bdim2);
    dim3 tdim(tdim0,tdim1,tdim2);

    for(int ki=0;ki<k;ki++)
    {
        hbeg=rand()%interval;
        wbeg=rand()%interval;
        knn_hard_example_mining_kernel<<<bdim,tdim>>>(
            feats.data<float>(),
            feats_ref.data<float>(),
            gt_loc.data<float>(),
            hard_idxs.data<int>(),
            interval,thresh_sq,ki,k,hbeg,wbeg,b,h,w,f,n
        );
        gpuErrchk(cudaGetLastError())
    }
}

//////////////////////////////////////////////////

__global__ void knn_semi_hard_example_mining_kernel(
    float* feats,       // b,h,w,f
    float* feats_dist,  // b,n
    float* feats_ref,   // b,n,f
    float* gt_loc,      // b,n,2
    int* hard_idxs,     // b,n,k,3
    int interval,       // 16 or 32
    float thresh_sq,    // 8 * 8
    float margin,
    int ki, int k,
    int hbeg, int wbeg,
    int b,int h, int w, int f, int n
)
{
    int bni=threadIdx.x+blockIdx.x*blockDim.x;
    if(bni>=b*n) return;
    int bi=bni/n;
    int ni=bni-bi*n;

    float* feats_cur=&feats_ref[bi*n*f+ni*f];
    float gtx=gt_loc[bi*n*2+ni*2];
    float gty=gt_loc[bi*n*2+ni*2+1];
    int* cur_hard_idxs=&hard_idxs[bi*n*k*3+ni*k*3];
    float cur_dist=feats_dist[bi*n+ni];

    int minh=0, minw=0;
    for(int hi=hbeg;hi<h;hi+=interval)
    for(int wi=wbeg;wi<w;wi+=interval)
    {
        if(DIST2D(wi,hi,gtx,gty)<thresh_sq)  continue;
        bool continue_flag=false;
        for(int ski=0;ski<ki;ski++)
        {
            int pw=cur_hard_idxs[ski*3+1];
            int ph=cur_hard_idxs[ski*3+2];
            if(DIST2D(wi,hi,float(pw),float(ph))<thresh_sq)
            {
                continue_flag=true;
                break;
            }
        }
        if(continue_flag) continue;

        float* feats_que=&feats[bi*h*w*f+hi*w*f+wi*f];
        float dist=0.f;
        for(int fi=0;fi<f;fi++) dist+=(feats_que[fi]-feats_cur[fi])*(feats_que[fi]-feats_cur[fi]);

        if(dist<(cur_dist+margin))
        {minh=hi;minw=wi;}
    }
    hard_idxs[bi*n*k*3+ni*k*3+ki*3]=bi;
    hard_idxs[bi*n*k*3+ni*k*3+ki*3+1]=minw;
    hard_idxs[bi*n*k*3+ni*k*3+ki*3+2]=minh;
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
)
{
    int b=feats.size(0);
    int h=feats.size(1);
    int w=feats.size(2);
    int f=feats.size(3);
    int n=feats_ref.size(1);
    int k=hard_idxs.size(2);

    assert(feats_ref.size(0)==b);
    assert(feats_ref.size(2)==f);
    assert(gt_loc.size(0)==b);
    assert(gt_loc.size(1)==n);
    assert(gt_loc.size(2)==2);
    assert(hard_idxs.size(0)==b);
    assert(hard_idxs.size(1)==n);
    assert(hard_idxs.size(3)==3);

    int bdim0,bdim1,bdim2;
    int tdim0,tdim1,tdim2;

    getGPULayout(b*n,1,1,&bdim0,&bdim1,&bdim2,&tdim0,&tdim1,&tdim2);

    srand(time(0));
    int hbeg=rand()%interval;
    int wbeg=rand()%interval;

    dim3 bdim(bdim0,bdim1,bdim2);
    dim3 tdim(tdim0,tdim1,tdim2);

    for(int ki=0;ki<k;ki++)
    {
        // hbeg=rand()%interval;
        // wbeg=rand()%interval;
        knn_semi_hard_example_mining_kernel<<<bdim,tdim>>>(
            feats.data<float>(),
            feats_dist.data<float>(),
            feats_ref.data<float>(),
            gt_loc.data<float>(),
            hard_idxs.data<int>(),
            interval,thresh_sq,margin,ki,k,hbeg,wbeg,b,h,w,f,n
        );
        gpuErrchk(cudaGetLastError())
    }
}