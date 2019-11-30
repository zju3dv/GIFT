#include <float.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int infTwoExp(int val)
{
    int inf=1;
    while(val>inf) inf<<=1;
    return inf;
}

void getGPULayout(
        int dim0,int dim1,int dim2,
        int* bdim0,int* bdim1,int* bdim2,
        int* tdim0,int* tdim1,int* tdim2
)
{
    (*tdim2)=64;
    if(dim2<(*tdim2)) (*tdim2)=infTwoExp(dim2);
    (*bdim2)=dim2/(*tdim2);
    if(dim2%(*tdim2)>0) (*bdim2)++;

    (*tdim1)=1024/(*tdim2);
    if(dim1<(*tdim1)) (*tdim1)=infTwoExp(dim1);
    (*bdim1)=dim1/(*tdim1);
    if(dim1%(*tdim1)>0) (*bdim1)++;

    (*tdim0)=1024/((*tdim1)*(*tdim2));
    if(dim0<(*tdim0)) (*tdim0)=infTwoExp(dim0);
    (*bdim0)=dim0/(*tdim0);
    if(dim0%(*tdim0)>0) (*bdim0)++;
}

__global__
void findNearestFeatureIdxKernel(
    float* ref_pts,   // [b,pn1,dim]
    float* que_pts,   // [b,pn2,dim]
    int* idxs,        // [b,pn2]
    int b,
    int pn1,
    int pn2,
    int dim,
    int exclude_self
)
{
    int bi = threadIdx.x + blockIdx.x*blockDim.x;
    int p2i = threadIdx.y + blockIdx.y*blockDim.y;
    if(p2i>=pn2||bi>=b) return;

    float* que_pt=&que_pts[bi*pn2*dim+p2i*dim];
    float min_dist=FLT_MAX;
    int min_idx=0;
    for(int p1i=0;p1i<pn1;p1i++)
    {
        if(exclude_self&&p1i==p2i) continue;
        float* ref_pt=&ref_pts[bi*pn1*dim+p1i*dim];

        float dist=0.f;
        for(int di=0;di<dim;di++)
            dist+=(ref_pt[di]-que_pt[di])*(ref_pt[di]-que_pt[di]);

        if(dist<min_dist)
        {
            min_dist=dist;
            min_idx=p1i;
        }
    }
    idxs[bi*pn2+p2i]=min_idx;
}

__global__
void findFirstAndSecondNearestFeatureIdxKernel(
    float* ref_pts,   // [b,pn1,dim]
    float* que_pts,   // [b,pn2,dim]
    int* idxs,        // [b,pn2,2]
    float* dists,     // [b,pn2,2]
    int b,
    int pn1,
    int pn2,
    int dim,
    int exclude_self
)
{
    int bi = threadIdx.x + blockIdx.x*blockDim.x;
    int p2i = threadIdx.y + blockIdx.y*blockDim.y;
    if(p2i>=pn2||bi>=b) return;

    float* que_pt=&que_pts[bi*pn2*dim+p2i*dim];
    float min_dist=FLT_MAX,min_dist2=FLT_MAX;
    int min_idx=0, min_idx2=0;
    for(int p1i=0;p1i<pn1;p1i++)
    {
        if(exclude_self&&p1i==p2i) continue;
        float* ref_pt=&ref_pts[bi*pn1*dim+p1i*dim];

        float dist=0.f;
        for(int di=0;di<dim;di++)
            dist+=(ref_pt[di]-que_pt[di])*(ref_pt[di]-que_pt[di]);

        if(dist<min_dist)
        {
            min_dist2=min_dist;
            min_idx2=min_idx;

            min_dist=dist;
            min_idx=p1i;
        }
        else if(dist<min_dist2)
        {
            min_dist2=dist;
            min_idx2=p1i;
        }
    }
    idxs[bi*pn2*2+p2i*2]=min_idx;
    idxs[bi*pn2*2+p2i*2+1]=min_idx2;
    dists[bi*pn2*2+p2i*2]=min_dist;
    dists[bi*pn2*2+p2i*2+1]=min_dist2;
}

#ifdef __cplusplus
extern "C" {
#endif

void findNearestPointIdxLauncher(
    float* ref_pts,   // [b,pn1,dim]
    float* que_pts,   // [b,pn2,dim]
    int* idxs,        // [b,pn2]
    int b,
    int pn1,
    int pn2,
    int dim,
    int exclude_self
)
{
    float* ref_pts_dev,* que_pts_dev;
    int* idxs_dev;
    gpuErrchk(cudaMalloc(&ref_pts_dev,b*pn1*sizeof(float)*dim))
    gpuErrchk(cudaMalloc(&que_pts_dev,b*pn2*sizeof(float)*dim))
    gpuErrchk(cudaMalloc(&idxs_dev,b*pn2*sizeof(int)))

    gpuErrchk(cudaMemcpy(ref_pts_dev,ref_pts,b*pn1*sizeof(float)*dim,cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(que_pts_dev,que_pts,b*pn2*sizeof(float)*dim,cudaMemcpyHostToDevice))

    int bdim0,bdim1,bdim2;
    int tdim0,tdim1,tdim2;

    getGPULayout(b,pn2,1,&bdim0,&bdim1,&bdim2,&tdim0,&tdim1,&tdim2);

    dim3 bdim(bdim0,bdim1,bdim2);
    dim3 tdim(tdim0,tdim1,tdim2);

    findNearestFeatureIdxKernel<<<bdim,tdim>>>(ref_pts_dev,que_pts_dev,idxs_dev,b,pn1,pn2,dim,exclude_self);
    gpuErrchk(cudaGetLastError())

    gpuErrchk(cudaMemcpy(idxs,idxs_dev,b*pn2*sizeof(int),cudaMemcpyDeviceToHost))
    gpuErrchk(cudaFree(ref_pts_dev))
    gpuErrchk(cudaFree(que_pts_dev))
    gpuErrchk(cudaFree(idxs_dev))
}

void findFirstAndSecondNearestFeatureIdxLauncher(
    float* ref_pts,   // [b,pn1,dim]
    float* que_pts,   // [b,pn2,dim]
    int* idxs,        // [b,pn2,2]
    float* dists,     // [b,pn2,2]
    int b,
    int pn1,
    int pn2,
    int dim,
    int exclude_self
)
{
    float* ref_pts_dev,* que_pts_dev;
    int* idxs_dev;
    float* dists_dev;
    gpuErrchk(cudaMalloc(&ref_pts_dev,b*pn1*sizeof(float)*dim))
    gpuErrchk(cudaMalloc(&que_pts_dev,b*pn2*sizeof(float)*dim))
    gpuErrchk(cudaMalloc(&idxs_dev,b*pn2*2*sizeof(int)))
    gpuErrchk(cudaMalloc(&dists_dev,b*pn2*2*sizeof(float)))

    gpuErrchk(cudaMemcpy(ref_pts_dev,ref_pts,b*pn1*sizeof(float)*dim,cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(que_pts_dev,que_pts,b*pn2*sizeof(float)*dim,cudaMemcpyHostToDevice))

    int bdim0,bdim1,bdim2;
    int tdim0,tdim1,tdim2;

    getGPULayout(b,pn2,1,&bdim0,&bdim1,&bdim2,&tdim0,&tdim1,&tdim2);

    dim3 bdim(bdim0,bdim1,bdim2);
    dim3 tdim(tdim0,tdim1,tdim2);

    findFirstAndSecondNearestFeatureIdxKernel<<<bdim,tdim>>>(ref_pts_dev,que_pts_dev,idxs_dev,
                                                             dists_dev,b,pn1,pn2,dim,exclude_self);
    gpuErrchk(cudaGetLastError())

    gpuErrchk(cudaMemcpy(idxs,idxs_dev,b*pn2*2*sizeof(int),cudaMemcpyDeviceToHost))
    gpuErrchk(cudaMemcpy(dists,dists_dev,b*pn2*2*sizeof(float),cudaMemcpyDeviceToHost))
    gpuErrchk(cudaFree(ref_pts_dev))
    gpuErrchk(cudaFree(que_pts_dev))
    gpuErrchk(cudaFree(idxs_dev))
    gpuErrchk(cudaFree(dists_dev))
}

#ifdef __cplusplus
}
#endif
