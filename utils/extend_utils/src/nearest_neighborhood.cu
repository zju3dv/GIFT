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
void findNearestSetFeatureIdxKernel(
    float* ref_pts,   // [b,pn1,k,dim]
    float* que_pts,   // [b,pn2,k,dim]
    int* idxs,        // [b,pn2]
    int b,
    int pn1,
    int pn2,
    int k,
    int dim,
    int exclude_self
)
{
    int bi = threadIdx.x + blockIdx.x*blockDim.x;
    int p2i = threadIdx.y + blockIdx.y*blockDim.y;
    if(p2i>=pn2||bi>=b) return;

    float* que_pt=&que_pts[bi*pn2*k*dim+p2i*k*dim];
    float min_dist=FLT_MAX;
    int min_idx=0;
    for(int p1i=0;p1i<pn1;p1i++)
    {
        if(exclude_self&&p1i==p2i) continue;
        float* ref_pt=&ref_pts[bi*pn1*k*dim+p1i*k*dim];

        float min_set_dist=FLT_MAX;
        for(int k1i=0;k1i<k;k1i++)
        {
            for(int k2i=0;k2i<k;k2i++)
            {
                float set_dist=0.f;
                for(int di=0;di<dim;di++)
                    set_dist+=(ref_pt[k1i*dim+di]-que_pt[k2i*dim+di])*(ref_pt[k1i*dim+di]-que_pt[k2i*dim+di]);
                if(set_dist<min_set_dist)
                    min_set_dist=set_dist;
            }
        }

        if(min_set_dist<min_dist)
        {
            min_dist=min_set_dist;
            min_idx=p1i;
        }
    }
    idxs[bi*pn2+p2i]=min_idx;
}

__global__
void findFeatsDistanceKernel(
    float* ref_pts,   // [b,pn1,dim]
    float* que_pts,   // [b,pn2,dim]
    float* dist,      // [b,pn1,pn2]
    int b,
    int pn1,
    int pn2,
    int dim
)
{
    int bp2i = threadIdx.x + blockIdx.x*blockDim.x;
    int p1i = threadIdx.y + blockIdx.y*blockDim.y;
    if(p1i>=pn1||bp2i>=b*pn2) return;
    int bi = bp2i/pn2;
    int p2i = bp2i%pn2;

    float* rpts=&ref_pts[bi*pn1*dim+p1i*dim];
    float* qpts=&que_pts[bi*pn2*dim+p2i*dim];
    float cur_dist=0.f;
    for(int di=0;di<dim;di++)
        cur_dist+=(rpts[di]-qpts[di])*(rpts[di]-qpts[di]);

    dist[bi*pn1*pn2+p1i*pn2+p2i]=cur_dist;
}

__global__
void findSetDistanceKernel(
    float* ref_pts,   // [b,pn1,k,dim]
    float* que_pts,   // [b,pn2,k,dim]
    float* dist,      // [b,pn1,pn2]
    int b,
    int pn1,
    int pn2,
    int k,
    int dim
)
{
    int bi = threadIdx.x + blockIdx.x*blockDim.x;
    int p12i = threadIdx.y + blockIdx.y*blockDim.y;
    if(p12i>=pn1*pn2||bi>=b) return;
    int p1i = p12i/pn2;
    int p2i = p12i%pn2;

    float* rpts=&ref_pts[bi*pn1*k*dim+p1i*k*dim];
    float* qpts=&que_pts[bi*pn2*k*dim+p2i*k*dim];
    float min_dist=FLT_MAX;
    for(int k1i=0;k1i<k;k1i++)
    for(int k2i=0;k2i<k;k2i++)
    {
        float cur_dist=0.f;
        for(int di=0;di<dim;di++)
            cur_dist+=(rpts[k1i*dim+di]-qpts[k2i*dim+di])*(rpts[k1i*dim+di]-qpts[k2i*dim+di]);

        if(min_dist>cur_dist)
            min_dist=cur_dist;
    }
    dist[bi*pn1*pn2+p1i*pn2+p2i]=min_dist;
}

__global__
void countNeighborhoodKernel(
    float* pts,
    int* count,
    float radius,
    int pn,
    int dim
)
{
    int pi = threadIdx.y + blockIdx.y*blockDim.y;

    if(pi>=pn) return;

    float* cpt=&pts[pi*dim];
    float radius_sq=radius*radius;
    count[pi]=0;
    for(int pqi=0;pqi<pn;pqi++)
    {
        float dist_sq=0.f;
        for(int di=0;di<dim;di++)
            dist_sq+=(pts[pqi*dim+di]-cpt[di])*(pts[pqi*dim+di]-cpt[di]);
        if(dist_sq<=radius_sq)
            count[pi]+=1;
    }
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
    gpuErrchk(cudaMemcpy(idxs_dev,idxs,b*pn2*sizeof(int),cudaMemcpyHostToDevice))

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

void findNearestSetPointIdxLauncher(
    float* ref_pts,   // [b,pn1,k,dim]
    float* que_pts,   // [b,pn2,k,dim]
    int* idxs,        // [b,pn2]
    int b,
    int pn1,
    int pn2,
    int k,
    int dim,
    int exclude_self
)
{
    float* ref_pts_dev,* que_pts_dev;
    int* idxs_dev;
    gpuErrchk(cudaMalloc(&ref_pts_dev,b*pn1*sizeof(float)*k*dim))
    gpuErrchk(cudaMalloc(&que_pts_dev,b*pn2*sizeof(float)*k*dim))
    gpuErrchk(cudaMalloc(&idxs_dev,b*pn2*sizeof(int)))

    gpuErrchk(cudaMemcpy(ref_pts_dev,ref_pts,b*pn1*sizeof(float)*k*dim,cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(que_pts_dev,que_pts,b*pn2*sizeof(float)*k*dim,cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(idxs_dev,idxs,b*pn2*sizeof(int),cudaMemcpyHostToDevice))

    int bdim0,bdim1,bdim2;
    int tdim0,tdim1,tdim2;

    getGPULayout(b,pn2,1,&bdim0,&bdim1,&bdim2,&tdim0,&tdim1,&tdim2);

    dim3 bdim(bdim0,bdim1,bdim2);
    dim3 tdim(tdim0,tdim1,tdim2);

    findNearestSetFeatureIdxKernel<<<bdim,tdim>>>(ref_pts_dev,que_pts_dev,idxs_dev,b,pn1,pn2,k,dim,exclude_self);
    gpuErrchk(cudaGetLastError())

    gpuErrchk(cudaMemcpy(idxs,idxs_dev,b*pn2*sizeof(int),cudaMemcpyDeviceToHost))
    gpuErrchk(cudaFree(ref_pts_dev))
    gpuErrchk(cudaFree(que_pts_dev))
    gpuErrchk(cudaFree(idxs_dev))
}

void findSetDistanceLauncher(
    float* ref_pts,   // [b,pn1,k,dim]
    float* que_pts,   // [b,pn2,k,dim]
    float* dist,      // [b,pn1,pn2]
    int b,
    int pn1,
    int pn2,
    int k,
    int dim
)
{
    // printf("b pn1 pn2 k dim %d %d %d %d %d\n",b,pn1,pn2,k,dim);
    float* ref_pts_dev,* que_pts_dev,* dist_dev;
    gpuErrchk(cudaMalloc(&ref_pts_dev,b*pn1*sizeof(float)*k*dim))
    gpuErrchk(cudaMalloc(&que_pts_dev,b*pn2*sizeof(float)*k*dim))
    gpuErrchk(cudaMalloc(&dist_dev,b*pn2*pn1*sizeof(float)))

    gpuErrchk(cudaMemcpy(ref_pts_dev,ref_pts,b*pn1*sizeof(float)*k*dim,cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(que_pts_dev,que_pts,b*pn2*sizeof(float)*k*dim,cudaMemcpyHostToDevice))

    int bdim0,bdim1,bdim2;
    int tdim0,tdim1,tdim2;

    getGPULayout(b,pn2*pn1,1,&bdim0,&bdim1,&bdim2,&tdim0,&tdim1,&tdim2);

    dim3 bdim(bdim0,bdim1,bdim2);
    dim3 tdim(tdim0,tdim1,tdim2);

    findSetDistanceKernel<<<bdim,tdim>>>(ref_pts_dev,que_pts_dev,dist_dev,b,pn1,pn2,k,dim);
    gpuErrchk(cudaGetLastError())

    gpuErrchk(cudaMemcpy(dist,dist_dev,b*pn1*pn2*sizeof(float),cudaMemcpyDeviceToHost))
    gpuErrchk(cudaFree(ref_pts_dev))
    gpuErrchk(cudaFree(que_pts_dev))
    gpuErrchk(cudaFree(dist_dev))
}

void findFeatsDistanceLauncher(
    float* ref_pts,   // [b,pn1,dim]
    float* que_pts,   // [b,pn2,dim]
    float* dist,      // [b,pn1,pn2]
    int b,
    int pn1,
    int pn2,
    int dim
)
{
    float* ref_pts_dev,* que_pts_dev,* dist_dev;
    gpuErrchk(cudaMalloc(&ref_pts_dev,b*pn1*sizeof(float)*dim))
    gpuErrchk(cudaMalloc(&que_pts_dev,b*pn2*sizeof(float)*dim))
    gpuErrchk(cudaMalloc(&dist_dev,b*pn2*pn1*sizeof(float)))

    gpuErrchk(cudaMemcpy(ref_pts_dev,ref_pts,b*pn1*sizeof(float)*dim,cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(que_pts_dev,que_pts,b*pn2*sizeof(float)*dim,cudaMemcpyHostToDevice))

    int bdim0,bdim1,bdim2;
    int tdim0,tdim1,tdim2;

    getGPULayout(b*pn2,pn1,1,&bdim0,&bdim1,&bdim2,&tdim0,&tdim1,&tdim2);

    dim3 bdim(bdim0,bdim1,bdim2);
    dim3 tdim(tdim0,tdim1,tdim2);

    // printf("%d %d %d %d %d %d \n",bdim0,bdim1,bdim2,tdim0,tdim1,tdim2);
    findFeatsDistanceKernel<<<bdim,tdim>>>(ref_pts_dev,que_pts_dev,dist_dev,b,pn1,pn2,dim);
    gpuErrchk(cudaGetLastError())

    gpuErrchk(cudaMemcpy(dist,dist_dev,b*pn1*pn2*sizeof(float),cudaMemcpyDeviceToHost))
    gpuErrchk(cudaFree(ref_pts_dev))
    gpuErrchk(cudaFree(que_pts_dev))
    gpuErrchk(cudaFree(dist_dev))
}

void countNeighborhoodLauncher(
    float* pts,       // [pn,dim]
    int* count,       // [pn]
    float radius,
    int pn,
    int dim
)
{
    float* pts_dev;
    int* count_dev;
    gpuErrchk(cudaMalloc(&pts_dev,pn*dim*sizeof(float)))
    gpuErrchk(cudaMalloc(&count_dev,pn*sizeof(int)))

    gpuErrchk(cudaMemcpy(pts_dev,pts,pn*dim*sizeof(float),cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(count_dev,count,pn*sizeof(int),cudaMemcpyHostToDevice))

    int bdim0,bdim1,bdim2;
    int tdim0,tdim1,tdim2;

    getGPULayout(1,pn,1,&bdim0,&bdim1,&bdim2,&tdim0,&tdim1,&tdim2);

    dim3 bdim(bdim0,bdim1,bdim2);
    dim3 tdim(tdim0,tdim1,tdim2);

    countNeighborhoodKernel<<<bdim,tdim>>>(pts_dev,count_dev,radius,pn,dim);
    gpuErrchk(cudaGetLastError())

    gpuErrchk(cudaMemcpy(count,count_dev,pn*sizeof(int),cudaMemcpyDeviceToHost))
    gpuErrchk(cudaFree(pts_dev))
    gpuErrchk(cudaFree(count_dev))
}

#ifdef __cplusplus
}
#endif
