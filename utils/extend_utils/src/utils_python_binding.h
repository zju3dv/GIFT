void findNearestPointIdxLauncher(
    float* ref_pts,   // [b,pn1,dim]
    float* que_pts,   // [b,pn2,dim]
    int* idxs,        // [b,pn2]
    int b,
    int pn1,
    int pn2,
    int dim,
    int exclude_self
);


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
);

void findSetDistanceLauncher(
    float* ref_pts,   // [b,pn1,k,dim]
    float* que_pts,   // [b,pn2,k,dim]
    float* dist,      // [b,pn1,pn2]
    int b,
    int pn1,
    int pn2,
    int k,
    int dim
);

void findFeatsDistanceLauncher(
    float* ref_pts,   // [b,pn1,dim]
    float* que_pts,   // [b,pn2,dim]
    float* dist,      // [b,pn1,pn2]
    int b,
    int pn1,
    int pn2,
    int dim
);

void countNeighborhoodLauncher(
    float* pts,       // [pn,dim]
    int* count,       // [pn]
    float radius,
    int pn,
    int dim
);