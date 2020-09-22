#ifndef H_VORONOI_H
#define H_VORONOI_H

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

#include "voronoi_defs.h"
#include "params.h"
#include "knearests.h"

#define cuda_check(x) if (x!=cudaSuccess) exit(1);

#define FOR(I,UPPERBND) for(int I = 0; I<int(UPPERBND); ++I)

typedef unsigned char uchar;  // local indices with special values

__shared__ uchar3 tr_data[VORO_BLOCK_SIZE * _MAX_T_]; // memory pool for chained lists of triangles
__shared__ uchar boundary_next_data[VORO_BLOCK_SIZE * _MAX_P_];
__shared__ float4 clip_data[VORO_BLOCK_SIZE * _MAX_P_]; // clipping planes

inline  __device__ uchar3& tr(int t) { return  tr_data[threadIdx.x*_MAX_T_ + t]; }
inline  __device__ uchar& boundary_next(int v) { return  boundary_next_data[threadIdx.x*_MAX_P_ + v]; }
inline  __device__ float4& clip(int v) { return  clip_data[threadIdx.x*_MAX_P_ + v]; }

static const uchar END_OF_LIST = 255;

struct VoronoiCell {
    Status status;
    uchar nb_t, nb_v;
    uchar3 tr[_MAX_T_];
    float4 clip[_MAX_P_];
    float4 voro_seed;
};

struct ConvexCell {
    __device__ ConvexCell(const int p_seed, const float* p_pts, const size_t pitch, Status* p_status);
    __device__ ConvexCell(
        const int p_seed, const float* p_pts, const size_t pitch, Status* p_status,
        const int tid, const float* vert, const size_t vert_pitch,
        const int* idx, const size_t idx_pitch
    );
    __device__ ConvexCell(const VoronoiCell& vc, Status* p_status);
    __device__ void clip_by_plane(int vid);
    __device__ void clip_by_plane(float4 eqn);
    __device__ float4 compute_triangle_point(uchar3 t, bool persp_divide=true) const;
    __device__ inline  uchar& ith_plane(uchar t, int i);
    __device__ int new_point(int vid);
    __device__ int new_point(float4 eqn);
    __device__ void new_triangle(uchar i, uchar j, uchar k);
    __device__ void compute_boundary();
    __device__ bool is_security_radius_reached(float4 last_neig);
    __device__ float4 point_from_index(int idx);
    
    __device__ bool triangle_is_in_conflict(uchar3 t, float4 eqn) const {
        return triangle_is_in_conflict_float(t, eqn);
    }

    __device__ bool triangle_is_in_conflict_float(uchar3 t, float4 eqn) const;
    __device__ bool triangle_is_in_conflict_double(uchar3 t, float4 eqn) const;

    
    Status* status;
    uchar nb_t;
    uchar nb_r;
    const float* pts = nullptr;
    const size_t pts_pitch = 0;
    int voro_id;
    float4 voro_seed;
    uchar nb_v;
    uchar first_boundary_;     
};

void compute_clipped_voro_diagram_GPU(
    const std::vector<float>& vertices,
    const std::vector<int>& indices,
    std::vector<float>& site,
    const int n_site, const bool site_is_transposed,
    int nb_Lloyd_iter = 1, int preferred_tet_k = 0
);

#endif // __VORONOI_H__

