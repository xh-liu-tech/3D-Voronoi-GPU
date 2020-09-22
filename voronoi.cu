#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

#include "stopwatch.h"
#include "knearests.h"
#include "voronoi.h"
#include "extern/kNN-CUDA/knncuda.h"

__device__ float4 minus4(float4 A, float4 B) {
    return make_float4(A.x-B.x, A.y-B.y, A.z-B.z, A.w-B.w);
}
__device__ float3 minus3(float3 A, float3 B) {
    return make_float3(A.x - B.x, A.y - B.y, A.z - B.z);
}
__device__ float4 plus4(float4 A, float4 B) {
    return make_float4(A.x+B.x, A.y+B.y, A.z+B.z, A.w+B.w);
}
__device__ float dot4(float4 A, float4 B) {
    return A.x*B.x + A.y*B.y + A.z*B.z + A.w*B.w;
}
__device__ float dot3(float4 A, float4 B) {
    return A.x*B.x + A.y*B.y + A.z*B.z;
}
__device__ float dot3(float3 A, float3 B) {
    return A.x * B.x + A.y * B.y + A.z * B.z;
}
__device__ float4 mul3(float s, float4 A) {
    return make_float4(s*A.x, s*A.y, s*A.z, 1.);
}
__device__ float4 cross3(float4 A, float4 B) {
    return make_float4(A.y*B.z - A.z*B.y, A.z*B.x - A.x*B.z, A.x*B.y - A.y*B.x, 0);
}
__device__ float3 cross3(float3 A, float3 B) {
    return make_float3(A.y * B.z - A.z * B.y, A.z * B.x - A.x * B.z, A.x * B.y - A.y * B.x);
}
__device__ float4 plane_from_point_and_normal(float4 P, float4 n) {
    return  make_float4(n.x, n.y, n.z, -dot3(P, n));
}
__device__ float4 plane_from_point_and_normal(float3 P, float3 n) {
    return  make_float4(n.x, n.y, n.z, -dot3(P, n));
}
__device__ inline float det2x2(float a11, float a12, float a21, float a22) {
    return a11*a22 - a12*a21;
}
__device__ inline float det3x3(float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33) {
    return a11*det2x2(a22, a23, a32, a33) - a21*det2x2(a12, a13, a32, a33) + a31*det2x2(a12, a13, a22, a23);
}

__device__ inline float det4x4(
    float a11, float a12, float a13, float a14,
    float a21, float a22, float a23, float a24,               
    float a31, float a32, float a33, float a34,  
    float a41, float a42, float a43, float a44  
) {
    float m12 = a21*a12 - a11*a22;
    float m13 = a31*a12 - a11*a32;
    float m14 = a41*a12 - a11*a42;
    float m23 = a31*a22 - a21*a32;
    float m24 = a41*a22 - a21*a42;
    float m34 = a41*a32 - a31*a42;
    
    float m123 = m23*a13 - m13*a23 + m12*a33;
    float m124 = m24*a13 - m14*a23 + m12*a43;
    float m134 = m34*a13 - m14*a33 + m13*a43;
    float m234 = m34*a23 - m24*a33 + m23*a43;
    
    return (m234*a14 - m134*a24 + m124*a34 - m123*a44);
}   

__device__ inline double det2x2(double a11, double a12, double a21, double a22) {
    return a11*a22 - a12*a21;
}

__device__ inline double det3x3(double a11, double a12, double a13, double a21, double a22, double a23, double a31, double a32, double a33) {
    return a11*det2x2(a22, a23, a32, a33) - a21*det2x2(a12, a13, a32, a33) + a31*det2x2(a12, a13, a22, a23);
}

__device__ inline double det4x4(
    double a11, double a12, double a13, double a14,
    double a21, double a22, double a23, double a24,               
    double a31, double a32, double a33, double a34,  
    double a41, double a42, double a43, double a44  
) {
    double m12 = a21*a12 - a11*a22;
    double m13 = a31*a12 - a11*a32;
    double m14 = a41*a12 - a11*a42;
    double m23 = a31*a22 - a21*a32;
    double m24 = a41*a22 - a21*a42;
    double m34 = a41*a32 - a31*a42;
    
    double m123 = m23*a13 - m13*a23 + m12*a33;
    double m124 = m24*a13 - m14*a23 + m12*a43;
    double m134 = m34*a13 - m14*a33 + m13*a43;
    double m234 = m34*a23 - m24*a33 + m23*a43;
    
    return (m234*a14 - m134*a24 + m124*a34 - m123*a44);
}   

__device__ inline float get_tet_volume(float4 A, float4 B, float4 C) {
    return -det3x3(A.x, A.y, A.z, B.x, B.y, B.z, C.x, C.y, C.z)/6.;
}
__device__ void get_tet_volume_and_barycenter(float4& bary, float& volume, float4 A, float4 B, float4 C, float4 D) {
    volume = get_tet_volume(minus4(A, D), minus4(B, D), minus4(C, D));
    bary = make_float4(.25f * (A.x + B.x + C.x + D.x), .25f * (A.y + B.y + C.y + D.y), .25f * (A.z + B.z + C.z + D.z), 1.0f);
}
__device__ float4 project_on_plane(float4 P, float4 plane) {
    float4 n = make_float4(plane.x, plane.y, plane.z, 0);
    float n_2 = dot4(n, n);
    float lambda = n_2 > 1e-2 ? (dot4(n, P) + plane.w) / n_2 : 0.0f;
    return plus4(P, mul3(-lambda, n));
}
template <typename T> __device__ void inline swap(T& a, T& b) { T c(a); a = b; b = c; }

__device__ float4 tri2plane(float3* vertices)
{
    float3 normal = cross3(minus3(vertices[1], vertices[0]), minus3(vertices[2], vertices[0]));
    return plane_from_point_and_normal(vertices[0], normal);
}

__device__ ConvexCell::ConvexCell(const int p_seed, const float* p_pts, const size_t pitch, Status* p_status) : pts_pitch(pitch), pts(p_pts)
{
    float eps = .1f;
    float xmin = -eps;
    float ymin = -eps;
    float zmin = -eps;
    float xmax = 1000 + eps;
    float ymax = 1000 + eps;
    float zmax = 1000 + eps;
    first_boundary_ = END_OF_LIST;
    FOR(i, _MAX_P_) boundary_next(i) = END_OF_LIST;
    voro_id = p_seed;
    if (pts_pitch)
        voro_seed = { pts[voro_id], pts[voro_id + pts_pitch], pts[voro_id + (pts_pitch << 1)], 1 };
    else
        voro_seed = { pts[3 * voro_id], pts[3 * voro_id + 1], pts[3 * voro_id + 2], 1 };
    status = p_status;
    *status = success;

    clip(0) = make_float4( 1.0,  0.0,  0.0, -xmin);
    clip(1) = make_float4(-1.0,  0.0,  0.0,  xmax);
    clip(2) = make_float4( 0.0,  1.0,  0.0, -ymin);
    clip(3) = make_float4( 0.0, -1.0,  0.0,  ymax);
    clip(4) = make_float4( 0.0,  0.0,  1.0, -zmin);
    clip(5) = make_float4( 0.0,  0.0, -1.0,  zmax);
    nb_v = 6;

    tr(0) = make_uchar3(2, 5, 0);
    tr(1) = make_uchar3(5, 3, 0);
    tr(2) = make_uchar3(1, 5, 2);
    tr(3) = make_uchar3(5, 1, 3);
    tr(4) = make_uchar3(4, 2, 0);
    tr(5) = make_uchar3(4, 0, 3);
    tr(6) = make_uchar3(2, 4, 1);
    tr(7) = make_uchar3(4, 3, 1);
    nb_t = 8;
}

__device__ ConvexCell::ConvexCell(
    const int p_seed, const float* p_pts, const size_t pitch, Status* p_status,
    const int tid, const float* vert, const size_t vert_pitch,
    const int* idx, const size_t idx_pitch
) : pts_pitch(pitch), pts(p_pts)
{
    first_boundary_ = END_OF_LIST;
    FOR(i, _MAX_P_) boundary_next(i) = END_OF_LIST;
    voro_id = p_seed;
    if (pts_pitch)
        voro_seed = { pts[voro_id], pts[voro_id + pts_pitch], pts[voro_id + (pts_pitch << 1)], 1 };
    else
        voro_seed = { pts[3 * voro_id], pts[3 * voro_id + 1], pts[3 * voro_id + 2], 1 };
    status = p_status;
    *status = success;

    int tet_sort_id[4][3] = { {2, 1, 3}, {0, 2, 3}, {1, 0, 3}, {0, 1, 2} }; // 4 faces of tet abcd: cbd acd bad abc

    FOR(i, 4)
    {
        float3 vertices[3];
        FOR(j, 3)
        {
            vertices[j] = {
                vert[idx[tid + (tet_sort_id[i][j] * idx_pitch)]],
                vert[idx[tid + (tet_sort_id[i][j] * idx_pitch)] + vert_pitch],
                vert[idx[tid + (tet_sort_id[i][j] * idx_pitch)] + (vert_pitch << 1)]
            };
        }
        clip(i) = tri2plane(vertices);
    }
    nb_v = 4;

    tr(0) = make_uchar3(1, 3, 2);
    tr(1) = make_uchar3(0, 1, 2);
    tr(2) = make_uchar3(0, 3, 1);
    tr(3) = make_uchar3(0, 2, 3);
    nb_t = 4;
}

__device__ ConvexCell::ConvexCell(const VoronoiCell& vc, Status* p_status)
{
    first_boundary_ = END_OF_LIST;
    FOR(i, _MAX_P_) boundary_next(i) = END_OF_LIST;
    voro_id = -1;
    voro_seed = vc.voro_seed;
    nb_v = vc.nb_v;
    nb_t = vc.nb_t;
    FOR(i, nb_v)
        clip(i) = vc.clip[i];
    FOR(i, nb_t)
        tr(i) = vc.tr[i];
    status = p_status;
    *status = vc.status;
}

__device__  bool ConvexCell::is_security_radius_reached(float4 last_neig) {
    // finds furthest voro vertex distance2
    float v_dist = 0;
    FOR(i, nb_t) {
        float4 pc = compute_triangle_point(tr(i));
        float4 diff = minus4(pc, voro_seed);
        float d2 = dot3(diff, diff); // TODO safe to put dot4 here, diff.w = 0
        v_dist = max(d2, v_dist);
    }
    //compare to new neighbors distance2
    float4 diff = minus4(last_neig, voro_seed); // TODO it really should take index of the neighbor instead of the float4, then would be safe to put dot4
    float d2 = dot3(diff, diff);
    return (d2 > 4*v_dist);
}

__device__ inline  uchar& ConvexCell::ith_plane(uchar t, int i) {
    return reinterpret_cast<uchar *>(&(tr(t)))[i];
}

__device__ float4 ConvexCell::compute_triangle_point(uchar3 t, bool persp_divide) const {
    float4 pi1 = clip(t.x);
    float4 pi2 = clip(t.y);
    float4 pi3 = clip(t.z);
    float4 result;
    result.x = -det3x3(pi1.w, pi1.y, pi1.z, pi2.w, pi2.y, pi2.z, pi3.w, pi3.y, pi3.z);
    result.y = -det3x3(pi1.x, pi1.w, pi1.z, pi2.x, pi2.w, pi2.z, pi3.x, pi3.w, pi3.z);
    result.z = -det3x3(pi1.x, pi1.y, pi1.w, pi2.x, pi2.y, pi2.w, pi3.x, pi3.y, pi3.w);
    result.w =  det3x3(pi1.x, pi1.y, pi1.z, pi2.x, pi2.y, pi2.z, pi3.x, pi3.y, pi3.z);
    if (persp_divide) return make_float4(result.x / result.w, result.y / result.w, result.z / result.w, 1);
    return result;
}

inline __device__ float max4(float a, float b, float c, float d) {
    return fmaxf(fmaxf(a,b),fmaxf(c,d));
}

inline __device__ void get_minmax3(
    float& m, float& M, float x1, float x2, float x3
) {
    m = fminf(fminf(x1,x2), x3);
    M = fmaxf(fmaxf(x1,x2), x3);
}

inline __device__ double max4(double a, double b, double c, double d) {
    return fmax(fmax(a,b),fmax(c,d));
}

inline __device__ void get_minmax3(
    double& m, double& M, double x1, double x2, double x3
) {
    m = fmin(fmin(x1,x2), x3);
    M = fmax(fmax(x1,x2), x3);
}


__device__ bool ConvexCell::triangle_is_in_conflict_float(uchar3 t, float4 eqn) const {
    float4 pi1 = clip(t.x);
    float4 pi2 = clip(t.y);
    float4 pi3 = clip(t.z);
    float det = det4x4(
	pi1.x, pi2.x, pi3.x, eqn.x,
	pi1.y, pi2.y, pi3.y, eqn.y,
	pi1.z, pi2.z, pi3.z, eqn.z,
	pi1.w, pi2.w, pi3.w, eqn.w
    );

#ifdef USE_ARITHMETIC_FILTER
    float maxx = max4(fabsf(pi1.x), fabsf(pi2.x), fabsf(pi3.x), fabsf(eqn.x));
    float maxy = max4(fabsf(pi1.y), fabsf(pi2.y), fabsf(pi3.y), fabsf(eqn.y));    
    float maxz = max4(fabsf(pi1.z), fabsf(pi2.z), fabsf(pi3.z), fabsf(eqn.z));    

    // The constant is computed by the program 
    // in predicate_generator/
    float eps = 6.6876506e-05 * maxx * maxy * maxz;
    
    float min_max;
    float max_max;
    get_minmax3(min_max, max_max, maxx, maxy, maxz);

    eps *= (max_max * max_max);

    if(fabsf(det) < eps) {
	*status = needs_exact_predicates;
    }
#endif

    return (det > 0.0f);
}

__device__ bool ConvexCell::triangle_is_in_conflict_double(uchar3 t, float4 eqn_f) const {
    float4 pi1_f = clip(t.x);
    float4 pi2_f = clip(t.y);
    float4 pi3_f = clip(t.z);

    double4 eqn = make_double4(eqn_f.x, eqn_f.y, eqn_f.z, eqn_f.w);
    double4 pi1 = make_double4(pi1_f.x, pi1_f.y, pi1_f.z, pi1_f.w);
    double4 pi2 = make_double4(pi2_f.x, pi2_f.y, pi2_f.z, pi2_f.w);
    double4 pi3 = make_double4(pi3_f.x, pi3_f.y, pi3_f.z, pi3_f.w);        
    
    double det = det4x4(
	pi1.x, pi2.x, pi3.x, eqn.x,
	pi1.y, pi2.y, pi3.y, eqn.y,
	pi1.z, pi2.z, pi3.z, eqn.z,
	pi1.w, pi2.w, pi3.w, eqn.w
    );

#ifdef USE_ARITHMETIC_FILTER
    double maxx = max4(fabs(pi1.x), fabs(pi2.x), fabs(pi3.x), fabs(eqn.x));
    double maxy = max4(fabs(pi1.y), fabs(pi2.y), fabs(pi3.y), fabs(eqn.y));    
    double maxz = max4(fabs(pi1.z), fabs(pi2.z), fabs(pi3.z), fabs(eqn.z));    

    // The constant is computed by the program 
    // in predicate_generator/
    double eps = 1.2466136531027298e-13 * maxx * maxy * maxz;
    
    double min_max;
    double max_max;
    get_minmax3(min_max, max_max, maxx, maxy, maxz);

    eps *= (max_max * max_max);

    if(fabs(det) < eps) {
	*status = needs_exact_predicates;
    }
#endif    
    
    return (det > 0.0f);
}

__device__ float4 ConvexCell::point_from_index(int idx)
{
    if (pts_pitch)
        return { pts[idx], pts[idx + pts_pitch], pts[idx + (pts_pitch << 1)], 1 };
    else
        return { pts[3 * idx], pts[3 * idx + 1], pts[3 * idx + 2], 1 };
}

__device__ void ConvexCell::new_triangle(uchar i, uchar j, uchar k) {
    if (nb_t+1 >= _MAX_T_) {
        *status = triangle_overflow;
        return; 
    }
    tr(nb_t) = make_uchar3(i, j, k);
    nb_t++;
}

__device__ int ConvexCell::new_point(int vid) {
    if (nb_v >= _MAX_P_) { 
        *status = vertex_overflow; 
        return -1; 
    }

    float4 B = point_from_index(vid);
    float4 dir = minus4(voro_seed, B);
    float4 ave2 = plus4(voro_seed, B);
    float dot = dot3(ave2,dir); // TODO safe to put dot4 here, dir.w = 0
    clip(nb_v) = make_float4(dir.x, dir.y, dir.z, -dot / 2.f);
    nb_v++;
    return nb_v - 1;
}

__device__ int ConvexCell::new_point(float4 eqn)
{
    if (nb_v >= _MAX_P_) {
        *status = vertex_overflow;
        return -1;
    }
    clip(nb_v) = eqn;
    nb_v++;
    return nb_v - 1;
}

__device__ void ConvexCell::compute_boundary() {
    // clean circular list of the boundary
    FOR(i, _MAX_P_) boundary_next(i) = END_OF_LIST;
    first_boundary_ = END_OF_LIST;

    int nb_iter = 0;
    uchar t = nb_t;
    while (nb_r>0) {
        if (nb_iter++>65535) {
            *status = inconsistent_boundary;
            return;
        }
        bool is_in_border[3];
        bool next_is_opp[3];
        FOR(e, 3)   is_in_border[e] = (boundary_next(ith_plane(t, e)) != END_OF_LIST);
        FOR(e, 3)   next_is_opp[e] = (boundary_next(ith_plane(t, (e + 1) % 3)) == ith_plane(t, e));

        bool new_border_is_simple = true;
        // check for non manifoldness
        FOR(e, 3) if (!next_is_opp[e] && !next_is_opp[(e + 1) % 3] && is_in_border[(e + 1) % 3]) new_border_is_simple = false;

        // check for more than one boundary ... or first triangle
        if (!next_is_opp[0] && !next_is_opp[1] && !next_is_opp[2]) {
            if (first_boundary_ == END_OF_LIST) {
                FOR(e, 3) boundary_next(ith_plane(t, e)) = ith_plane(t, (e + 1) % 3);
                first_boundary_ = tr(t).x;
            }
            else new_border_is_simple = false;
        }

        if (!new_border_is_simple) {
            t++;
            if (t == nb_t + nb_r) t = nb_t;
            continue;
        }

        // link next
        FOR(e, 3) if (!next_is_opp[e]) boundary_next(ith_plane(t, e)) = ith_plane(t, (e + 1) % 3);

        // destroy link from removed vertices
        FOR(e, 3)  if (next_is_opp[e] && next_is_opp[(e + 1) % 3]) {
            if (first_boundary_ == ith_plane(t, (e + 1) % 3)) first_boundary_ = boundary_next(ith_plane(t, (e + 1) % 3));
            boundary_next(ith_plane(t, (e + 1) % 3)) = END_OF_LIST;
        }

        //remove triangle from R, and restart iterating on R
        swap(tr(t), tr(nb_t+nb_r-1));
        t = nb_t;
        nb_r--;
    }
}

__device__ void  ConvexCell::clip_by_plane(int vid) {
    int cur_v = new_point(vid); // add new plane equation
    if (*status == vertex_overflow) return;
    float4 eqn = clip(cur_v);
    nb_r = 0;

    int i = 0;
    while (i < nb_t) { // for all vertices of the cell
	if(triangle_is_in_conflict(tr(i), eqn)) {
            nb_t--;
            swap(tr(i), tr(nb_t));
            nb_r++;
        }
        else i++;
    }

    if (*status == needs_exact_predicates) {
	return;
    }
    
    if (nb_r == 0) { // if no clips, then remove the plane equation
        nb_v--;
        return;
    }

    if (nb_t == 0)
    {
        *status = no_intersection;
        return;
    }

    // Step 2: compute cavity boundary
    compute_boundary();
    if (*status != success) return;
    if (first_boundary_ == END_OF_LIST) return;

    // Step 3: Triangulate cavity
    uchar cir = first_boundary_;
    do {
        new_triangle(cur_v, cir, boundary_next(cir));
        if (*status != success) return;
        cir = boundary_next(cir);
    } while (cir != first_boundary_);
}

__device__ void ConvexCell::clip_by_plane(float4 eqn)
{
    int cur_v = new_point(eqn);
    if (*status == vertex_overflow) return;
    nb_r = 0;

    int i = 0;
    while (i < nb_t) { // for all vertices of the cell
        if (triangle_is_in_conflict(tr(i), eqn)) {
            nb_t--;
            swap(tr(i), tr(nb_t));
            nb_r++;
        }
        else i++;
    }

    if (*status == needs_exact_predicates) {
        return;
    }

    if (nb_r == 0) { // if no clips, then remove the plane equation
        nb_v--;
        return;
    }

    if (nb_t == 0)
    {
        *status = no_intersection;
        return;
    }
    // Step 2: compute cavity boundary
    compute_boundary();
    if (*status != success && *status != security_radius_not_reached) return;
    if (first_boundary_ == END_OF_LIST) return;

    // Step 3: Triangulate cavity
    uchar cir = first_boundary_;
    do {
        new_triangle(cur_v, cir, boundary_next(cir));
        if (*status != success && *status != security_radius_not_reached) return;
        cir = boundary_next(cir);
    } while (cir != first_boundary_);
}

__device__ void get_tet_decomposition_of_vertex(ConvexCell& cc, int t, float4* P) {
    float4 C = cc.voro_seed;
    float4 A = cc.compute_triangle_point(tr(t));
    FOR(i,3)  P[2*i  ] = project_on_plane(C, clip(cc.ith_plane(t,i)));
    FOR(i, 3) P[2*i+1] = project_on_plane(A, plane_from_point_and_normal(C, cross3(minus4(P[2*i], C), minus4(P[(2*(i+1))%6], C))));
}

//###################  KERNEL   ######################
__device__ void compute_voro_cell(
    const float* site, const size_t site_pitch,
    const int* site_knn, const size_t site_knn_pitch,
    int seed, VoronoiCell& vc
)
{
    //create BBox
    ConvexCell cc(seed, site, site_pitch, &vc.status);

    for (int v = 1 - (site_pitch == 0); v <= _K_ - (site_pitch == 0); ++v)
    {
        int idx = site_pitch ? seed + v * site_knn_pitch : _K_ * seed + v;
        int z = site_knn[idx];
        
        cc.clip_by_plane(z);
        if (cc.is_security_radius_reached(cc.point_from_index(z))) {
            break;
        }
        if (vc.status != success) {
            return;
        }
    }
    // check security radius
    int last_idx = site_pitch ? seed + _K_ * site_knn_pitch : _K_ * (seed + 1) - 1;
    if (!cc.is_security_radius_reached(cc.point_from_index(site_knn[last_idx]))) {
        vc.status = security_radius_not_reached;
    }

    vc.nb_t = cc.nb_t;
    vc.nb_v = cc.nb_v;
    FOR(i, cc.nb_t)
        vc.tr[i] = tr(i);
    FOR(i, cc.nb_v)
        vc.clip[i] = clip(i);
    vc.voro_seed = cc.voro_seed;
}

__device__ void atomic_add_bary_and_volume(
    ConvexCell& cc, int seed,
    float* cell_bary_sum, const size_t cell_bary_sum_pitch, float* cell_vol
)
{
    float4 tet_bary;
    float tet_vol;
    float4 bary_sum = { 0.0f, 0.0f, 0.0f, 0.0f };
    float cur_cell_vol = 0;
    float4 P[6];
    float4 C = cc.voro_seed;

    FOR(t, cc.nb_t) {
        float4 A = cc.compute_triangle_point(tr(t));
        get_tet_decomposition_of_vertex(cc, t, P);
        FOR(i, 6) {
            get_tet_volume_and_barycenter(tet_bary, tet_vol, P[i], P[(i + 1) % 6], C, A);
            bary_sum = plus4(bary_sum, mul3(tet_vol, tet_bary));
            cur_cell_vol += tet_vol;
        }
    }

    if (abs(cur_cell_vol) < 0.1)
    {
        *cc.status = no_intersection;
        return;
    }

    if (cell_bary_sum_pitch)
    {
        atomicAdd(cell_bary_sum + seed, bary_sum.x);
        atomicAdd(cell_bary_sum + seed + cell_bary_sum_pitch, bary_sum.y);
        atomicAdd(cell_bary_sum + seed + (cell_bary_sum_pitch << 1), bary_sum.z);
    }
    else
    {
        atomicAdd(cell_bary_sum + 3 * seed, bary_sum.x);
        atomicAdd(cell_bary_sum + 3 * seed + 1, bary_sum.y);
        atomicAdd(cell_bary_sum + 3 * seed + 2, bary_sum.z);
    }
    atomicAdd(cell_vol + seed, cur_cell_vol);
}

__device__ void clip_voro_cell(
    ConvexCell& cc, int tid,
    float* vert, size_t vert_pitch,
    int* idx, size_t idx_pitch
)
{
    int tet_sort_id[4][3] = { {2, 1, 3}, {0, 2, 3}, {1, 0, 3}, {0, 1, 2} }; // 4 faces of tet abcd: cbd acd bad abc

    FOR(i, 4)
    {
        float3 vertices[3];
        FOR(j, 3)
        {
            vertices[j] = {
                vert[idx[tid + (tet_sort_id[i][j] * idx_pitch)]],
                vert[idx[tid + (tet_sort_id[i][j] * idx_pitch)] + vert_pitch],
                vert[idx[tid + (tet_sort_id[i][j] * idx_pitch)] + (vert_pitch << 1)]
            };
        }
        
        cc.clip_by_plane(tri2plane(vertices));

        if (*cc.status == no_intersection)
            return;
    }
}

//----------------------------------KERNEL
__global__ void voro_cell_test_GPU_param(
    const float* site, const int n_site, const size_t site_pitch,
    const int* site_knn, const size_t site_knn_pitch,
    VoronoiCell* voronoi_cells
) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= n_site) return;
    compute_voro_cell(
        site, site_pitch,
        site_knn, site_knn_pitch,
        seed, voronoi_cells[seed]
    );
}

__global__ void clipped_voro_cell_test_GPU_param_tet(
    const float* site, const int n_site, const size_t site_pitch,
    const int* site_knn, const size_t site_knn_pitch,
    const float* vert, const int n_vert, const size_t vert_pitch,
    const int* idx, const int n_tet, const size_t idx_pitch,
    const int* tet_knn, const size_t tet_knn_pitch, const int tet_k,
    Status* gpu_stat, VoronoiCell* voronoi_cells,
    float* cell_bary_sum, const size_t cell_bary_sum_pitch, float* cell_vol
)
{
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = thread / tet_k; // tet index
    if (tid >= n_tet) return;

    int seed = tet_knn[(thread % tet_k) * tet_knn_pitch + tid];
    ConvexCell cc(seed, site, site_pitch, &(gpu_stat[thread]), tid, vert, vert_pitch, idx, idx_pitch);

    for (int v = 1 - (site_pitch == 0); v <= _K_ - (site_pitch == 0); ++v)
    {
        int idx = site_pitch ? seed + v * site_knn_pitch : _K_ * seed + v;
        int z = site_knn[idx];

        cc.clip_by_plane(z);

        if (cc.is_security_radius_reached(cc.point_from_index(z))) {
            break;
        }
        if (*cc.status != success) {
            return;
        }
    }
    // check security radius
    int last_idx = site_pitch ? seed + _K_ * site_knn_pitch : _K_ * (seed + 1) - 1;
    if (!cc.is_security_radius_reached(cc.point_from_index(site_knn[last_idx]))) {
        *cc.status = security_radius_not_reached;
    }

    if (*cc.status != no_intersection)
        atomic_add_bary_and_volume(cc, seed, cell_bary_sum, cell_bary_sum_pitch, cell_vol);
}

__global__ void clipped_voro_cell_test_GPU_param(
    float* vert, int n_vert, size_t vert_pitch,
    int* idx, int n_tet, size_t idx_pitch,
    int* tet_knn, size_t tet_knn_pitch, int tet_k,
    Status* gpu_stat, VoronoiCell* voronoi_cells,
    float* cell_bary_sum, const size_t cell_bary_sum_pitch, float* cell_vol
)
{
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = thread / tet_k; // tet index
    if (tid >= n_tet) return;

    int seed = tet_knn[(thread % tet_k) * tet_knn_pitch + tid];

    ConvexCell cc(voronoi_cells[seed], &(gpu_stat[thread]));

    clip_voro_cell(
        cc, tid, vert, vert_pitch,
        idx, idx_pitch
    );

    if (*cc.status != no_intersection)
        atomic_add_bary_and_volume(cc, seed, cell_bary_sum, cell_bary_sum_pitch, cell_vol);
}

__global__ void compute_new_site(float* site, int n_site, size_t site_pitch, float* cell_bary_sum, const size_t cell_bary_sum_pitch, float* cell_vol)
{
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= n_site)
        return;

    if (cell_vol[seed] != 0)
    {
        if (site_pitch)
        {
            site[seed] = cell_bary_sum[seed] / cell_vol[seed];
            site[seed + site_pitch] = cell_bary_sum[seed + cell_bary_sum_pitch] / cell_vol[seed];
            site[seed + (site_pitch << 1)] = cell_bary_sum[seed + (cell_bary_sum_pitch << 1)] / cell_vol[seed];
        }
        else
        {
            site[3 * seed] = cell_bary_sum[3 * seed] / cell_vol[seed];
            site[3 * seed + 1] = cell_bary_sum[3 * seed + 1] / cell_vol[seed];
            site[3 * seed + 2] = cell_bary_sum[3 * seed + 2] / cell_vol[seed];
        }
    }
}

__global__ void compute_tet_centroid(
    const float* vert,
    const int n_vert,
    const size_t vert_pitch,
    const int* idx,
    const int n_tet,
    const size_t idx_pitch,
    float* tet_centroid,
    const size_t tet_centroid_pitch
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_tet) return;

    float3 centroid = { 0.0f, 0.0f, 0.0f };

    FOR(i, 4)
    {
        centroid.x += vert[idx[tid + i * idx_pitch]];
        centroid.y += vert[idx[tid + i * idx_pitch] + vert_pitch];
        centroid.z += vert[idx[tid + i * idx_pitch] + (vert_pitch << 1)];
    }

    centroid.x *= 0.25f; centroid.y *= 0.25f; centroid.z *= 0.25f;

    tet_centroid[tid] = centroid.x;
    tet_centroid[tid + tet_centroid_pitch] = centroid.y;
    tet_centroid[tid + (tet_centroid_pitch << 1)] = centroid.z;
}

__global__ void transpose_site(const float* site, const int n_site, float* site_transposed, const size_t site_transposed_pitch)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    site_transposed[idx] = site[3 * idx];
    site_transposed[idx + site_transposed_pitch] = site[3 * idx + 1];
    site_transposed[idx + (site_transposed_pitch << 1)] = site[3 * idx + 2];
}

//----------------------------------WRAPPER
template <class T> struct GPUBuffer {
    void init(T* data) {
        IF_VERBOSE(std::cerr << "GPU: " << size * sizeof(T)/1048576 << " Mb used" << std::endl);
        cpu_data = data;
        cuda_check(cudaMalloc((void**)& gpu_data, size * sizeof(T)));
        cpu2gpu();
    }
    GPUBuffer(std::vector<T>& v) {size = v.size() ;init(v.data());}
    ~GPUBuffer() { cuda_check(cudaFree(gpu_data)); }

    void cpu2gpu() { cuda_check(cudaMemcpy(gpu_data, cpu_data, size * sizeof(T), cudaMemcpyHostToDevice)); }
    void gpu2cpu() { cuda_check(cudaMemcpy(cpu_data, gpu_data, size * sizeof(T), cudaMemcpyDeviceToHost)); }

    T* cpu_data;
    T* gpu_data;
    int size;
};

char StatusStr[7][128] = {
    "triangle_overflow","vertex_overflow","inconsistent_boundary","security_radius_not_reached","success", "needs_exact_predicates", "no_intersection"
};

void show_status_stats(std::vector<Status> &stat) {
    IF_VERBOSE(std::cerr << " \n\n\n---------Summary of success/failure------------\n");
    std::vector<int> nb_statuss(7, 0);
    FOR(i, stat.size()) nb_statuss[stat[i]]++;
    IF_VERBOSE(FOR(r, 7) std::cerr << " " << StatusStr[r] << "   " << nb_statuss[r] << "\n";)
        std::cerr << " " << StatusStr[4] << "   " << nb_statuss[4] << " /  " << stat.size() << "\n";
}

void cuda_check_error() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "Failed (1) (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
}

void compute_tet_knn_dev(
    const float* vert_dev,
    const int n_vert,
    const size_t vert_pitch,
    const int* idx_dev,
    const int n_tet,
    const size_t idx_pitch,
    const float* site_dev,
    const int n_site,
    const size_t site_pitch,
    int* tet_knn_dev,
    const size_t tet_knn_pitch,
    const int tet_k
)
{
    float* tet_centroid_dev = nullptr;
    size_t tet_centroid_pitch_in_bytes;
    cudaMallocPitch((void**)& tet_centroid_dev, &tet_centroid_pitch_in_bytes, n_tet * sizeof(float), 3);
    cuda_check_error();
    size_t tet_centroid_pitch = tet_centroid_pitch_in_bytes / sizeof(float);

    compute_tet_centroid<<<n_tet / VORO_BLOCK_SIZE + 1, VORO_BLOCK_SIZE>>>(
        vert_dev, n_vert, vert_pitch,
        idx_dev, n_tet, idx_pitch,
        tet_centroid_dev, tet_centroid_pitch
    );

    if (site_pitch) // has been transposed
        knn_cuda_global_dev(
            site_dev, n_site, site_pitch,
            tet_centroid_dev, n_tet, tet_centroid_pitch,
            3, tet_k,
            tet_knn_dev, tet_knn_pitch
        );
    else
    {
        float* site_transposed_dev = nullptr;
        size_t site_transposed_pitch_in_bytes;
        cudaMallocPitch((void**)& site_transposed_dev, &site_transposed_pitch_in_bytes, n_site * sizeof(float), 3);
        cuda_check_error();
        size_t site_transposed_pitch = site_transposed_pitch_in_bytes / sizeof(float);
        transpose_site<<<n_site / KNN_BLOCK_SIZE + 1, KNN_BLOCK_SIZE>>>(site_dev, n_site, site_transposed_dev, site_transposed_pitch);

        knn_cuda_global_dev(
            site_transposed_dev, n_site, site_transposed_pitch,
            tet_centroid_dev, n_tet, tet_centroid_pitch,
            3, tet_k,
            tet_knn_dev, tet_knn_pitch
        );

        cudaFree(site_transposed_dev);
    }

    cudaFree(tet_centroid_dev);
}

void copy_tet_data(
    const std::vector<float>& vertices,
    const std::vector<int>& indices,
    float*& vert_dev,
    size_t& vert_pitch,
    int*& idx_dev,
    size_t& idx_pitch
)
{
    size_t n_vert = vertices.size() / 3, n_tet = (indices.size() >> 2);
    
    // transpose
    float* vert_T = new float[3 * n_vert];
    int* idx_T = new int[n_tet << 2];
    FOR(i, n_vert)
    {
        vert_T[i] = vertices[3 * i];
        vert_T[i + n_vert] = vertices[3 * i + 1];
        vert_T[i + (n_vert << 1)] = vertices[3 * i + 2];
    }
    FOR(i, n_tet)
    {
        idx_T[i] = indices[(i << 2)];
        idx_T[i + n_tet] = indices[(i << 2) + 1];
        idx_T[i + (n_tet << 1)] = indices[(i << 2) + 2];
        idx_T[i + n_tet * 3] = indices[(i << 2) + 3];
    }

    size_t vert_pitch_in_bytes, idx_pitch_in_bytes;
    cudaMallocPitch((void**)& vert_dev, &vert_pitch_in_bytes, n_vert * sizeof(float), 3);
    cuda_check_error();
    cudaMallocPitch((void**)& idx_dev, &idx_pitch_in_bytes, n_tet * sizeof(int), 4);
    cuda_check_error();
    vert_pitch = vert_pitch_in_bytes / sizeof(float);
    idx_pitch = idx_pitch_in_bytes / sizeof(int);
    cudaMemcpy2D(vert_dev, vert_pitch_in_bytes, vert_T, n_vert * sizeof(float), n_vert * sizeof(float), 3, cudaMemcpyHostToDevice);
    cuda_check_error();
    cudaMemcpy2D(idx_dev, idx_pitch_in_bytes, idx_T, n_tet * sizeof(int), n_tet * sizeof(int), 4, cudaMemcpyHostToDevice);
    cuda_check_error();

    delete[] vert_T;
    delete[] idx_T;
}

void compute_clipped_voro_diagram_GPU(
    const std::vector<float>& vertices,
    const std::vector<int>& indices,
    std::vector<float>& site,
    const int n_site, const bool site_is_transposed,
    int nb_Lloyd_iter, int preferred_tet_k
)
{
    cudaSetDevice(0); // specify a device to be used for GPU computation

    int n_vert = vertices.size() / 3, n_tet = (indices.size() >> 2);
    int tet_k = preferred_tet_k ? preferred_tet_k : max(10 * (int)ceil(8.0 * log10(7.0 * n_site / n_tet + 1.0)), 20);

    std::vector<Status> stat(n_tet * tet_k, security_radius_not_reached);

    kn_problem* kn = NULL;

    // copy tet vertices and indices to device
    float* vert_dev = nullptr;
    int* idx_dev = nullptr;
    size_t vert_pitch, idx_pitch;
    copy_tet_data(vertices, indices, vert_dev, vert_pitch, idx_dev, idx_pitch);
    
    // allocate memory for tet knn
    int* tet_knn_dev = nullptr;
    size_t tet_knn_pitch_in_bytes;
    cudaMallocPitch((void**)& tet_knn_dev, &tet_knn_pitch_in_bytes, n_tet * sizeof(int), tet_k);
    cuda_check_error();
    size_t tet_knn_pitch = tet_knn_pitch_in_bytes / sizeof(int);

    // allocate memory for voronoi cell
    VoronoiCell* voronoi_cells_dev = nullptr;
    cudaMalloc((void**)& voronoi_cells_dev, n_site * sizeof(VoronoiCell));
    cuda_check_error();

    // allocate memory for output points
    float* cell_bary_sum_dev = nullptr;
    size_t cell_bary_sum_pitch_in_bytes = 0, cell_bary_sum_pitch = 0;
    if (site_is_transposed)
    {
        cudaMallocPitch((void**)& cell_bary_sum_dev, &cell_bary_sum_pitch_in_bytes, n_site * sizeof(float), 3);
        cell_bary_sum_pitch = cell_bary_sum_pitch_in_bytes / sizeof(float);
    }
    else
        cudaMalloc((void**)& cell_bary_sum_dev, 3 * n_site * sizeof(float));
    cuda_check_error();

    GPUBuffer<Status> gpu_stat(stat);

    // allocate memory for cell volume
    float* cell_vol_dev = nullptr;
    cudaMalloc((void**)& cell_vol_dev, n_site * sizeof(float));
    cuda_check_error();

    // allocate memory for site and site knn
    float* site_transposed_dev = nullptr;
    int* site_knn_dev = nullptr;
    size_t site_pitch = 0, site_pitch_in_bytes = 0, site_knn_pitch = 0, site_knn_pitch_in_bytes = 0;

    std::ofstream record("record.csv", std::ios::app);
    record << "n_site, n_tet, TET_K, SITE_K, Iter, k-NN, Clip_tet-cell, Clip_cell-tet, Compute_new_site, GPU2CPU\n";
    record << std::setprecision(5) << std::setiosflags(std::ios::fixed);

    FOR(i, nb_Lloyd_iter)
    {
        printf("--------------------Iter %d--------------------\n", i);
        record << n_site << ", " << n_tet << ", " << tet_k << ", " << _K_ << ", " << i << ", ";
        Stopwatch sw("Iteration");
        double start_time = 0.0, stop_time = 0.0;
        start_time = sw.now();
        { // GPU KNN
            if (site_is_transposed)
            {
                // allocate memory for site and site knn
                cudaMallocPitch((void**)& site_knn_dev, &site_knn_pitch_in_bytes, n_site * sizeof(int), _K_ + 1);
                cudaMallocPitch((void**)& site_transposed_dev, &site_pitch_in_bytes, n_site * sizeof(float), 3);
                cuda_check_error();
                site_knn_pitch = site_knn_pitch_in_bytes / sizeof(int);

                // copy sites to device
                cudaMemcpy2D(site_transposed_dev, site_pitch_in_bytes, site.data(), n_site * sizeof(float), n_site * sizeof(float), 3, cudaMemcpyHostToDevice);
                cuda_check_error();
                
                site_pitch = site_pitch_in_bytes / sizeof(float);

                knn_cuda_global_dev(site_transposed_dev, n_site, site_pitch, site_transposed_dev, n_site, site_pitch, 3, _K_ + 1, site_knn_dev, site_knn_pitch);
                cuda_check_error();
            }
            else
            {
                kn = kn_prepare((float3*)site.data(), n_site);
                cudaMemcpy(site.data(), kn->d_stored_points, kn->allocated_points * sizeof(float) * 3, cudaMemcpyDeviceToHost);
                cuda_check_error();
                kn_solve(kn);
            }
        } // GPU KNN
    
        {
            if (site_is_transposed)
                compute_tet_knn_dev(vert_dev, n_vert, vert_pitch, idx_dev, n_tet, idx_pitch, site_transposed_dev, n_site, site_pitch, tet_knn_dev, tet_knn_pitch, tet_k);
            else
                compute_tet_knn_dev(vert_dev, n_vert, vert_pitch, idx_dev, n_tet, idx_pitch, (float*)kn->d_stored_points, n_site, site_pitch, tet_knn_dev, tet_knn_pitch, tet_k);
            cuda_check_error();
        }

        stop_time = sw.now(); // GPU KNN Both
        record << stop_time - start_time << ", ";
        start_time = sw.now();

        {
            if (site_is_transposed)
                cudaMemset2D(cell_bary_sum_dev, cell_bary_sum_pitch_in_bytes, 0, n_site * sizeof(float), 3);
            else
                cudaMemset(cell_bary_sum_dev, 0, 3 * n_site * sizeof(float));
            cudaMemset(cell_vol_dev, 0, n_site * sizeof(float));
            cuda_check_error();
        }
        
        { // GPU voro kernel only
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            if (site_is_transposed)
                clipped_voro_cell_test_GPU_param_tet<<<n_tet * tet_k / VORO_BLOCK_SIZE + 1, VORO_BLOCK_SIZE>>>(
                    site_transposed_dev, n_site, site_pitch,
                    site_knn_dev, site_knn_pitch,
                    vert_dev, n_vert, vert_pitch,
                    idx_dev, n_tet, idx_pitch,
                    tet_knn_dev, tet_knn_pitch, tet_k,
                    gpu_stat.gpu_data, voronoi_cells_dev,
                    cell_bary_sum_dev, cell_bary_sum_pitch, cell_vol_dev
                );
            else
                clipped_voro_cell_test_GPU_param_tet<<<n_tet * tet_k / VORO_BLOCK_SIZE + 1, VORO_BLOCK_SIZE>>>(
                    (float*)kn->d_stored_points, n_site, site_pitch,
                    (int*)kn->d_knearests, site_knn_pitch,
                    vert_dev, n_vert, vert_pitch,
                    idx_dev, n_tet, idx_pitch,
                    tet_knn_dev, tet_knn_pitch, tet_k,
                    gpu_stat.gpu_data, voronoi_cells_dev,
                    cell_bary_sum_dev, cell_bary_sum_pitch, cell_vol_dev
                );
            cuda_check_error();

            cudaEventRecord(stop);
            cudaEventSynchronize(start);
            cudaEventSynchronize(stop);

            stop_time = sw.now(); // Clip tet
            record << stop_time - start_time << ", ";

            start_time = sw.now();
            cudaEventRecord(start);

            {
                if (site_is_transposed)
                    cudaMemset2D(cell_bary_sum_dev, cell_bary_sum_pitch_in_bytes, 0, n_site * sizeof(float), 3);
                else
                    cudaMemset(cell_bary_sum_dev, 0, 3 * n_site * sizeof(float));
                cudaMemset(cell_vol_dev, 0, n_site * sizeof(float));
                cuda_check_error();
            }

            if (site_is_transposed)
                voro_cell_test_GPU_param<<<n_site / VORO_BLOCK_SIZE + 1, VORO_BLOCK_SIZE>>>(
                    site_transposed_dev, n_site, site_pitch,
                    site_knn_dev, site_knn_pitch, voronoi_cells_dev
                );
            else
                voro_cell_test_GPU_param<<<n_site / VORO_BLOCK_SIZE + 1, VORO_BLOCK_SIZE>>>(
                    (float*)kn->d_stored_points, n_site, site_pitch,
                    (int*)kn->d_knearests, site_knn_pitch, voronoi_cells_dev
                );
            cuda_check_error();

            clipped_voro_cell_test_GPU_param<<<n_tet * tet_k / VORO_BLOCK_SIZE + 1, VORO_BLOCK_SIZE>>>(
                vert_dev, n_vert, vert_pitch,
                idx_dev, n_tet, idx_pitch,
                tet_knn_dev, tet_knn_pitch, tet_k,
                gpu_stat.gpu_data, voronoi_cells_dev,
                cell_bary_sum_dev, cell_bary_sum_pitch, cell_vol_dev
            );
            cuda_check_error();

            cudaEventRecord(stop);
            cudaEventSynchronize(start);
            cudaEventSynchronize(stop);

            stop_time = sw.now(); // Clip cell
            record << stop_time - start_time << ", ";

            start_time = sw.now();
            cudaEventRecord(start);

            if (site_is_transposed)
                compute_new_site<<<n_site / VORO_BLOCK_SIZE + 1, VORO_BLOCK_SIZE>>>(site_transposed_dev, n_site, site_pitch, cell_bary_sum_dev, cell_bary_sum_pitch, cell_vol_dev);
            else
                compute_new_site<<<n_site / VORO_BLOCK_SIZE + 1, VORO_BLOCK_SIZE>>>((float*)kn->d_stored_points, n_site, site_pitch, cell_bary_sum_dev, cell_bary_sum_pitch, cell_vol_dev);
            cuda_check_error();
        
            cudaEventRecord(stop);
            cudaEventSynchronize(start);
            cudaEventSynchronize(stop);

            stop_time = sw.now(); // Compute new site
            record << stop_time - start_time << ", ";

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        } // GPU voro kernel only

        { // copy data back to the cpu
            start_time = sw.now();

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            if (site_is_transposed)
                cudaMemcpy2D(site.data(), n_site * sizeof(float), site_transposed_dev, site_pitch_in_bytes, n_site * sizeof(float), 3, cudaMemcpyDeviceToHost);
            else
            {
                cudaMemcpy(site.data(), (float*)kn->d_stored_points, 3 * n_site * sizeof(float), cudaMemcpyDeviceToHost);
                kn_free(&kn);
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(start);
            cudaEventSynchronize(stop);

            stop_time = sw.now(); // Compute new site
            record << stop_time - start_time << ", ";

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        } // copy data back to the cpu


        std::ofstream file("iter_" + std::to_string(i) + ".xyz");
        file << n_site << std::endl;
        for (int i = 0; i < n_site; i++)
            if (site_is_transposed)
                file << site[i] << "  " << site[i + n_site] << "  " << site[i + (n_site << 1)] << std::endl;
            else
                file << site[3 * i] << "  " << site[3 * i + 1] << "  " << site[3 * i + 2] << std::endl;
        file.close();

        record << std::endl;
    }

    cudaFree(vert_dev);
    cudaFree(idx_dev);
    cudaFree(tet_knn_dev);
    cudaFree(cell_vol_dev);
    cudaFree(voronoi_cells_dev);
    cudaFree(cell_bary_sum_dev);
    cudaFree(site_transposed_dev);
    cudaFree(site_knn_dev);

    record.close();
}
