#pragma once

typedef struct {
    int allocated_points;        // number of input points
    int dimx, dimy, dimz;        // grid resolution
    int num_cell_offsets;        // actual number of cells in the offset grid
    int *d_cell_offsets;         // cell offsets (sorted by rings), Nmax*Nmax*Nmax*Nmax
    float *d_cell_offset_dists;  // stores min dist to the cells in the rings
    float *d_cell_max;           // KNN_BLOCK_SIZE floats, store max visited ring per thread id
    unsigned int *d_permutation; // allows to restore original point order
    int *d_counters;             // counters per cell,   dimx*dimy*dimz
    int *d_ptrs;                 // cell start pointers, dimx*dimy*dimz
    int *d_globcounter;          // global allocation counter, 1
    float3 *d_stored_points;     // input points sorted, numpoints 
    unsigned int *d_knearests;   // knn, allocated_points * KN
} kn_problem;

// ------------------------------------------------------------

kn_problem   *kn_prepare(float3 *points, int numpoints);
void          kn_solve(kn_problem *kn);
void          kn_free(kn_problem **kn);
void          kn_print_stats(kn_problem *kn);

float3        *kn_get_points(kn_problem *kn);
unsigned int *kn_get_knearests(kn_problem *kn);
unsigned int *kn_get_permutation(kn_problem *kn);

