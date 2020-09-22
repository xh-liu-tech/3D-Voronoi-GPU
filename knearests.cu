#include <stdio.h>

#include <cuda_runtime.h>
#include <assert.h>

#include <iostream>
#include <set>
#include <vector>
#include <map>
#include <float.h>

#include "knearests.h"
#include "stopwatch.h"
#include "params.h"

// ------------------------------------------------------------

// it is supposed that all points fit in range [0,1000]^3
__device__ int cellFromPoint(int xdim, int ydim, int zdim, float3 p) {
    int   i = (int)floor(p.x * (float)xdim / 1000.f);
    int   j = (int)floor(p.y * (float)ydim / 1000.f);
    int   k = (int)floor(p.z * (float)zdim / 1000.f);
    i = max(0, min(i, xdim - 1));
    j = max(0, min(j, ydim - 1));
    k = max(0, min(k, zdim - 1));
    return i + j*xdim + k*xdim*ydim;
}

__global__ void count(const float3 *points, int numPoints, int xdim, int ydim, int zdim, int *counters) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < numPoints) {
        int cell = cellFromPoint(xdim, ydim, zdim, points[id]);
        atomicAdd(counters + cell, 1);
    }
}

__global__ void reserve(int xdim, int ydim, int zdim, const int *counters, int *globalcounter, int *ptrs) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < xdim*ydim*zdim) {
        int cnt = counters[id];
        if (cnt > 0) {
            ptrs[id] = atomicAdd(globalcounter, cnt);
        }
    }
}

// it supposes that counters buffer is set to zero
__global__ void store(const float3 *points, int numPoints, int xdim, int ydim, int zdim, const int *ptrs, int *counters, int num_stored, float3 *stored_points, unsigned int *permutation) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < numPoints) {
        float3 p = points[id];
        int cell = cellFromPoint(xdim, ydim, zdim, p);
        int pos = ptrs[cell] + atomicAdd(counters + cell, 1);
        permutation[pos] = id;
        stored_points[pos] = p;
    }
}

template <typename T> __device__ void inline swap_on_device(T& a, T& b) {
    T c(a); a=b; b=c;
}

__device__ void heapify(unsigned int *keys, float *vals, int node, int size) {
    int j = node;
    while (true) { 
        int left  = 2*j+1;
        int right = 2*j+2;
        int largest = j;
        if ( left<size && vals[ left]>vals[largest]) {
            largest = left;
        }
        if (right<size && vals[right]>vals[largest]) {
            largest = right;
        }
        if (largest==j) return;
        swap_on_device(vals[j], vals[largest]);
        swap_on_device(keys[j], keys[largest]);
        j = largest;
    }
}

__device__ void heapsort(unsigned int *keys, float *vals, int size) {
    while (size) {
        swap_on_device(vals[0], vals[size-1]);
        swap_on_device(keys[0], keys[size-1]);
        heapify(keys, vals, 0, --size);
    }
}

__global__ void knearest(int xdim, int ydim, int zdim, int num_stored, const int *ptrs, const int *counters, const float3 *stored_points, int num_cell_offsets, const int *cell_offsets, const float *cell_offset_distances, unsigned int *g_knearests, float *d_cell_max) {
    // each thread updates its k-nearests
    __shared__ unsigned int knearests      [_K_*KNN_BLOCK_SIZE];
    __shared__ float        knearests_dists[_K_*KNN_BLOCK_SIZE];

    int point_in = threadIdx.x + blockIdx.x*KNN_BLOCK_SIZE;
    if (point_in >= num_stored) return;

    // point considered by this thread
    float3 p = stored_points[point_in];

    int cell_in = cellFromPoint(xdim, ydim, zdim, p);
    int offs = threadIdx.x*_K_;

    for (int i = 0; i < _K_; i++) {
        knearests      [offs + i] = UINT_MAX;
        knearests_dists[offs + i] = FLT_MAX;
    }

    int o = 0;
    do {
        float min_dist = cell_offset_distances[o];
        if (min_dist>d_cell_max[threadIdx.x]) d_cell_max[threadIdx.x] = min_dist;
        if (knearests_dists[offs] < min_dist) break;

        int cell = cell_in + cell_offsets[o];
        if (cell>=0 && cell<xdim*ydim*zdim) {
            int cell_base = ptrs[cell];
            int num = counters[cell];
            for (int ptr=cell_base; ptr<cell_base+num; ptr++) {
                if (ptr==point_in) continue; // exclude the point itself from its neighbors
                float3 p_cmp = stored_points[ptr];
                float d = (p_cmp.x-p.x)*(p_cmp.x-p.x) + (p_cmp.y-p.y)*(p_cmp.y-p.y) + (p_cmp.z-p.z)*(p_cmp.z-p.z);

                if (d < knearests_dists[offs]) {
                    // replace current max
                    knearests[offs] = ptr;
                    knearests_dists[offs] = d;

                    heapify(knearests+offs, knearests_dists+offs, 0, _K_);
                }
            } // pts inside the cell
        } // valid cell id
    } while (o++<num_cell_offsets); // cell offsets
    if (o==num_cell_offsets) {
        d_cell_max[threadIdx.x] = FLT_MAX; // no guarantee to have found k nearest
    }

    heapsort(knearests+offs, knearests_dists+offs, _K_);

    // store result
    for (int i = 0; i < _K_; i++) {
        g_knearests[point_in*_K_ + i] = knearests[offs + i];
        //g_knearests[point_in*_K_ + i] = knearests[offs + _K_-1 -i];
    }
}

// ------------------------------------------------------------

void kn_firstbuild(kn_problem *kn, float3 *d_points, int numpoints) {
    cudaError_t err = cudaSuccess;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    { // count points per grid cell
        int threadsPerBlock = 256;
        int blocksPerGrid = (numpoints + threadsPerBlock - 1) / threadsPerBlock;
        count << <blocksPerGrid, threadsPerBlock >> >(d_points, numpoints, kn->dimx, kn->dimy, kn->dimz, kn->d_counters);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Failed  (error code " << cudaGetErrorString(err) << ")! [file: " << __FILE__ << ", line: " <<  __LINE__ << "]" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    { // reserve memory for stored points
        int threadsPerBlock = 4; // prolly even 1
        int blocksPerGrid = (kn->dimx*kn->dimy*kn->dimz + threadsPerBlock - 1) / threadsPerBlock;
        reserve << <blocksPerGrid, threadsPerBlock >> >(kn->dimx, kn->dimy, kn->dimz, kn->d_counters, kn->d_globcounter, kn->d_ptrs);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Failed  (error code " << cudaGetErrorString(err) << ")! [file: " << __FILE__ << ", line: " <<  __LINE__ << "]" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    { // store
        // zero counters
        cudaMemset(kn->d_counters, 0x00, kn->dimx*kn->dimy*kn->dimz*sizeof(int));
        // call kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (numpoints + threadsPerBlock - 1) / threadsPerBlock;
        store << <blocksPerGrid, threadsPerBlock >> >(d_points, numpoints, kn->dimx, kn->dimy, kn->dimz, kn->d_ptrs, kn->d_counters, kn->allocated_points, kn->d_stored_points, kn->d_permutation);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Failed  (error code " << cudaGetErrorString(err) << ")! [file: " << __FILE__ << ", line: " <<  __LINE__ << "]" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    IF_VERBOSE(std::cerr << "kn_firstbuild: " << milliseconds << " msec" << std::endl;)
}

// ------------------------------------------------------------

void gpuMalloc(void **ptr, size_t size) {
    cudaError_t err = cudaMalloc(ptr, size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate (error code << " << cudaGetErrorString(err) << ")! [file: " << __FILE__ << ", line: " <<  __LINE__ << "]" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void gpuMallocNCopy(void **dst, const void *src, size_t size) {
    gpuMalloc(dst, size);
    cudaError_t err = cudaMemcpy(*dst, src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy from host to device (error code << " << cudaGetErrorString(err) << ")! [file: " << __FILE__ << ", line: " <<  __LINE__ << "]" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void gpuMallocNMemset(void **ptr, int value, size_t size) {
    gpuMalloc(ptr, size);
    cudaError_t err = cudaMemset(*ptr, value, size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to write to device memory (error code << " << cudaGetErrorString(err) << ")! [file: " << __FILE__ << ", line: " <<  __LINE__ << "]" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// ------------------------------------------------------------

kn_problem *kn_prepare(float3 *points, int numpoints) {
    kn_problem *kn = (kn_problem*)malloc(sizeof(kn_problem));
    kn->allocated_points = numpoints;

    kn->d_permutation       = NULL;
    kn->d_cell_offsets      = NULL;
    kn->d_cell_offset_dists = NULL;
    kn->d_counters          = NULL;
    kn->d_ptrs              = NULL;
    kn->d_globcounter       = NULL;
    kn->d_stored_points     = NULL;
    kn->d_knearests         = NULL;

    int sz = max(1,(int)round(pow(numpoints / 3.1f, 1.0f / 3.0)));
    kn->dimx = sz;
    kn->dimy = sz;
    kn->dimz = sz;

    int Nmax = 16;
    if (sz < Nmax) {
        std::cerr << "Current implementation does not support low number of input points" << std::endl;
        exit(EXIT_FAILURE);
    }
    // create cell offsets, very naive approach, should be fine, pre-computed once
    int alloc = Nmax*Nmax*Nmax*Nmax;
    int   *cell_offsets      =   (int*)malloc(alloc*sizeof(int));
    float *cell_offset_dists = (float*)malloc(alloc*sizeof(float));
    cell_offsets[0] = 0;
    cell_offset_dists[0] = 0.0f;
    kn->num_cell_offsets = 1;
    for (int ring = 1; ring < Nmax; ring++) {
        for (int k = -Nmax; k <= Nmax; k++) {
            for (int j = -Nmax; j <= Nmax; j++) {
                for (int i = -Nmax; i <= Nmax; i++) {
                    if (max(abs(i), max(abs(j), abs(k))) != ring) continue;

                    int id_offset = i + j*kn->dimx + k*kn->dimx*kn->dimy;
                    if (id_offset == 0) { 
                        std::cerr << "Error generating offsets" << std::endl;
                        exit(EXIT_FAILURE); 
                    }
                    cell_offsets[kn->num_cell_offsets] = id_offset;
                    float d = 1000.*(float)(ring - 1) / (float)max(kn->dimx, max(kn->dimy, kn->dimz));
                    cell_offset_dists[kn->num_cell_offsets] = d*d; // squared
                    kn->num_cell_offsets++;
                    if (kn->num_cell_offsets >= alloc) {
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
    }
    std::cerr << "num_cell_offsets = " << kn->num_cell_offsets << std::endl;

    size_t memory_used = 0, bufsize = 0;
    
    bufsize = kn->num_cell_offsets*sizeof(int); // allocate cell offsets
    memory_used += bufsize;
    gpuMallocNCopy((void **)&kn->d_cell_offsets, cell_offsets, bufsize);
    free(cell_offsets);

    bufsize = kn->num_cell_offsets*sizeof(float); // allocate cell offsets distances
    memory_used += bufsize;
    gpuMallocNCopy((void **)&kn->d_cell_offset_dists, cell_offset_dists, bufsize);
    free(cell_offset_dists);


    bufsize = KNN_BLOCK_SIZE*sizeof(float);
    memory_used += bufsize;
    gpuMallocNMemset((void **)&kn->d_cell_max, 0x00, bufsize); 

 
    float3 *d_points = NULL;
    bufsize = numpoints*sizeof(float3); // allocate input points
    memory_used += bufsize;
    gpuMallocNCopy((void **)&d_points, points, bufsize); 

    bufsize = kn->dimx*kn->dimy*kn->dimz*sizeof(int);  // allocate cell counters
    memory_used += bufsize;
    gpuMallocNMemset((void **)&kn->d_counters, 0x00, bufsize); 
    
    bufsize = kn->dimx*kn->dimy*kn->dimz*sizeof(int); // allocate cell start pointers
    memory_used += bufsize;
    gpuMallocNMemset((void **)&kn->d_ptrs, 0x00, bufsize); 

    bufsize = sizeof(int);
    memory_used += bufsize;
    gpuMallocNMemset((void **)&kn->d_globcounter, 0x00, bufsize); 

    bufsize = kn->allocated_points*sizeof(float3); // allocate stored points
    memory_used += bufsize;
    gpuMallocNMemset((void **)&kn->d_stored_points, 0x00, bufsize); 

    bufsize += kn->allocated_points*_K_*sizeof(int);
    memory_used += bufsize;
    gpuMallocNMemset((void **)&kn->d_knearests, 0xFF, bufsize); 

    bufsize += kn->allocated_points*sizeof(int); // keep the track of reordering
    memory_used += bufsize;
    gpuMallocNMemset((void **)&kn->d_permutation, 0xFF, bufsize); 

    // construct initial structure
    kn_firstbuild(kn, d_points, numpoints);

    // we no longer need the initial points
    cudaFree(d_points);
    IF_VERBOSE(std::cerr << "GPU memory used: " << memory_used/1048576 << " Mb" << std::endl);
    return kn;
}

// ------------------------------------------------------------

void kn_solve(kn_problem *kn) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = KNN_BLOCK_SIZE;
    int blocksPerGrid = (kn->allocated_points + threadsPerBlock - 1) / KNN_BLOCK_SIZE;

    IF_VERBOSE(std::cerr << "threads per block: " << threadsPerBlock << ", blocks per grid: " << blocksPerGrid << std::endl);

    cudaEventRecord(start);

    knearest << <blocksPerGrid, threadsPerBlock >> >(
            kn->dimx, kn->dimy, kn->dimz, kn->allocated_points,
            kn->d_ptrs, kn->d_counters, (float3 *)kn->d_stored_points,
            kn->num_cell_offsets, kn->d_cell_offsets, kn->d_cell_offset_dists,
            kn->d_knearests, kn->d_cell_max);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed  (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    IF_VERBOSE(std::cerr << "kn_solve: " << milliseconds << " msec" << std::endl);

    {
        float *cell_max = (float*)malloc(KNN_BLOCK_SIZE * sizeof(float));
        cudaError_t err = cudaMemcpy(cell_max, kn->d_cell_max, KNN_BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy from device to host (error code " << cudaGetErrorString(err) << ")!" << std::endl;
            exit(EXIT_FAILURE);
        }
        float m = 0;
        for (int i=0; i<KNN_BLOCK_SIZE; i++) {
            if (cell_max[i]>m) m = cell_max[i];
        }
        std::cerr << "Max visited ring: " << (sqrtf(m)/1000.*kn->dimx+1) << " / " << (pow(kn->num_cell_offsets, 1./3.)-1)/2. << std::endl;
    }

}

// ------------------------------------------------------------

void kn_free(kn_problem **kn) {
    cudaFree((*kn)->d_cell_offsets);
    cudaFree((*kn)->d_cell_offset_dists);
    cudaFree((*kn)->d_cell_max);
    cudaFree((*kn)->d_counters);
    cudaFree((*kn)->d_ptrs);
    cudaFree((*kn)->d_globcounter);
    cudaFree((*kn)->d_stored_points);
    cudaFree((*kn)->d_knearests);
    cudaFree((*kn)->d_permutation);
    free(*kn);
    *kn = NULL;
}

float3 *kn_get_points(kn_problem *kn) {
    float3 *stored_points = (float3*)malloc(kn->allocated_points * sizeof(float3));
    cudaError_t err = cudaMemcpy(stored_points, kn->d_stored_points, kn->allocated_points * sizeof(float3), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "[kn_get_points] Failed to copy from device to host (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }
    return stored_points;
}

unsigned int *kn_get_permutation(kn_problem *kn) {
    unsigned int *permutation = (unsigned int*)malloc(kn->allocated_points*sizeof(int));
    cudaError_t err = cudaMemcpy(permutation, kn->d_permutation, kn->allocated_points * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "[kn_get_permutation] Failed to copy from device to host (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }
    return permutation;
}

unsigned int *kn_get_knearests(kn_problem *kn) {
    unsigned int *knearests = (unsigned int*)malloc(kn->allocated_points * _K_ * sizeof(int));
    cudaError_t err = cudaMemcpy(knearests, kn->d_knearests, kn->allocated_points * _K_ * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "[kn_print_stats] Failed to copy from device to host (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }
    return knearests;
}

void kn_print_stats(kn_problem *kn) {
    Stopwatch W("kn_print_stats");
    cudaError_t err = cudaSuccess;

    int *counters = (int*)malloc(kn->dimx*kn->dimy*kn->dimz*sizeof(int));
    err = cudaMemcpy(counters, kn->d_counters, kn->dimx*kn->dimy*kn->dimz*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "[kn_print_stats] Failed to copy from device to host (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // stats on counters
    int tot = 0;
    int cmin = INT_MAX, cmax = 0;
    std::map<int, int> histo;
    for (int c = 0; c < kn->dimx*kn->dimy*kn->dimz; c++) {
        histo[counters[c]]++;
        cmin = min(cmin, counters[c]);
        cmax = max(cmax, counters[c]);
        tot += counters[c];
    }
    std::cerr << "Grid:  points per cell: " << cmin << " (min), " << cmax << " (max), " << (kn->allocated_points)/(float)(kn->dimx*kn->dimy*kn->dimz) << " avg, total " << tot << std::endl;
    for (std::map<int,int>::const_iterator H = histo.begin(); H!=histo.end(); ++H) {
        std::cerr << "[" << H->first << "] => " << H->second << std::endl;
    }
    free(counters);
}

