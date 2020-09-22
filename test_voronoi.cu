#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cstdlib>
#include <cstdio>

#include "params.h"
#include "voronoi.h"
#include "stopwatch.h"

void get_bbox(const std::vector<float>& vertices, float& xmin, float& ymin, float& zmin, float& xmax, float& ymax, float& zmax) {
    int nb_v = vertices.size() / 3;
    xmin = xmax = vertices[0];
    ymin = ymax = vertices[1];
    zmin = zmax = vertices[2];
    for (int i = 1; i < nb_v; ++i)
    {
        xmin = std::min(xmin, vertices[3 * i]);
        ymin = std::min(ymin, vertices[3 * i + 1]);
        zmin = std::min(zmin, vertices[3 * i + 2]);
        xmax = std::max(xmax, vertices[3 * i]);
        ymax = std::max(ymax, vertices[3 * i + 1]);
        zmax = std::max(zmax, vertices[3 * i + 2]);
    }
    float d = xmax - xmin;
    d = std::max(d, ymax - ymin);
    d = std::max(d, zmax - zmin);
    d = 0.001f * d;
    xmin -= d;
    ymin -= d;
    zmin -= d;
    xmax += d;
    ymax += d;
    zmax += d;
}

bool load_tet(
    const std::string& filename,
    std::vector<float>& vertices,
    std::vector<int>& indices,
    bool normalize = true
)
{
    std::string s;
    int n_vertex, n_tet, temp;

    std::ifstream input(filename);
    if (input.fail())
        return false;

    std::string ext = filename.substr(filename.find_last_of('.') + 1);
    if (ext == "tet")
    {
        input >> n_vertex;
        std::getline(input, s);
        input >> n_tet;
        std::getline(input, s);

        vertices.resize(3 * n_vertex);
        indices.resize(n_tet << 2);

        for (int i = 0; i < n_vertex; ++i)
            input >> vertices[3 * i] >> vertices[3 * i + 1] >> vertices[3 * i + 2];

        for (int i = 0; i < n_tet; ++i)
        {
            input >> temp >> indices[(i << 2)] >> indices[(i << 2) + 1] >> indices[(i << 2) + 2] >> indices[(i << 2) + 3];
            assert(temp == 4);
        }
    }
    else if (ext == "vtk")
    {
        for (int i = 0; i < 4; ++i)
            std::getline(input, s); // skip first 4 lines
        
        input >> s >> n_vertex >> s;
        vertices.resize(3 * n_vertex);
        for (int i = 0; i < n_vertex; ++i)
            input >> vertices[3 * i] >> vertices[3 * i + 1] >> vertices[3 * i + 2];

        input >> s >> n_tet >> s;
        indices.resize(n_tet << 2);
        for (int i = 0; i < n_tet; ++i)
        {
            input >> temp >> indices[(i << 2)] >> indices[(i << 2) + 1] >> indices[(i << 2) + 2] >> indices[(i << 2) + 3];
            assert(temp == 4);
            for (int j = 0; j < 4; ++j)
                --indices[(i << 2) + j];
        }
    }
    else
    {
        input.close();
        return false;
    }


    input.close();

    float xmin, ymin, zmin, xmax, ymax, zmax;
    get_bbox(vertices, xmin, ymin, zmin, xmax, ymax, zmax);

    if (normalize) // normalize vertices between [0,1000]^3
    {
        float maxside = std::max(std::max(xmax - xmin, ymax - ymin), zmax - zmin);
#pragma omp parallel for
        for (int i = 0; i < n_vertex; i++)
        {
            vertices[3 * i] = 1000.f * (vertices[3 * i] - xmin) / maxside;
            vertices[3 * i + 1] = 1000.f * (vertices[3 * i + 1] - ymin) / maxside;
            vertices[3 * i + 2] = 1000.f * (vertices[3 * i + 2] - zmin) / maxside;
        }
        get_bbox(vertices, xmin, ymin, zmin, xmax, ymax, zmax);
        std::cerr << "bbox [" << xmin << ":" << xmax << "], [" << ymin << ":" << ymax << "], [" << zmin << ":" << zmax << "]" << std::endl;
    }

    return true;
}

void drop_xyz_file(const bool site_is_transposed, const std::vector<float>& site, const int n_site, const char *filename) {
    std::fstream file;
    file.open(filename, std::ios_base::out);
    file << n_site << std::endl;
    for(int i = 0; i < n_site; i++)
        if (site_is_transposed)
            file << site[i] << "  " << site[i + n_site] << "  " << site[i + (n_site << 1)] << std::endl;
        else
            file << site[3 * i] << "  " << site[3 * i + 1] << "  " << site[3 * i + 2] << std::endl;
    file.close();
}

void load_xyz_file(bool& site_is_transposed, std::vector<float>& site, int& n_site, const char* filename)
{
    std::ifstream file(filename);
    file >> n_site;
    site_is_transposed = n_site < KNEARESTS_MIN_N; // use knn_cuda_global_dev if true, else knearests
    site.resize(n_site * 3);
    for (int i = 0; i < n_site; ++i)
        if (site_is_transposed)
            file >> site[i] >> site[i + n_site] >> site[i + (n_site << 1)];
        else
            file >> site[3 * i] >> site[3 * i + 1] >> site[3 * i + 2];
    file.close();
}

void printDevProp() {
    
    int devCount; // Number of CUDA devices
    cudaError_t err = cudaGetDeviceCount(&devCount);
    if (err != cudaSuccess) {
        std::cerr << "Failed to initialize CUDA / failed to count CUDA devices (error code << "
		  << cudaGetErrorString(err) << ")! [file: " << __FILE__ << ", line: " <<  __LINE__ << "]" << std::endl;
        exit(1);
    }
    
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);

    // Iterate through devices
    for (int i=0; i<devCount; ++i) {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printf("Major revision number:         %d\n",  devProp.major);
        printf("Minor revision number:         %d\n",  devProp.minor);
        printf("Name:                          %s\n",  devProp.name);
        printf("Total global memory:           %lu\n",  devProp.totalGlobalMem);
        printf("Total shared memory per block: %lu\n",  devProp.sharedMemPerBlock);
        printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
        printf("Warp size:                     %d\n",  devProp.warpSize);
        printf("Maximum memory pitch:          %lu\n",  devProp.memPitch);
        printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
        for (int i = 0; i < 3; ++i)
            printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
        for (int i = 0; i < 3; ++i)
            printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
        printf("Clock rate:                    %d\n",  devProp.clockRate);
        printf("Total constant memory:         %lu\n",  devProp.totalConstMem);
        printf("Texture alignment:             %lu\n",  devProp.textureAlignment);
        printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
        printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
        printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    }
}

int main(int argc, char** argv) {
    printDevProp();
    if (4 > argc)
    {
        std::cerr << "Usage: " << argv[0] << " <tet_mesh.tet/vtk> <sites_file.xyz> <nb_iter> <k (optional)> (e.g.: " << argv[0] << " ../data/joint.tet ../data/joint.xyz 120 0)" << std::endl;
        return 1;
    }
    int* initptr = nullptr;
    cudaError_t err = cudaMalloc(&initptr, sizeof(int)); // unused memory, needed for initialize the GPU before time measurements
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate (error code << " << cudaGetErrorString(err) << ")! [file: " << __FILE__ << ", line: " << __LINE__ << "]" << std::endl;
        return 1;
    }

    std::vector<float> vertices;
    std::vector<int> indices;

    if (!load_tet(argv[1], vertices, indices))
    {
        std::cerr << argv[1] << ": could not load file" << std::endl;
        return 1;
    }

    int n_site;
    bool site_is_transposed;
    std::vector<float> site;
    load_xyz_file(site_is_transposed, site, n_site, argv[2]);

    if (5 == argc)
        compute_clipped_voro_diagram_GPU(vertices, indices, site, n_site, site_is_transposed, atoi(argv[3]), atoi(argv[4]));
    else
        compute_clipped_voro_diagram_GPU(vertices, indices, site, n_site, site_is_transposed, atoi(argv[3]));

    drop_xyz_file(site_is_transposed, site, n_site, "out.xyz");

    cudaFree(initptr);
    return 0;
}
