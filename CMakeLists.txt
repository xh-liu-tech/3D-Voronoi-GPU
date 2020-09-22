cmake_minimum_required(VERSION 2.8)

project(VolumeVoronoiGPU)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_DEBUG_POSTFIX "d")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

cuda_add_executable(VolumeVoronoiGPU
    test_voronoi voronoi.cu knearests.cu test_voronoi.cu
    OPTIONS "-arch sm_61 -lineinfo"
)

target_compile_definitions(VolumeVoronoiGPU PUBLIC -Dgeogram_EXPORTS)

if (${MSVC})
    if (${CMAKE_VERSION} VERSION_GREATER 3.6.0)
        set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT "VolumeVoronoiGPU")
    endif()
endif()

add_subdirectory(extern/tetgen)
add_subdirectory(extern/kNN-CUDA)

target_link_directories(VolumeVoronoiGPU PUBLIC "${CMAKE_BINARY_DIR}/lib")
target_link_libraries(VolumeVoronoiGPU
	debug tetd
	debug kNN-CUDAd
    optimized tet
	optimized kNN-CUDA
    general ${CUDA_LIBRARIES}
	general cuda
	general cublas
)
