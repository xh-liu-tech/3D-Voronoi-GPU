#define KNEARESTS_MIN_N 14000
#define WIN_RANDOM(x) ((long long)(((double)rand() * x) / RAND_MAX)) // random int in [0, x]

#define PRESET 0

#if 0==PRESET               // conservative settings (white noise)
#define VORO_BLOCK_SIZE 16
#define KNN_BLOCK_SIZE  32
#define _K_             90
// #define _TET_K_         70
#define _MAX_P_         64
#define _MAX_T_         96
#elif 1==PRESET            // perturbed grid settings
#define VORO_BLOCK_SIZE 16
#define KNN_BLOCK_SIZE  32
#define _K_             90
#define _MAX_P_         50
#define _MAX_T_         96
#elif 2==PRESET            // blue noise settings
#define VORO_BLOCK_SIZE 16
#define KNN_BLOCK_SIZE  64
#define _K_             35
#define _MAX_P_         32
#define _MAX_T_         96
#endif

// Uncomment to activate arithmetic filters.
//   If arithmetic filters are activated,
//   status is set to needs_exact_predicates
//   whenever predicates could not be evaluated
//   using floating points on the GPU
// #define USE_ARITHMETIC_FILTER

#define IF_VERBOSE(x) x

