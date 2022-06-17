#define WARP_SIZE 32 // how to initialize this?

template <typename Size1, typename Size2>
__host__ __device__
Size1 DIVIDE_INTO(Size1 N, Size2 granularity)
{
    return (N + (granularity - 1)) / granularity;
}

// segmented reduction in shared memory
__device__ float segreduce_warp(const int thread_lane, int row, double val, int * rows, int * vals)
{
    rows[threadIdx.x] = row;
    vals[threadIdx.x] = val;

    if( thread_lane >=  1 && row == rows[threadIdx.x -  1] ) {
        vals[threadIdx.x] = val = val + vals[threadIdx.x -  1];
    }
    if( thread_lane >=  2 && row == rows[threadIdx.x -  2] ) {
        vals[threadIdx.x] = val = val + vals[threadIdx.x -  2];
    }
    if( thread_lane >=  4 && row == rows[threadIdx.x -  4] ) {
        vals[threadIdx.x] = val = val + vals[threadIdx.x -  4];
    }
    if( thread_lane >=  8 && row == rows[threadIdx.x -  8] ) {
        vals[threadIdx.x] = val = val + vals[threadIdx.x -  8];
    }
    if( thread_lane >= 16 && row == rows[threadIdx.x - 16] ) {
        vals[threadIdx.x] = val = val + vals[threadIdx.x - 16];
    }

    return val;
}

__device__ void segreduce_block(const int *idx, double *val)
{
    float left = 0;
    if( threadIdx.x >=   1 && idx[threadIdx.x] == idx[threadIdx.x -   1] ) {
        left = val[threadIdx.x -   1];
    }
    __syncthreads();
    val[threadIdx.x] = val[threadIdx.x] + left;
    left = 0;
    __syncthreads();
    if( threadIdx.x >=   2 && idx[threadIdx.x] == idx[threadIdx.x -   2] ) {
        left = val[threadIdx.x -   2];
    }
    __syncthreads();
    val[threadIdx.x] = val[threadIdx.x] + left;
    left = 0;
    __syncthreads();
    if( threadIdx.x >=   4 && idx[threadIdx.x] == idx[threadIdx.x -   4] ) {
        left = val[threadIdx.x -   4];
    }
    __syncthreads();
    val[threadIdx.x] = val[threadIdx.x] + left;
    left = 0;
    __syncthreads();
    if( threadIdx.x >=   8 && idx[threadIdx.x] == idx[threadIdx.x -   8] ) {
        left = val[threadIdx.x -   8];
    }
    __syncthreads();
    val[threadIdx.x] = val[threadIdx.x] + left;
    left = 0;
    __syncthreads();
    if( threadIdx.x >=  16 && idx[threadIdx.x] == idx[threadIdx.x -  16] ) {
        left = val[threadIdx.x -  16];
    }
    __syncthreads();
    val[threadIdx.x] = val[threadIdx.x] + left;
    left = 0;
    __syncthreads();
    if( threadIdx.x >=  32 && idx[threadIdx.x] == idx[threadIdx.x -  32] ) {
        left = val[threadIdx.x -  32];
    }
    __syncthreads();
    val[threadIdx.x] = val[threadIdx.x] + left;
    left = 0;
    __syncthreads();
    if( threadIdx.x >=  64 && idx[threadIdx.x] == idx[threadIdx.x -  64] ) {
        left = val[threadIdx.x -  64];
    }
    __syncthreads();
    val[threadIdx.x] = val[threadIdx.x] + left;
    left = 0;
    __syncthreads();
    if( threadIdx.x >= 128 && idx[threadIdx.x] == idx[threadIdx.x - 128] ) {
        left = val[threadIdx.x - 128];
    }
    __syncthreads();
    val[threadIdx.x] = val[threadIdx.x] + left;
    left = 0;
    __syncthreads();
    if( threadIdx.x >= 256 && idx[threadIdx.x] == idx[threadIdx.x - 256] ) {
        left = val[threadIdx.x - 256];
    }
    __syncthreads();
    val[threadIdx.x] = val[threadIdx.x] + left;
    left = 0;
    __syncthreads();
}


__global__ void spmv_coo_normal(const int *x, const int *y, const double *val, const double *vec, double *result, int N) {
    // Uses a grid-stride loop to perform dot product
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        atomicAdd(result + x[i], val[i] * vec[y[i]]);
    }
}


template <unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE,1)
__global__ void
spmv_coo_flat_kernel(const int num_nonzeros,
                     const unsigned int interval_size,
                     const int *I,
                     const int *J,
                     const double *V,
                     const double *x,
                     double* y,
                     int *temp_rows,
                     double *temp_vals)
{

    __shared__ volatile int rows[48 *(BLOCK_SIZE/32)];
    __shared__ volatile double vals[BLOCK_SIZE];


    const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;                         // global thread index
    const int thread_lane = threadIdx.x & (WARP_SIZE-1);                                   // thread index within the warp
    const int warp_id     = thread_id   / WARP_SIZE;                                       // global warp index

    const int interval_begin = warp_id * interval_size;                                    // warp's offset into I,J,V
    const int interval_end = (interval_begin + interval_size < num_nonzeros) ? interval_begin + interval_size : num_nonzeros;  // end of warps's work

    const int idx = 16 * (threadIdx.x/32 + 1) + threadIdx.x;                               // thread's index into padded rows array

    rows[idx - 16] = -1;                                                                         // fill padding with invalid row index

    if(interval_begin >= interval_end)                                                           // warp has no work to do
        return;

    if (thread_lane == 31)
    {
        // initialize the carry in values
        rows[idx] = I[interval_begin];
        vals[threadIdx.x] = 0.0;
    }

    for(int n = interval_begin + thread_lane; n < interval_end; n += WARP_SIZE)
    {
        int row = I[n];                                         // row index (i)
        float val = V[n] * x[ J[n] ];            // A(i,j) * x(j)

        if (thread_lane == 0)
        {
            if(row == rows[idx + 31])
                val = val + vals[threadIdx.x + 31];                        // row continues
            else
                y[rows[idx + 31]] = y[rows[idx + 31]] + vals[threadIdx.x + 31];  // row terminated
        }

        rows[idx]         = row;
        vals[threadIdx.x] = val;

        if(row == rows[idx -  1]) {
            vals[threadIdx.x] = val = val + vals[threadIdx.x -  1];
        }
        if(row == rows[idx -  2]) {
            vals[threadIdx.x] = val = val + vals[threadIdx.x -  2];
        }
        if(row == rows[idx -  4]) {
            vals[threadIdx.x] = val = val + vals[threadIdx.x -  4];
        }
        if(row == rows[idx -  8]) {
            vals[threadIdx.x] = val = val + vals[threadIdx.x -  8];
        }
        if(row == rows[idx - 16]) {
            vals[threadIdx.x] = val = val + vals[threadIdx.x - 16];
        }

        if(thread_lane < 31 && row != rows[idx + 1])
            y[row] = y[row] + vals[threadIdx.x];                                            // row terminated
    }

    if(thread_lane == 31)
    {
        // write the carry out values
        temp_rows[warp_id] = (int)(rows[idx]);
        temp_vals[warp_id] = (double)(vals[threadIdx.x]);
    }
}

// The second level of the segmented reduction operation
template <unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE,1)
__global__ void
spmv_coo_reduce_update_kernel(const unsigned int num_warps,
                              const int* temp_rows,
                              const double* temp_vals,
                              double* y)
{
    __shared__ int rows[BLOCK_SIZE + 1];
    __shared__ double vals[BLOCK_SIZE + 1];

    const int end = num_warps - (num_warps & (BLOCK_SIZE - 1));

    if (threadIdx.x == 0)
    {
        rows[BLOCK_SIZE] = -1;
        vals[BLOCK_SIZE] = 0.0;
    }

    __syncthreads();

    int i = threadIdx.x;

    while (i < end)
    {
        // do full blocks
        rows[threadIdx.x] = temp_rows[i];
        vals[threadIdx.x] = temp_vals[i];

        __syncthreads();

        segreduce_block(rows, vals);

        if (rows[threadIdx.x] != rows[threadIdx.x + 1])
            y[rows[threadIdx.x]] = y[rows[threadIdx.x]] + vals[threadIdx.x];

        __syncthreads();

        i += BLOCK_SIZE;
    }

    if (end < num_warps) {
        if (i < num_warps) {
            rows[threadIdx.x] = temp_rows[i];
            vals[threadIdx.x] = temp_vals[i];
        } else {
            rows[threadIdx.x] = -1;
            vals[threadIdx.x] = 0.0;
        }

        __syncthreads();

        segreduce_block(rows, vals);

        if (i < num_warps)
            if (rows[threadIdx.x] != rows[threadIdx.x + 1])
                y[rows[threadIdx.x]] = y[rows[threadIdx.x]] + vals[threadIdx.x];
    }
}






// ----------------CPU CALLER--------------
void __spmv_coo_flat(const int* x, 
                    const int* y, 
                    const double* val, //matrix: we have to split it into 3 vector COO
                    const double* pr_gpu, //multiplying vector
                    double* gpu_result, //result vector
                    const int num_nonzeros,
                    cudaStream_t spmv_stream)
{

    const unsigned int BLOCK_SIZE = 256;
    const unsigned int MAX_BLOCKS = 40; // on NVIDIA GTX 1050: 5 SM * 8 active_blocks per SM
    const unsigned int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

    const unsigned int num_units  = num_nonzeros / WARP_SIZE;
    const unsigned int num_warps  = (num_units < WARPS_PER_BLOCK * MAX_BLOCKS) ? num_units : WARPS_PER_BLOCK * MAX_BLOCKS;
    const unsigned int num_blocks = DIVIDE_INTO(num_warps, WARPS_PER_BLOCK);
    const unsigned int num_iters  = DIVIDE_INTO(num_units, num_warps);

    const unsigned int interval_size = WARP_SIZE * num_iters;

    const int tail = num_units * WARP_SIZE; // do the last few nonzeros separately (fewer than WARP_SIZE elements)

    const unsigned int active_warps = (interval_size == 0) ? 0 : DIVIDE_INTO(tail, interval_size);

    int *temp_rows;
    double *temp_vals;
    cudaMalloc(&temp_rows, sizeof(int) * active_warps);
    cudaMalloc(&temp_vals, sizeof(double) * active_warps);

    spmv_coo_flat_kernel<BLOCK_SIZE> <<<num_blocks, BLOCK_SIZE, 0, spmv_stream>>>
    (tail, interval_size,
     x, y, val,
     pr_gpu, gpu_result,
     temp_rows, temp_vals);

    spmv_coo_reduce_update_kernel<BLOCK_SIZE> <<<1, BLOCK_SIZE, 0, spmv_stream>>>
    (active_warps, temp_rows, temp_vals, gpu_result);


    //take care of the tail (that doesn't align with warp)
    spmv_coo_normal<<<1,1,0, spmv_stream>>>
    (x + tail, y + tail, val + tail, pr_gpu, gpu_result, num_nonzeros - tail);
}