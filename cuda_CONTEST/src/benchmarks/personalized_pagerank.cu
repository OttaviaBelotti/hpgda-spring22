// Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NECSTLab nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//  * Neither the name of Politecnico di Milano nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <sstream>
#include "personalized_pagerank.cuh"
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include "spmv_seg.cu"
#include <cuda_fp16.h>

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

//////////////////////////////
//////////////////////////////

// Write GPU kernel here!

/**
 * @brief GPU function that perform the warp reduction.
 * 
 * @param input input array to reduce
 * @param threadId index of the thread
 */
__inline__ __device__ void warp_reduction(volatile float *input, int threadId)
// we need volatile flag here, otherwise the compiler might introduce some optimizations in the "input" variable
// and place it in registers instead of shared memory!
{
	input[threadId] += input[threadId + 32];
	input[threadId] += input[threadId + 16];
	input[threadId] += input[threadId + 8];
	input[threadId] += input[threadId + 4];
	input[threadId] += input[threadId + 2];
	input[threadId] += input[threadId + 1];
}

/**
 * @brief GPU function that compute the final parallel reduction for the collected data.
 * 
 * @param input input array to reduce
 */
__device__ void collect_res_gpu(float *input)
{
    int i, threadId = threadIdx.x;

    for (i = blockDim.x / 2; i > 32; i >>= 1) 
    {
        if (threadId < i)
        {
            input[threadId] += input[threadId + i];
        }
        __syncthreads();
    }

    if(threadId<32)
        warp_reduction(input,threadId);
    __syncthreads();
}

__device__ int binary_search_element(const int elem, const int *arr, int low, int high){
    int mid;
    while( low != high ){
        mid = (low + high)/2;
        if (elem == arr[mid])
        return mid;

        else if (elem > arr[mid])   // elem is on the right side
            low = mid + 1;

        else                        // elem is on the left side
            high = mid - 1;
    }
    return -1; //not found
}

/**
 * @brief Parallel GPU version of matrix-vector multiplication, uses a grid-stride loop to perform the dot product.
 * 
 * @param x row indices (COO format matrix - vector)
 * @param y column indices (COO format matrix - vector)
 * @param val matrix values (COO format matrix - vector)
 * @param vec vector
 * @param result vector for result of the multiplication
 * @param N vector dimension
 */
__global__ void spmv_coo(const int *x, const int *y, const double *val, const double *vec, double *result, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        atomicAdd(result + x[i], (float)val[i] * (float)vec[y[i]]);
    }
}

/**
 * @brief Computation for the dangling factor.
 * 
 * @param a dangling seed vector
 * @param b page ranking vector
 * @param N vector size
 * @param result pointer to the dangling factor (where to place the result of the function)
 */
__global__ void compute_dangling_factor_gpu(const int *a, const double *b, const int N, float *result){
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x){
        atomicAdd(result, a[i] * b[i]);
    }
}

/**
 * @brief Computation for the dangling factor with parallel reduction.
 * 
 * @param a dangling seed vector
 * @param b page ranking vector
 * @param N vector size
 * @param result pointer to the dangling factor (where to place the result of the function)
 */
__global__ void compute_dangling_factor_gpu_reduction(const int *a, const double *b, const int N, float *result){
    // using a share temp_result might speed up but we have problems in sync. the blocks
    extern __shared__ float temp[];
    //half a_h, b_h;
    temp[threadIdx.x]=0.0;
    __syncthreads();

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        temp[threadIdx.x] = (float)a[idx] * (float)b[idx];
    }
    __syncthreads();

    collect_res_gpu(temp);
    if (threadIdx.x == 0) {
        atomicAdd(result, temp[0]);
    }
}

/**
 * @brief Final formula for PR
 * 
 * @param alpha damping factor
 * @param x vector of intermediate pr values
 * @param beta damping factor: alpha * damping_factor / V
 * @param result final vector of the PR value results
 * @param N vectors dimension
*/
__global__ void axpb_personalized_gpu(double alpha, double *x, double beta, const int personalization_vertex, double *result, const int N){
    float one_minus_alpha;
    float alpha_h = (float) alpha;
    float beta_h = (float) beta;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    one_minus_alpha = 1 - alpha_h;

    for(; i < N; i+= blockDim.x * gridDim.x){
        result[i] = alpha_h * x[i] + beta_h + ((personalization_vertex == i) ? one_minus_alpha : (float)0.0);
    }
}

/**
 * @brief GPU parallelized version for euclidean distance
 * 
 * @param x vector of x (row) coordinates
 * @param y vector of y (column) coordinates
 * @param N dimension of the vectors
 * @param result pointer to the computed distance result
 */
__global__ void euclidean_distance_gpu(const double *x, const double *y, const int N, double *result) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        atomicAdd(result, (x[i] - y[i]) * (x[i] - y[i]));
    }
}

/**
 * @brief GPU parallelized version for euclidean distance with parallel reduction
 * 
 * @param x vector of x (row) coordinates
 * @param y vector of y (column) coordinates
 * @param N dimension of the vectors
 * @param result pointer to the computed distance result
 */
__global__ void euclidean_distance_gpu_reduction(const double* x , const double* y , const int N, double* result/*, bool *excluded_pages*/) {
    extern __shared__ float temp[];
    temp[threadIdx.x]=0;
    __syncthreads();
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float var;

    if (idx < N) {
        var = (float)x[idx] - (float)y[idx];
        temp[threadIdx.x] = var * var;
    }
    __syncthreads();
    
    collect_res_gpu(temp);
    if (threadIdx.x == 0) {
        atomicAdd(result, temp[0]);
    }
}

__global__ void spmv_coo_1(const int *x, const int *y, const double *val, const double *vec, double *result, const int N, const int res_size, int *shrinked_x){
    extern __shared__ int s[]; //must be initialized with N elements from the caller

    int *temp_idx = s;
    double *temp_res = (double*)&temp_idx[res_size];
    //int *last_idx = (int*)&temp_res[res_size];
    int binary_search_res;


    for (int i = threadIdx.x, j = blockIdx.x * blockDim.x + threadIdx.x; i < res_size; i += blockDim.x, j += blockDim.x * gridDim.x){ 
        temp_res[i] = 0.0;
        temp_idx[i] =shrinked_x[j];
    }

    __syncthreads();

    // Uses a grid-stride loop to perform dot product
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        binary_search_res = binary_search_element(x[i], temp_idx, 0, res_size-1);
        atomicAdd_block(temp_res + binary_search_res, (val[i] * vec[y[i]])); //only thread-safe, not block-safe
    }

    __syncthreads();
    //riassamble all the partial results: the speedup should be that we're doing num_blocks*V atomicAdd() instead of num_blocks*E atomicAdd()
    for(int i=threadIdx.x; i < res_size; i += blockDim.x){
        atomicAdd(&result[temp_idx[i]], temp_res[i]);
    }

}

//////////////////////////////
//////////////////////////////

// CPU Utility functions;

#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            printf("Err: %s\n", cudaGetErrorName(err));                                   \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

// Read the input graph and initialize it;
void PersonalizedPageRank::initialize_graph() {
    // Read the graph from an MTX file;
    int num_rows = 0;
    int num_columns = 0;
    read_mtx(graph_file_path.c_str(), &x, &y, &val,
        &num_rows, &num_columns, &E, // Store the number of vertices (row and columns must be the same value), and edges;
        true,                        // If true, read edges TRANSPOSED, i.e. edge (2, 3) is loaded as (3, 2). We set this true as it simplifies the PPR computation;
        false,                       // If true, read the third column of the matrix file. If false, set all values to 1 (this is what you want when reading a graph topology);
        debug,                 
        false,                       // MTX files use indices starting from 1. If for whatever reason your MTX files uses indices that start from 0, set zero_indexed_file=true;
        true                         // If true, sort the edges in (x, y) order. If you have a sorted MTX file, turn this to false to make loading faster;
    );
    if (num_rows != num_columns) {
        if (debug) std::cout << "error, the matrix is not squared, rows=" << num_rows << ", columns=" << num_columns << std::endl;
        exit(-1);
    } else {
        V = num_rows;
    }
    if (debug) std::cout << "loaded graph, |V|=" << V << ", |E|=" << E << std::endl;

    // Compute the dangling vector. A vertex is not dangling if it has at least 1 outgoing edge;
    dangling.resize(V);
    std::fill(dangling.begin(), dangling.end(), 1);  // Initially assume all vertices to be dangling;
    for (int i = 0; i < E; i++) {
        // Ignore self-loops, a vertex is still dangling if it has only self-loops;
        if (x[i] != y[i]) dangling[y[i]] = 0;
    }
    // Initialize the CPU PageRank vector;
    pr.resize(V);
    pr_golden.resize(V);
    // Initialize the value vector of the graph (1 / outdegree of each vertex).
    // Count how many edges start in each vertex (here, the source vertex is y as the matrix is transposed);
    int *outdegree = (int *) calloc(V, sizeof(int));
    for (int i = 0; i < E; i++) {
        outdegree[y[i]]++;
    }
    // Divide each edge value by the outdegree of the source vertex;
    for (int i = 0; i < E; i++) {
        val[i] = 1.0 / outdegree[y[i]];  
    }
    free(outdegree);
}

//////////////////////////////
//////////////////////////////

// Allocate data on the CPU and GPU;
void PersonalizedPageRank::alloc() {
    // Load the input graph and preprocess it;
    initialize_graph(); //CPU loading data

    // Allocate any GPU data here;
    // TODO!

    // Allocate GPU data: cloning x, y and val vectors in GPU global memory
    CHECK(cudaMalloc(&x_d, sizeof(int)*E));
    CHECK(cudaMalloc(&y_d, sizeof(int)*E));
    CHECK(cudaMalloc(&val_d, sizeof(double)*E));
    CHECK(cudaMalloc(&pr_gpu, sizeof(double)*V));  
    CHECK(cudaMalloc(&gpu_result, sizeof(double)*V));
    CHECK(cudaMalloc(&gpu_err, sizeof(double)));
    CHECK(cudaMalloc(&dangling_factor_gpu, sizeof(float)));
    CHECK(cudaMalloc(&dangling_bitmap, sizeof(int)*dangling.size()));

    CHECK(cudaMemcpy(x_d, &x[0], sizeof(int) * x.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(y_d, &y[0], sizeof(int) * y.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(val_d, &val[0], sizeof(double) * val.size(), cudaMemcpyHostToDevice));
    
}

// Initialize data;
void PersonalizedPageRank::init() {
    // Do any additional CPU or GPU setup here;
    // TODO!
    
    int last_idx = -1;
    int last_idx_qty = 0;

    std::vector<std::pair<int, int>> in_degree;
    in_degree.resize(V);

    //assumption: E > V
    for(int i=0; i<E; i++){
        //compute in_degree for each page: if page i not it x[i] --> in_degree[i] = 0
        if(x[i] != last_idx){
            if(last_idx >= 0){
                in_degree[last_idx].first = last_idx_qty;
                in_degree[last_idx].second = last_idx;
            }

            last_idx++;
            last_idx_qty = (x[i] != last_idx) ? 0 : 1;

        }else{
            last_idx_qty++;
        }
    }
    in_degree[last_idx].first = last_idx_qty; // finish writing the qty for the last page
    in_degree[last_idx].second = last_idx;

    while(last_idx < V){
        // last pages are not in x, so finish up in_degree with zeros
        last_idx++;
        in_degree[last_idx].first = 0;
        in_degree[last_idx].second = last_idx;
    }

    auto greater_key = [](std::pair<int,int> e1, std::pair<int,int> e2){
        return e1.first > e2.first;
        };

    // sort in_degree_ranked
    // key: in_degree   values: in_degree_ranked    comparer: reverting
    sort(in_degree.begin(), in_degree.end(), greater_key);


    int precise_vertex_qty = (int)(V * heuristic_precision);
    std::cout << "Precision: " << heuristic_precision << ", #precise vertex: " << precise_vertex_qty << std::endl;

    //excluded_pages_cpu.resize(V);
    excluded_pages_cpu = (bool*)malloc(sizeof(bool) * V);
  
    //std::cout << "Excluded pages:" << std::endl;
    for(int i=0; i<V; i++){
        excluded_pages_cpu[in_degree[i].second] = (i< precise_vertex_qty && in_degree[i].first != 0) ? false : true;
    }

    num_effective_vertex = std::count(excluded_pages_cpu, &excluded_pages_cpu[V-1], false);
}

// Reset the state of the computation after every iteration.
// Reset the result, and transfer data to the GPU if necessary;
void PersonalizedPageRank::reset() {
    // Reset the PageRank vector (uniform initialization, 1 / V for each vertex)
    if(implementation == 0){
        std::fill(pr.begin(), pr.end(), 1.0 / V); 
    }
    // Reset the PageRank vector (non-uniform initialization)
    else {
        double penalty_weight = 0.1;
        float weighted_sum = num_effective_vertex * (1 - penalty_weight) + (V - num_effective_vertex) * penalty_weight;

        for(int i=0; i<V; i++){
            if(excluded_pages_cpu[i]){
                pr[i] = penalty_weight * (1/weighted_sum);
            }else{
                pr[i] = (1-penalty_weight) * (1/weighted_sum);
            }
        }
    }
    
    // Generate a new personalization vertex for this iteration;
    personalization_vertex = rand() % V; 
    if (debug) std::cout << "personalization vertex=" << personalization_vertex << std::endl;

    // Do any GPU reset here, and also transfer data to the GPU;
    // TODO!

    CHECK(cudaMemcpy(pr_gpu, &pr[0], sizeof(double) * pr.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(gpu_result, 0.0, sizeof(double)*V));   // initialize GPU result array with all 0s
    CHECK(cudaMemcpy(dangling_bitmap, dangling.data(), sizeof(int)*dangling.size(), cudaMemcpyHostToDevice));
}

/**
 * @brief First implementation, plain CPU-to-GPU CUDA transposition.
 * 
 * @param iter iteration number
*/
void PersonalizedPageRank::personalized_pagerank_0(int iter){
    auto start_tmp = clock_type::now();
 
    int blockSize = block_size; // take block size from option -t
    int gridSize = (E + blockSize - 1) / blockSize;

    dim3 blocksPerGrid(gridSize, 1, 1);
    dim3 threadsPerBlock(blockSize, 1, 1);

    double *temp;
    float dangling_factor_val;
    double *err = (double *) malloc(sizeof(double)); 

    int number_of_iterations = 0;
    bool conv = false;
    while (!conv && number_of_iterations < max_iterations) {
        CHECK(cudaMemset(gpu_result, 0.0, sizeof(double) * V));    // reset GPU result
        CHECK(cudaMemset(gpu_err, 0.0, sizeof(double)));           // reset error 
        cudaMemset(dangling_factor_gpu, 0.0, sizeof(float));       // reset dangling factor

        spmv_coo<<<blocksPerGrid, threadsPerBlock>>>(x_d, y_d, val_d, pr_gpu, gpu_result, E);
        CHECK_KERNELCALL();

        compute_dangling_factor_gpu<<<blocksPerGrid, threadsPerBlock>>>(dangling_bitmap, pr_gpu, V, dangling_factor_gpu);
        CHECK_KERNELCALL();

        CHECK(cudaMemcpy(&dangling_factor_val, dangling_factor_gpu, sizeof(float), cudaMemcpyDeviceToHost));

        axpb_personalized_gpu<<<blocksPerGrid, threadsPerBlock>>>(custom_alpha, gpu_result, custom_alpha * dangling_factor_val / V, personalization_vertex, gpu_result, V);
        CHECK_KERNELCALL();
        
        // Check convergence
        euclidean_distance_gpu<<<blocksPerGrid, threadsPerBlock>>>(pr_gpu, gpu_result, V, gpu_err);
        CHECK_KERNELCALL();

        cudaMemcpy(err, gpu_err, sizeof(double), cudaMemcpyDeviceToHost);
        *err = std::sqrt((double) *err);
        conv = *err <= convergence_threshold;

        temp = pr_gpu;
        pr_gpu = gpu_result;
        gpu_result = temp; 

        number_of_iterations++;
    }

    CHECK(cudaDeviceSynchronize());

    if (debug) {
        // Synchronize computation by hand to measure GPU exec. time;
        auto end_tmp = clock_type::now();
        auto exec_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
        std::cout << "  pure GPU execution(" << iter << ")=" << double(exec_time) / 1000 << " ms, " << (3 * sizeof(double) * N * N / (exec_time * 1e3)) << " GB/s" << std::endl;
    }

    // save the GPU PPR values into the "pr" array
    CHECK(cudaMemcpy(&pr[0], pr_gpu, sizeof(double) * V, cudaMemcpyDeviceToHost));
}

/**
 * @brief Second implementation, final optimized version.
 * 
 * @param iter iteration number
*/
void PersonalizedPageRank::personalized_pagerank_1(int iter){
    auto start_tmp = clock_type::now();
 
    int blockSize = block_size; // take block size from option -t
    int gridSize = (E + blockSize - 1) / blockSize;

    dim3 blocksPerGrid(gridSize, 1, 1);
    dim3 threadsPerBlock(blockSize, 1, 1);

    double *temp;
    float dangling_factor_val;
    double *err = (double *) malloc(sizeof(double));   // convergence error

    cudaStream_t spmv_stream, dangling_factor_stream;
    cudaStreamCreate(&spmv_stream);
    cudaStreamCreate(&dangling_factor_stream);

    int number_of_iterations = 0;
    bool conv = false;
    while (!conv && number_of_iterations < max_iterations) {
        CHECK(cudaMemset(gpu_result, 0.0, sizeof(double) * V));      // reset GPU result
        CHECK(cudaMemset(gpu_err, 0.0, sizeof(double)));             // reset error 

        spmv_coo<<<blocksPerGrid, threadsPerBlock, 0 , spmv_stream>>>(x_d, y_d, val_d, pr_gpu, gpu_result, E);
        CHECK_KERNELCALL();

        compute_dangling_factor_gpu_reduction<<<blocksPerGrid, threadsPerBlock, blockSize * sizeof(float), dangling_factor_stream>>>(dangling_bitmap, pr_gpu, V, dangling_factor_gpu);
        CHECK_KERNELCALL();

        CHECK(cudaMemcpy(&dangling_factor_val, dangling_factor_gpu, sizeof(float), cudaMemcpyDeviceToHost));

        axpb_personalized_gpu<<<blocksPerGrid, threadsPerBlock>>>(custom_alpha, gpu_result, custom_alpha * dangling_factor_val / V, personalization_vertex, gpu_result, V);
        CHECK_KERNELCALL();

        CHECK(cudaMemsetAsync(dangling_factor_gpu, 0.0, sizeof(float), dangling_factor_stream));  // asynchronously reset the dangling factor on the GPU
        
        // Check convergence
        euclidean_distance_gpu_reduction<<<blocksPerGrid, threadsPerBlock, blockSize * sizeof(float)>>>(pr_gpu, gpu_result, V, gpu_err/*, excluded_pages_gpu*/);
        CHECK_KERNELCALL();

        cudaMemcpy(err, gpu_err, sizeof(double), cudaMemcpyDeviceToHost);
        *err = std::sqrt((double) *err);
        conv = *err <= convergence_threshold;

        temp = pr_gpu;
        pr_gpu = gpu_result;
        gpu_result = temp; 

        number_of_iterations++;
    }

    CHECK(cudaDeviceSynchronize());

    if (debug) {
        // Synchronize computation by hand to measure GPU exec. time;
        auto end_tmp = clock_type::now();
        auto exec_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
        std::cout << "  pure GPU execution(" << iter << ")=" << double(exec_time) / 1000 << " ms, " << (3 * sizeof(double) * N * N / (exec_time * 1e3)) << " GB/s" << std::endl;
    }

    // destroy the streams
    cudaStreamDestroy(spmv_stream);
    cudaStreamDestroy(dangling_factor_stream);

    // save the GPU PPR values into the "pr" array
    CHECK(cudaMemcpy(&pr[0], pr_gpu, sizeof(double) * V, cudaMemcpyDeviceToHost));
}

void PersonalizedPageRank::execute(int iter) {
    // Do the GPU computation here, and also transfer results to the CPU;
    switch (implementation)
    {
    case 0:
        personalized_pagerank_0(iter); // first implementation
        break;
    case 1:
        personalized_pagerank_1(iter); // second implementation
        break;
    default:
        break;
    }
}

void PersonalizedPageRank::cpu_validation(int iter) {

    // Reset the CPU PageRank vector (uniform initialization, 1 / V for each vertex);
    std::fill(pr_golden.begin(), pr_golden.end(), 1.0 / V);

    // Do Personalized PageRank on CPU;
    auto start_tmp = clock_type::now();
    personalized_pagerank_cpu(x.data(), y.data(), val.data(), V, E, pr_golden.data(), dangling.data(), personalization_vertex, alpha, 1e-6, 100);
    auto end_tmp = clock_type::now();
    auto exec_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
    std::cout << "exec time CPU=" << double(exec_time) / 1000 << " ms" << std::endl;

    // Obtain the vertices with highest PPR value;
    std::vector<std::pair<int, double>> sorted_pr_tuples = sort_pr(pr.data(), V);
    std::vector<std::pair<int, double>> sorted_pr_golden_tuples = sort_pr(pr_golden.data(), V);

    // Check how many of the correct top-20 PPR vertices are retrieved by the GPU;
    std::set<int> top_pr_indices;
    std::set<int> top_pr_golden_indices;
    int old_precision = std::cout.precision();
    std::cout.precision(4);
    int topk = std::min(V, topk_vertices);
    for (int i = 0; i < topk; i++) {
        int pr_id_gpu = sorted_pr_tuples[i].first;
        int pr_id_cpu = sorted_pr_golden_tuples[i].first;
        top_pr_indices.insert(pr_id_gpu);
        top_pr_golden_indices.insert(pr_id_cpu);
        if (debug) {
            double pr_val_gpu = sorted_pr_tuples[i].second;
            double pr_val_cpu = sorted_pr_golden_tuples[i].second;
            if (pr_id_gpu != pr_id_cpu) {
                std::cout << "* error in rank! (" << i << ") correct=" << pr_id_cpu << " (val=" << pr_val_cpu << "), found=" << pr_id_gpu << " (val=" << pr_val_gpu << ")" << std::endl;
            } else if (std::abs(sorted_pr_tuples[i].second - sorted_pr_golden_tuples[i].second) > 1e-6) {
                std::cout << "* error in value! (" << i << ") correct=" << pr_id_cpu << " (val=" << pr_val_cpu << "), found=" << pr_id_gpu << " (val=" << pr_val_gpu << ")" << std::endl;
            }
        }
    }
    std::cout.precision(old_precision);
    // Set intersection to find correctly retrieved vertices;
    std::vector<int> correctly_retrieved_vertices;
    set_intersection(top_pr_indices.begin(), top_pr_indices.end(), top_pr_golden_indices.begin(), top_pr_golden_indices.end(), std::back_inserter(correctly_retrieved_vertices));
    precision = double(correctly_retrieved_vertices.size()) / topk;
    if (debug) std::cout << "correctly retrived top-" << topk << " vertices=" << correctly_retrieved_vertices.size() << " (" << 100 * precision << "%)" << std::endl;
}

std::string PersonalizedPageRank::print_result(bool short_form) {
    if (short_form) {
        return std::to_string(precision);
    } else {
        // Print the first few PageRank values (not sorted);
        std::ostringstream out;
        out.precision(3);
        out << "[";
        for (int i = 0; i < std::min(20, V); i++) {
            out << pr[i] << ", ";
        }
        out << "...]";
        return out.str();
    }
}

void PersonalizedPageRank::clean() {
    // Delete any GPU data or additional CPU data;
    // TODO!
    
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(val_d);
    cudaFree(pr_gpu);
    cudaFree(gpu_result);
    cudaFree(gpu_err);
    cudaFree(dangling_factor_gpu);
    cudaFree(dangling_bitmap);


    free(excluded_pages_cpu);
}
