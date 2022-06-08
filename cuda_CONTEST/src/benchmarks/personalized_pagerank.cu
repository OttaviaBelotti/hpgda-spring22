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
#include <algorithm>

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

//////////////////////////////
//////////////////////////////

// Write GPU kernel here!

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

    printf("CHECK COO FORMAT\n");
    for(int i = 0; i < 6; i++){
        printf("%d -> x: %d, y: %d, val: %lf\n", i, x[i], y[i], val[i]);
    }

    // Allocate any GPU data here;
    // TODO!
   
    // Allocate GPU data: cloning x, y and val vectors in GPU global memory
    CHECK(cudaMalloc(&x_d, sizeof(int)*E));
    CHECK(cudaMalloc(&y_d, sizeof(int)*E));
    CHECK(cudaMalloc(&val_d, sizeof(double)*E));
    CHECK(cudaMalloc(&pr_gpu, sizeof(double)*V));  
    CHECK(cudaMalloc(&gpu_result, sizeof(double)*V));
    CHECK(cudaMalloc(&gpu_err, sizeof(double)));

    CHECK(cudaMemcpy(x_d, &x[0], sizeof(int) * x.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(y_d, &y[0], sizeof(int) * y.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(val_d, &val[0], sizeof(double) * val.size(), cudaMemcpyHostToDevice));
    
}

// Initialize data;
void PersonalizedPageRank::init() {
    // Do any additional CPU or GPU setup here;
    // TODO!
    CHECK(cudaMalloc(&dangling_factor_gpu, sizeof(double)));
    CHECK(cudaMalloc(&dangling_bitmap, sizeof(int)*dangling.size()));

}

// Reset the state of the computation after every iteration.
// Reset the result, and transfer data to the GPU if necessary;
void PersonalizedPageRank::reset() {
    // Reset the PageRank vector (uniform initialization, 1 / V for each vertex);
    std::fill(pr.begin(), pr.end(), 1.0 / V); 
    // Generate a new personalization vertex for this iteration;
    personalization_vertex = rand() % V; 
    if (debug) std::cout << "personalization vertex=" << personalization_vertex << std::endl;

    CHECK(cudaMemcpy(pr_gpu, &pr[0], sizeof(double) * pr.size(), cudaMemcpyHostToDevice));

    // initialize GPU result array with all 0s
    CHECK(cudaMemset(gpu_result, 0.0, sizeof(double)*V));

    // Do any GPU reset here, and also transfer data to the GPU;
    // TODO!

    CHECK(cudaMemcpy(dangling_bitmap, dangling.data(), sizeof(int)*dangling.size(), cudaMemcpyHostToDevice));

    //compute the out_degree and the in_degree of the personalization_vertex
    int in_degree = 0, out_degree = 0;
    for(int i=0; i < E; i++){
        if(x[i] == personalization_vertex)
            in_degree++; //because in COO we have row/cols transposed
        if(y[i] == personalization_vertex)
            out_degree++;
    }
    printf("in_degree: %d\tout_degree: %d\n", in_degree, out_degree);

    //if the personalization vector is lowly connected, then we lower the alpha
    // if(in_degree < 0.001*V || out_degree < 0.001*V)
    //     custom_alpha = 0.25;
    // else
    //     custom_alpha = DEFAULT_ALPHA;
    printf("custom alpha: %lf\n", custom_alpha);
}


__global__ void spmv_coo_v3(const int num_vals, int *row_ids, int *col_ids, double *values, int num_rows, double *x, double *y) {
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += blockDim.x * gridDim.x) {
        if(i < num_vals){
            atomicAdd(y+row_ids[i], values[i]*x[col_ids[i]]);
        }
    }
}

/**
 * @brief Parallel GPU version of matrix-vector multiplication.
 * 
 * @param x row indices (COO format matrix - vector)
 * @param y column indices (COO format matrix - vector)
 * @param val matrix values (COO format matrix - vector)
 * @param vec vector
 * @param result vector for result of the multiplication
 * @param N vector dimension
 */
__global__ void spmv_coo_final(const int *x, const int *y, const double *val, const double *vec, double *result, int N) {
    // Uses a grid-stride loop to perform dot product
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        atomicAdd(result + x[i], val[i] * vec[y[i]]);
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
__global__ void compute_dangling_factor_gpu(const int *a, const double *b, const int N, double *result){
    // using a share temp_result might speed up but we have problems in sync. the blocks

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x){
        atomicAdd(result, a[i] * b[i]);
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
    __shared__ double one_minus_alpha;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadIdx.x == 0)
        one_minus_alpha = 1 - alpha;
    
    __syncthreads();

    for(; i < N; i+= blockDim.x * gridDim.x){
        result[i] = alpha * x[i] + beta + ((personalization_vertex == i) ? one_minus_alpha : 0.0);
    }
}


__global__ void axpb_personalized_gpu_v2(double alpha, double *x, int *dang_vector, double beta, const int personalization_vertex, double *result, const int N){
    __shared__ double one_minus_alpha;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i == 0)
        one_minus_alpha = 1 - alpha;
    
    __syncthreads();

    for(; i < N; i+= blockDim.x * gridDim.x){
        result[i] = alpha * x[i] + beta * dang_vector[i] + ((personalization_vertex == i) ? one_minus_alpha : 0.0);
    }
}

/**
 * @brief GPU parallelized version for euclidean distance
 * 
 * @param x vector of x (row) coordinates
 * @param y vecotr of y (column) coordinates
 * @param N dimension of the vectors
 * @param result pointer to the computed distance result
 */
__global__ void euclidean_distance_gpu(const double *x, const double *y, const int N, double *result) {
    //can be improved

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        atomicAdd(result, (x[i] - y[i]) * (x[i] - y[i]));
    }
}

void PersonalizedPageRank::execute(int iter) {
    // Do the GPU computation here, and also transfer results to the CPU;

    dim3 blocksPerGrid((E + blocksize - 1) / blocksize, 1, 1);
    dim3 threadsPerBlock(blocksize, 1, 1);
    
    double *temp;
    double dangling_factor_val;
    double *prev_pr = (double *) malloc(sizeof(double) * V); // previous values of pr
    double *curr_pr = (double *) malloc(sizeof(double) * V); // current values of pr
    double *err = (double *) malloc(sizeof(double));         // convergence error

    printf("blocksPerGrid: %d\tthreadsPerBlock:%d\n", (E + blocksize - 1) / blocksize, blocksize);

    int number_of_iterations = 0;
    bool conv = false;
    while (!conv && number_of_iterations < DEFAULT_MAX_ITER) {
        CHECK(cudaMemset(gpu_err, 0.0, sizeof(double)));           // reset error 
        CHECK(cudaMemset(gpu_result, 0.0, sizeof(double) * V));    // reset GPU result
        cudaMemset(dangling_factor_gpu, 0.0, sizeof(double));  //reset dangling factor

        spmv_coo_final<<<blocksPerGrid, threadsPerBlock>>>(x_d, y_d, val_d, pr_gpu, gpu_result, E);
        CHECK_KERNELCALL();
        CHECK(cudaDeviceSynchronize());

        compute_dangling_factor_gpu<<<blocksPerGrid, threadsPerBlock>>>(dangling_bitmap, pr_gpu, V, dangling_factor_gpu);
        CHECK_KERNELCALL();
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(&dangling_factor_val, dangling_factor_gpu, sizeof(double), cudaMemcpyDeviceToHost));

        axpb_personalized_gpu<<<blocksPerGrid, threadsPerBlock>>>(custom_alpha, gpu_result, custom_alpha * dangling_factor_val / V, personalization_vertex, gpu_result, V);
        CHECK_KERNELCALL();
        CHECK(cudaDeviceSynchronize());
        
        // Check convergence
        euclidean_distance_gpu<<<blocksPerGrid, threadsPerBlock>>>(pr_gpu, gpu_result, V, gpu_err);
        CHECK_KERNELCALL();
        CHECK(cudaDeviceSynchronize());

        cudaMemcpy(err, gpu_err, sizeof(double), cudaMemcpyDeviceToHost);
        *err = std::sqrt((float) *err);
        conv = *err <= convergence_threshold;

        temp = pr_gpu;
        pr_gpu = gpu_result;
        gpu_result = temp; 

        number_of_iterations++;
    }

    // save the GPU PPR values into the "pr" array
        CHECK(cudaMemcpy(&pr[0], pr_gpu, sizeof(double) * V, cudaMemcpyDeviceToHost));
    //printf("NUMBER OF GPU ITERATIONS: %d\n", number_of_iterations);
    free(prev_pr);
    free(curr_pr);
    free(err);
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
    std::unordered_set<int> top_pr_indices;
    std::unordered_set<int> top_pr_golden_indices;
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
}
