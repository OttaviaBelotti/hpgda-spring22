# Log PageRank Optimizations
Run:
```shell
 nvprof bin/b -d -c -n 1000 -b ppr -I 0 -i 1 -t 1024 -B 64 -v
```
## Threads and Blocks Tuning
For now, only one of the kernel (`axpb_personalized_gpu`) makes us of `__shared__` memory which is faster but also requires a bit of synchronization among the threads of the same block. We refer to it as `shared kernel` in the followinf table. 
The number of blocks is always computed as:
```
gridSize = (E + blockSize -1) / blockSize
gridSize_shared = (V + blockSize_shared - 1) / blockSize_shared
```

We performed the following tests and at the 30th iteration we obtain:

|Type   | #threads  |California.mtx time    | California.mtx bandwidth  | Wikipedia.mtx | Wikipedia.mtx bandwidth|
|:--|:---:|:------:|:-----:|:---------:|:-------:|
Uniform | 1024 |?|?| 5904.36 ms | 0.00406479 GB/s|
Uniform | 256 |20.728 ms| 1.15785 GB/s| 5843.14 ms | 0.00410738 GB/s|
Mix | 1024_N + 256_S | 20.474 ms | 1.7205 GB/s | 5777.48 ms | 0.00415406 GB/s|

Other combinations have been tested (e.g. mix 1024_N+512_S or 128_S) but they yield worst results than these.

We tested also the starting point suggested by the function `cudaMaxOccupancy()`, but the max occupancy doesn't result a winning strategy because of the mandatory synchronization among threads and blocks (`atomicAdd()`)

## Dumping Factor (Alpha) Tuning

## Shared Memory
### Statistics with and without shared memory

| Kernel function | #threads| Time California CPU | Time California GPU | Time Wikipedia GPU |
|:----------------|:-----:|:----:|:---------------:|:--------------:|
|with spmv_coo_0| 1024| |26.728 ms | |
|with spmv_coo_0| 256| |25.072 ms | |
|with spmv_coo_1| 256 | 3.76 ms |32.33 ms | |
|with spmv_coo_1| 1024 | 3.586 ms |29.2 ms | |

* Had to downgread temp_res from double to float to be able to allocate the result vector in shared memory.
* Still, even with float, Wikipedia result vector is too large to fit in shared memory

## Segmentation reduction
* Figure out MAX_BLOCKS --> for NVIDIA GeForce GTX 1050 (comput. 6.1) 5 active SM, from CUDA_OCCUPANCY_CALCULATOR we find that we have max 8 ative blocks per SM --> MAX_BLOCKS = 8 * 5 = 40


## Reduce precision
* wikipedia 30 run, 256 threads per block, uniform initialization of PR vector
    - float precision on spmv, half precision on dangling: 100% accuracy, 2055.22 ms avg
    - half precision on spmv: no, accuracy too low
    - float precision on spmv, half on dangling, half on axpb: 100% accuracy, 2038.63 ms
    - all the above  + float precision on euclidean_distance (should also reduce the number of iterations): 100% accuracy, 1915.1 ms
* wikipedia 100 run, 256 threads per block, uniform initialization
    - float precision on spmv, half precision on dangling, float on axpb, float on euclidean_distance: 100% accuracy but one run at 95%, one 85%, one 65%, 1917.38 ms

* wikipedia 100 run, 256 threads per block, non-uniform pr penalty weight 0.1:
    - float precision on spmv, half precision on dangling, float on axpb, float on euclidean_distance: 1 95%, 1 65% (it.57), 1 85%,  1869 ms