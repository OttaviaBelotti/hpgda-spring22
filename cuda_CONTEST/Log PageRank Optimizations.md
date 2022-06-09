# Log PageRank Optimizations
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
