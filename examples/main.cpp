#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "log.h"
#include "cudaLib/cuda_func.h"
#include "commLib/Communicator.h"

int main(int argc, char *argv[]) {
    int64_t N = 1<<28;
    float initVal = 2.0;
    float *x = nullptr;
    float *y = nullptr;
    Communicator comm;

    check(argc==4, "Usage: %s server_ip num_nodes port_num", argv[0]);

    check(comm.init(argv[1], argv[2], argv[3])==true,
    			"failed to initialize communicator");
    
    gpu_mem_alloc(&x, N);
    check(x != nullptr, "failed to call cudaMallocManaged on x");

    gpu_mem_alloc(&y, N);
    check(y != nullptr, "failed to call cudaMallocManaged on y");
    
    gpu_init(N, x, initVal);
    
    comm.all_reduce(x, y, N);
    std::cout << "all reduce done" << std::endl;

    if (comm.get_rank() == 0) {
	float targetVal = initVal * atoi(argv[2]);
	for (int64_t i = 0; i < N; ++i) {
	    if (y[i] != targetVal) {
		std::cout << i << ": " << y[i] << std::endl;
	    }
	}
    }
    std::cout << "all reduce done" << std::endl;
    
    cudaFree(x);
    cudaFree(y);
    return 0;

error:
    if (x != nullptr) {
	cudaFree(x);
    }
    if (y != nullptr) {
	cudaFree(y);
    }
    return -1;
}
