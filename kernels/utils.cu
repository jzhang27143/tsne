#include "utils.h"
#include "cfloat"
#include "math.h"
#include "random"
#include "stdio.h"
#include "thrust/device_vector.h"
#include <vector>

__global__ void kernel_symmetrize_matrix(volatile float *__restrict__ pij_sym,
                                         const float *__restrict__ pij_unsym,
                                         const int *__restrict__ nn_indices, 
                                         int num_points, int k) {
    register int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index >= k * num_points) {
        return;
    }

    int i = index / k;
    int j = nn_indices[index];

    pij_sym[index] = pij_unsym[index];
    for (int idx = j*k; idx < j*k + k; idx++) {
        if (nn_indices[idx] == i) {
            pij_sym[index] += pij_unsym[idx];
        }
    }
    
    pij_sym[index] /= (2 * num_points);
}

void symmetrize_matrix(thrust::device_vector<float> &pij_unsym,
                       thrust::device_vector<float> &pij_sym,
                       thrust::device_vector<int> &nn_indices,
                       int num_points, int k) {
    const int BLOCKSIZE = 1024;
    const int NBLOCKS = (num_points * k + BLOCKSIZE - 1) / BLOCKSIZE;

    kernel_symmetrize_matrix<<<NBLOCKS, BLOCKSIZE>>>(
        thrust::raw_pointer_cast(pij_unsym.data()), 
        thrust::raw_pointer_cast(pij_sym.data()),
        thrust::raw_pointer_cast(nn_indices.data()),
        num_points, k
    ); 
}

void initialize_points(thrust::device_vector<float> &embed_x,
                       thrust::device_vector<float> &embed_y, int num_points) {
    thrust::host_vector<float> host_embed_x(num_points);
    thrust::host_vector<float> host_embed_y(num_points);

    std::default_random_engine generator(15618);
    generator.seed(4);
    std::normal_distribution<float> norm_dist(0.0, 0.0001);
    for (int i = 0; i < num_points; i++) {
        host_embed_x[i] = norm_dist(generator);
        host_embed_y[i] = norm_dist(generator);
    }

    thrust::copy(host_embed_x.begin(), host_embed_x.end(), embed_x.begin());   
    thrust::copy(host_embed_y.begin(), host_embed_y.end(), embed_y.begin());
}
