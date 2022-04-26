#include <cuda.h>
#include "thrust/device_vector.h"
#include "gradients.h"

__global__ void kernel_attractive_forces(const float *__restrict__ pij,
                                         const float2 *__restrict__ embed,
                                         const int *__restrict__ nn_indices,
                                         float2 *__restrict__ grad_attract,
                                         int num_points, int k) {
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) {
        return;
    }

    float2 grad_attr_i = make_float2(0.f, 0.f);
    // Add attractive gradients for all neighbors of point i
    for (int n_index = 0; n_index < k; n_index++) {
        int j = nn_indices[i * k + n_index];
        float dy1 = embed[i].x - embed[j].x;
        float dy2 = embed[i].y - embed[j].y;
        grad_attr_i.x += pij[i * k + n_index] * dy1 / (1 + dy1 * dy1 + dy2 * dy2);
        grad_attr_i.y += pij[i * k + n_index] * dy2 / (1 + dy1 * dy1 + dy2 * dy2);
    }
    grad_attract[i] = grad_attr_i;
}

void compute_attractive_forces(thrust::device_vector<float> &pij,
                               thrust::device_vector<float2> &embed,
                               thrust::device_vector<int> &nn_indices,
                               thrust::device_vector<float2> &grad_attract,
                               int num_points, int k) {
    const int BLOCKSIZE = 1024;
    const int NBLOCKS = (num_points + BLOCKSIZE - 1) / BLOCKSIZE;

    kernel_attractive_forces<<<NBLOCKS, BLOCKSIZE>>>(
        thrust::raw_pointer_cast(pij.data()),
        thrust::raw_pointer_cast(embed.data()),
        thrust::raw_pointer_cast(nn_indices.data()),
        thrust::raw_pointer_cast(grad_attract.data()),
        num_points, k
    );
}
