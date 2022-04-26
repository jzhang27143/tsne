#include <cuda.h>
#include <thrust/device_vector.h>

void compute_attractive_forces(thrust::device_vector<float> &pij,
                               thrust::device_vector<float2> &embed,
                               thrust::device_vector<int> &nn_indices,
                               thrust::device_vector<float2> &grad_attract,
                               int num_points, int k); 
