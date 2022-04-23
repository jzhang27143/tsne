#include <cuda.h>
#include <thrust/device_vector.h>

void symmetrize_matrix(thrust::device_vector<float> &pij_unsym,
                       thrust::device_vector<float> &pij_sym,
                       thrust::device_vector<int> &nn_indices,
                       int num_points, int k);

void initialize_points(thrust::device_vector<float2> &ys, int num_points); 
