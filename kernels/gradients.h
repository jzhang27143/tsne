#include <cuda.h>
#include <thrust/device_vector.h>

void compute_attractive_forces(thrust::device_vector<float> &pij,
                               thrust::device_vector<float> &embed_x,
                               thrust::device_vector<float> &embed_y,
                               thrust::device_vector<int> &nn_indices,
                               thrust::device_vector<float> &grad_attract_x,
                               thrust::device_vector<float> &grad_attract_y,
                               int num_points, int k);

void compute_repulsive_forces(thrust::device_vector<float> &embed_x,
                              thrust::device_vector<float> &embed_y,
                              thrust::device_vector<float> &grad_repulsive_x,
                              thrust::device_vector<float> &grad_repulsive_y, 
                              int num_points, float theta);

void apply_forces(thrust::device_vector<float> &embed_x,
                  thrust::device_vector<float> &embed_y,
                  thrust::device_vector<float> &gains_x,
                  thrust::device_vector<float> &gains_y,
                  thrust::device_vector<float> &old_forces_x,                 
                  thrust::device_vector<float> &old_forces_y,
                  thrust::device_vector<float> &grad_attract_x,
                  thrust::device_vector<float> &grad_attract_y,
                  thrust::device_vector<float> &grad_repulsive_x,
                  thrust::device_vector<float> &grad_repulsive_y,
                  float learning_rate, float momentum,
                  float exaggeration, int num_points);


