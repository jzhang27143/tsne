#include <thrust/device_vector.h>

void search_perplexity(thrust::device_vector<float> &pij,
                       thrust::device_vector<float> &dist, 
                       const float perplexity_target, const float epsilon,
                       const int num_points, const int k);
