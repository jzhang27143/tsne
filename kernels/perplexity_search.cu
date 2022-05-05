// For pij initialization, we borrow code from https://github.com/CannyLab/tsne-cuda
#include "perplexity_search.h"
#include <cfloat>
#include <math.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <vector>

struct functional_entropy {
    __host__ __device__ float operator()(const float x) const {
        float val = x * log(x);
        return (val != val || isinf(val)) ? 0 : val;
    }
};

__global__ void kernel_reduce_sum(volatile float *__restrict__ row_sum,
                                  float *__restrict__ pij,
                                  const unsigned int num_points,
                                  const unsigned int k,
                                  const float alpha) {
    register int index = threadIdx.x + blockIdx.x * blockDim.x;
    register float cur_row_sum = 0.f;
    
    if (index >= num_points) {
        return;
    }
    
    for (int i = 0; i < k; i++) {
        cur_row_sum += pij[index * k + i];
    }
    row_sum[index] = alpha * cur_row_sum;
}

__global__ void kernel_matrix_vector_div(volatile float *__restrict__ row_sum,
                                         float *__restrict__ pij,
                                         const unsigned int num_points,
                                         const unsigned int k) {
    register int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num_points) {
        return;
    }

    register float total = row_sum[index];
    for (int i = 0; i < k; i++) {
        pij[index * k + i] /= total;
    }
}

__global__ void kernel_compute_pij(volatile float *__restrict__ pij,
                                   const float *__restrict__ dists,
                                   const float *__restrict__ betas,
                                   const unsigned int num_points,
                                   const unsigned int k) {
    register int index, i;
    register float dist, beta;

    index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num_points * k) {
        return;
    }

    i = index / k;
    beta = betas[i];
    dist = dists[index] * dists[index];

    pij[index] = __expf(-beta * dist);
}


__global__ void kernel_perplexity_search(volatile float *__restrict__ betas,
                                         volatile float *__restrict__ lower_bound,
                                         volatile float *__restrict__ upper_bound,
                                         volatile int *__restrict__ found,
                                         const float *__restrict__ neg_entropy,
                                         const float *__restrict__ row_sum,
                                         const float perplexity_target,
                                         const float epsilon,
                                         const int num_points) {
    register int i, is_found;
    register float perplexity, neg_ent, sum_P, perplexity_diff, beta, min_beta, max_beta;

    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_points) {
        return;
    }

    neg_ent = neg_entropy[i];
    sum_P = row_sum[i];
    beta = betas[i];

    min_beta = lower_bound[i];
    max_beta = upper_bound[i];
    
    perplexity = (neg_ent / sum_P) + __logf(sum_P);
    perplexity_diff = perplexity - __logf(perplexity_target);
    is_found = (perplexity_diff < epsilon && -perplexity_diff < epsilon);

    if (!is_found) {
        if (perplexity_diff > 0) {
            min_beta = beta;
            beta = (max_beta == FLT_MAX || max_beta == -FLT_MAX) ? beta * 2.0f : (beta + max_beta) / 2.0f;
        } else {
            max_beta = beta;
            beta = (min_beta == -FLT_MAX || min_beta == FLT_MAX) ? beta / 2.0f : (beta + min_beta) / 2.0f;
        }

        lower_bound[i] = min_beta;
        upper_bound[i] = max_beta;
        betas[i] = beta;
    }

    found[i] = is_found;
}



void search_perplexity(thrust::device_vector<float> &pij, thrust::device_vector<float> &dist, 
                       const float perplexity_target, const float epsilon,
                       const int num_points, const int k) {
    thrust::device_vector<float> betas(num_points, 1.0f);
    thrust::device_vector<float> lower_bound_beta(num_points, -FLT_MAX);
    thrust::device_vector<float> upper_bound_beta(num_points, FLT_MAX);
    thrust::device_vector<float> entropy(num_points * k);
    thrust::device_vector<int> found(num_points);
 
    const int BLOCKSIZE1 = 1024;
    const int NBLOCKS1 = (num_points * k + BLOCKSIZE1 - 1) / BLOCKSIZE1;

    const int BLOCKSIZE2 = 128;
    const int NBLOCKS2 = (num_points + BLOCKSIZE2 - 1) / BLOCKSIZE2;
    size_t iters = 0;
    int all_found = 0;
    thrust::device_vector<float> row_sum(num_points, 0.0f);
    thrust::device_vector<float> neg_entropy(num_points, 0.0f);

    do {
        kernel_compute_pij<<<NBLOCKS1, BLOCKSIZE1>>>(
            thrust::raw_pointer_cast(pij.data()),
            thrust::raw_pointer_cast(dist.data()),
            thrust::raw_pointer_cast(betas.data()),
            num_points, k
        );
        cudaDeviceSynchronize();
        kernel_reduce_sum<<<NBLOCKS1, BLOCKSIZE1>>>(thrust::raw_pointer_cast(row_sum.data()),
                                                    thrust::raw_pointer_cast(pij.data()), 
                                                    num_points, k, 1.f);
        thrust::transform(pij.begin(), pij.end(), entropy.begin(), functional_entropy());

        kernel_reduce_sum<<<NBLOCKS1, BLOCKSIZE1>>>(thrust::raw_pointer_cast(neg_entropy.data()),
                                                    thrust::raw_pointer_cast(entropy.data()),
                                                    num_points, k, -1.f);
        kernel_perplexity_search<<<NBLOCKS2, BLOCKSIZE2>>>(
            thrust::raw_pointer_cast(betas.data()),
            thrust::raw_pointer_cast(lower_bound_beta.data()),
            thrust::raw_pointer_cast(upper_bound_beta.data()),
            thrust::raw_pointer_cast(found.data()),
            thrust::raw_pointer_cast(neg_entropy.data()),
            thrust::raw_pointer_cast(row_sum.data()),
            perplexity_target, epsilon, num_points           
        );
        cudaDeviceSynchronize();

        all_found = thrust::reduce(found.begin(), found.end(), 1, thrust::minimum<int>());
        iters++;

    } while (!all_found && iters < 200);

    kernel_matrix_vector_div<<<NBLOCKS1, BLOCKSIZE1>>>(
        thrust::raw_pointer_cast(row_sum.data()),
        thrust::raw_pointer_cast(pij.data()),
        num_points, k
    );
}

