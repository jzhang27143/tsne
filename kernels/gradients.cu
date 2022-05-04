#include <chrono>
#include <cuda.h>
#include <queue>

#include "gradients.h"
#include "quad_tree.h"
#include "thrust/device_vector.h"
#include "thrust/reduce.h"

__global__ void kernel_attractive_forces(const float *__restrict__ pij,
                                         const float *__restrict__ embed_x,
                                         const float *__restrict__ embed_y,
                                         const int *__restrict__ nn_indices,
                                         float *__restrict__ grad_attract_x,
                                         float *__restrict__ grad_attract_y,
                                         int num_points, int k) {
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) {
        return;
    }

    float grad_attr_i_x = 0.f;
    float grad_attr_i_y = 0.f;
    // Add attractive gradients for all neighbors of point i
    for (int n_index = 0; n_index < k; n_index++) {
        int j = nn_indices[i * k + n_index];
        float dy1 = embed_x[i] - embed_x[j];
        float dy2 = embed_y[i] - embed_y[j];
        grad_attr_i_x += pij[i * k + n_index] * dy1 / (1 + dy1 * dy1 + dy2 * dy2);
        grad_attr_i_y += pij[i * k + n_index] * dy2 / (1 + dy1 * dy1 + dy2 * dy2);
    }
    grad_attract_x[i] = grad_attr_i_x;
    grad_attract_y[i] = grad_attr_i_y;
}

__device__ __inline__ void device_compute_partial_forces(QuadTreeNode_t *nodes,
                                                         float *__restrict__ grad_repulsive_x,
                                                         float *__restrict__ grad_repulsive_y,
                                                         float *__restrict__ z_partials,
                                                         int point_index, float target_x, float target_y,
                                                         int root_idx, float theta) {

    QuadTreeNode_t *root = &nodes[root_idx];
    if (!root->is_node) {
        return;
    }

    float dx = target_x - root->center_of_mass.x;
    float dy = target_y - root->center_of_mass.y;
    float dist = pow(dx * dx + dy * dy, 0.5);
    float N_cell = (float) root->num_points;
    float box_width = find_box_width(root);
    float box_height = find_box_height(root);
    float r_cell = (box_width > box_height) ? box_width : box_height;

    // Base case: When sufficiently far enough or leaf node
    if (root->is_leaf || theta * dist > r_cell) {
        grad_repulsive_x[point_index] += N_cell * dx / ((1.f + dx * dx + dy * dy) * (1.f + dx * dx + dy * dy));
        grad_repulsive_y[point_index] += N_cell * dy / ((1.f + dx * dx + dy * dy) * (1.f + dx * dx + dy * dy));
        z_partials[point_index] += N_cell / (1.f + dx * dx + dy * dy);
        return;
    }
    
    for (int ofs = 1; ofs <= 4; ofs++) {
        if (nodes[4 * root_idx + ofs].is_node) {
            device_compute_partial_forces(nodes, grad_repulsive_x, grad_repulsive_y, z_partials,
                                          point_index, target_x, target_y, 4 * root_idx + ofs, theta);
        }
    }

    /*
    device_compute_partial_forces(nodes, grad_repulsive_x, grad_repulsive_y, z_partials,
                                  point_index, target_x, target_y, 4 * root_idx + 1, theta);
    device_compute_partial_forces(nodes, grad_repulsive_x, grad_repulsive_y, z_partials,
                                  point_index, target_x, target_y, 4 * root_idx + 2, theta);
    device_compute_partial_forces(nodes, grad_repulsive_x, grad_repulsive_y, z_partials,
                                  point_index, target_x, target_y, 4 * root_idx + 3, theta);
    device_compute_partial_forces(nodes, grad_repulsive_x, grad_repulsive_y, z_partials,
                                  point_index, target_x, target_y, 4 * root_idx + 4, theta);
    */
}

__global__ void kernel_repulsive_forces(QuadTreeNode_t *nodes,
                                        const float *__restrict__ embed_x,
                                        const float *__restrict__ embed_y,
                                        float *__restrict__ grad_repulsive_x,
                                        float *__restrict__ grad_repulsive_y,
                                        float *__restrict__ z_partials,
                                        int num_points, int theta) {
    /*
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) {
        return;
    } 

    float target_x = embed_x[i];
    float target_y = embed_y[i];

    device_compute_partial_forces(nodes, grad_repulsive_x, grad_repulsive_y, z_partials, i,
                                  target_x, target_y, 0, theta);   
    */

    register int num_threads = blockDim.x * gridDim.x;
    register int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < num_points; i += num_threads) {
        float target_x = embed_x[i];
        float target_y = embed_y[i];
        device_compute_partial_forces(nodes, grad_repulsive_x, grad_repulsive_y, z_partials, i,
                                      target_x, target_y, 0, theta);
    }
}

__global__ void kernel_init_root(QuadTreeNode_t *root,
                                 float2 top_left, float2 bottom_right,
                                 int num_points) {
    
    root->top_left = top_left;
    root->bottom_right = bottom_right;
    root->center_of_mass = make_float2(0.f, 0.f);
    root->is_node = true;
    root->is_leaf = false;
    root->start = 0;
    root->end = num_points;
}

__global__ void kernel_normalize_forces(float *__restrict__ grad_repulsive_x,
                                        float *__restrict__ grad_repulsive_y,
                                        float sum_z, int num_points) {
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) {
        return;
    }

    grad_repulsive_x[i] /= sum_z;
    grad_repulsive_y[i] /= sum_z;

}

__global__ void kernel_apply_forces(float *__restrict__ embed_x,
                                    float *__restrict__ embed_y,
                                    float *__restrict__ gains_x,
                                    float *__restrict__ gains_y,
                                    float *__restrict__ old_forces_x,
                                    float *__restrict__ old_forces_y,
                                    float *__restrict__ grad_attract_x,
                                    float *__restrict__ grad_attract_y,
                                    float *__restrict__ grad_repulsive_x,
                                    float *__restrict__ grad_repulsive_y,
                                    float learning_rate, float momentum,
                                    float exaggeration, int num_points) {
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) {
        return;
    }

    float delta_x = old_forces_x[i];
    float delta_y = old_forces_y[i];
    float gain_x = gains_x[i];
    float gain_y = gains_y[i];
    
    float gradient_x = exaggeration * grad_attract_x[i] - grad_repulsive_x[i];
    float gradient_y = exaggeration * grad_attract_y[i] - grad_repulsive_y[i];
    
    gain_x = (gradient_x * delta_x < 0) ? gain_x + 0.2 : gain_x * 0.8;
    gain_y = (gradient_y * delta_y < 0) ? gain_y + 0.2 : gain_y * 0.8;
 
    gain_x = (gain_x < 0.01) ? 0.01 : gain_x;
    gain_y = (gain_y < 0.01) ? 0.01 : gain_y;
    // Calculate gradient using momentum and gains
    delta_x = momentum * delta_x - learning_rate * gain_x * gradient_x;
    delta_y = momentum * delta_y - learning_rate * gain_y * gradient_y;
    // Update gradient
    embed_x[i] += delta_x;
    embed_y[i] += delta_y;

    old_forces_x[i] = delta_x;
    old_forces_y[i] = delta_y;

    gains_x[i] = gain_x;
    gains_y[i] = gain_y;

    // Reset gradients to 0
    grad_attract_x[i] = 0.f;
    grad_attract_y[i] = 0.f;
    grad_repulsive_x[i] = 0.f;
    grad_repulsive_y[i] = 0.f;
}

void compute_attractive_forces(thrust::device_vector<float> &pij,
                               thrust::device_vector<float> &embed_x,
                               thrust::device_vector<float> &embed_y,
                               thrust::device_vector<int> &nn_indices,
                               thrust::device_vector<float> &grad_attract_x,
                               thrust::device_vector<float> &grad_attract_y,
                               int num_points, int k) {
    const int BLOCKSIZE = 1024;
    const int NBLOCKS = (num_points + BLOCKSIZE - 1) / BLOCKSIZE;

    kernel_attractive_forces<<<NBLOCKS, BLOCKSIZE>>>(
        thrust::raw_pointer_cast(pij.data()),
        thrust::raw_pointer_cast(embed_x.data()),
        thrust::raw_pointer_cast(embed_y.data()),
        thrust::raw_pointer_cast(nn_indices.data()),
        thrust::raw_pointer_cast(grad_attract_x.data()),
        thrust::raw_pointer_cast(grad_attract_y.data()),
        num_points, k
    );
}

void compute_repulsive_forces(thrust::device_vector<float> &embed_x,
                              thrust::device_vector<float> &embed_y,
                              thrust::device_vector<float> &embed_x_out,
                              thrust::device_vector<float> &embed_y_out,
                              thrust::device_vector<float> &grad_repulsive_x,
                              thrust::device_vector<float> &grad_repulsive_y,
                              int num_points, float theta) {

    using namespace std::chrono;
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<double> dsec;

    // Step 1: Build the quad tree with the embeded points
    auto build_start = Clock::now();
    auto min_max_x = thrust::minmax_element(embed_x.begin(), embed_x.end());
    auto min_max_y = thrust::minmax_element(embed_y.begin(), embed_y.end());

    float2 top_left = make_float2(min_max_x.first[0] - 1e-5,
                                  min_max_y.first[0] - 1e-5);
    float2 bottom_right = make_float2(min_max_x.second[0] + 1e-5,
                                      min_max_y.second[0] + 1e-5);

    int max_depth = 7;
    int max_nodes = ((1 << (2 * max_depth + 2)) - 1) / 3; // 1 + 4 + 4^2 + ... + 4^max_depth

    QuadTreeNode_t *d_nodes;
    cudaMalloc((void **)&d_nodes, max_nodes * sizeof(QuadTreeNode_t));
    cudaMemset(d_nodes, 0, max_nodes * sizeof(QuadTreeNode_t));
    kernel_init_root<<<1, 1>>>(d_nodes, top_left, bottom_right, num_points);

    int smem_size = 10 * 1024 * sizeof(int);
    thrust::device_vector<float> embed_x_in(embed_x);
    thrust::device_vector<float> embed_y_in(embed_y);
    
    kernel_build_quadtree<<<1, 1024, smem_size>>>(
        thrust::raw_pointer_cast(embed_x_in.data()),
        thrust::raw_pointer_cast(embed_y_in.data()),
        thrust::raw_pointer_cast(embed_x_out.data()),
        thrust::raw_pointer_cast(embed_y_out.data()),
        d_nodes, max_depth, 0
    );

    cudaDeviceSynchronize();
 
    double build_time = duration_cast<dsec>(Clock::now() - build_start).count();
    auto center_mass_start = Clock::now();
    kernel_center_of_mass<<<1, 1024>>>(d_nodes, max_depth);
    cudaDeviceSynchronize();
    double center_mass_time = duration_cast<dsec>(Clock::now() - center_mass_start).count();
    
    //QuadTreeNode_t *h_nodes = (QuadTreeNode_t *) malloc(max_nodes * sizeof(QuadTreeNode_t));
    //cudaMemcpy(h_nodes, d_nodes, max_nodes * sizeof(QuadTreeNode_t), cudaMemcpyDeviceToHost);

    /*
    int num_nodes = 0;
    for (int i = 0; i < max_nodes; i++) {
        QuadTreeNode_t *node = &h_nodes[i];
        if (node->is_node) {
            if (node->is_leaf) num_nodes += node->num_points;
            std::cout << i << ", num points: " << node->num_points <<
            ", center of mass: " << node->center_of_mass.x << " " << node->center_of_mass.y
            << ", tl: " << node->top_left.x << " " << node->top_left.y
            << ", br: " << node->bottom_right.x << " " << node->bottom_right.y
            << ", start: " << node->start << ", end: " << node->end << ", is_leaf: " << node->is_leaf << std::endl;
        }
    }
    */

    // Step 3: Traverse tree
    auto traverse_start = Clock::now();
    const int BLOCKSIZE = 1024;
    const int NBLOCKS = (num_points + BLOCKSIZE - 1) / BLOCKSIZE;

    thrust::device_vector<float> z_partials(num_points, 0.f);
    kernel_repulsive_forces<<<150, 1>>>(d_nodes,
                                        thrust::raw_pointer_cast(embed_x.data()),
                                        thrust::raw_pointer_cast(embed_y.data()),
                                        thrust::raw_pointer_cast(grad_repulsive_x.data()),
                                        thrust::raw_pointer_cast(grad_repulsive_y.data()),
                                        thrust::raw_pointer_cast(z_partials.data()),
                                        num_points, theta);
    cudaDeviceSynchronize();
    /* 
    int num_nodes = 0;
    for (int i = 0; i < max_nodes; i++) {
        QuadTreeNode_t *node = &h_nodes[i];
        if (node->is_node) {
            if (node->is_leaf) num_nodes += node->num_points;
            std::cout << i << ", num points: " << node->num_points <<
            ", center of mass: " << node->center_of_mass.x << " " << node->center_of_mass.y
            << ", tl: " << node->top_left.x << " " << node->top_left.y
            << ", br: " << node->bottom_right.x << " " << node->bottom_right.y
            << ", start: " << node->start << ", end: " << node->end << ", is_leaf: " << node->is_leaf << std::endl;
        }
    }
    free(h_nodes);
    */
 
    double traverse_time = duration_cast<dsec>(Clock::now() - traverse_start).count();

    // Step 4: Normalize forces
    auto normalize_start = Clock::now();

    float sum_z = thrust::reduce(z_partials.begin(), z_partials.end(), 0.f, thrust::plus<float>());
    kernel_normalize_forces<<<NBLOCKS, BLOCKSIZE>>>(thrust::raw_pointer_cast(grad_repulsive_x.data()),
                                                    thrust::raw_pointer_cast(grad_repulsive_y.data()),
                                                    sum_z, num_points);
    cudaDeviceSynchronize();
    double normalize_time = duration_cast<dsec>(Clock::now() - normalize_start).count();

    printf("Building Tree Time: %lf.\n", build_time);
    printf("Finding Center of Mass Time: %lf.\n", center_mass_time);
    printf("Traversing Tree Time: %lf.\n", traverse_time);
    printf("Normalizing/Reducing Time: %lf.\n", normalize_time);
    cudaFree(d_nodes);
}

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
                  float exaggeration, int num_points) {

    const int BLOCKSIZE = 1024;
    const int NBLOCKS = (num_points + BLOCKSIZE - 1) / BLOCKSIZE;

    kernel_apply_forces<<<NBLOCKS, BLOCKSIZE>>>(thrust::raw_pointer_cast(embed_x.data()),
                                                thrust::raw_pointer_cast(embed_y.data()),
                                                thrust::raw_pointer_cast(gains_x.data()),
                                                thrust::raw_pointer_cast(gains_y.data()),
                                                thrust::raw_pointer_cast(old_forces_x.data()),
                                                thrust::raw_pointer_cast(old_forces_y.data()),
                                                thrust::raw_pointer_cast(grad_attract_x.data()),
                                                thrust::raw_pointer_cast(grad_attract_y.data()),
                                                thrust::raw_pointer_cast(grad_repulsive_x.data()),
                                                thrust::raw_pointer_cast(grad_repulsive_y.data()),
                                                learning_rate, momentum, exaggeration,  num_points);

}

