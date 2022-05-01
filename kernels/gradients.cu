#include <chrono>
#include <cuda.h>
#include "thrust/device_vector.h"
#include "thrust/reduce.h"
#include "gradients.h"
#include "quad_tree2.h"
#include <queue>

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

    if (root_idx < 0) {
        return;
    }

    QuadTreeNode_t *root = &nodes[root_idx];

    float dx = target_x - root->center_of_mass.x;
    float dy = target_y - root->center_of_mass.y;
    float dist = pow(dx * dx + dy * dy, 0.5);
    float N_cell = (float) root->num_points;
    float box_width = find_box_width(root);
    float box_height = find_box_height(root);
    float r_cell = (box_width > box_height) ? box_width : box_height;

    // Base case: When sufficiently far enough or leaf node
    bool is_leaf = (root->top_left_child_idx < 0) && (root->top_right_child_idx < 0) &&
                   (root->bottom_left_child_idx < 0) && (root->bottom_right_child_idx < 0); 
    if (is_leaf || theta * dist > r_cell) {
        grad_repulsive_x[point_index] += N_cell * dx / ((1.f + dx * dx + dy * dy) * (1.f + dx * dx + dy * dy));
        grad_repulsive_y[point_index] += N_cell * dy / ((1.f + dx * dx + dy * dy) * (1.f + dx * dx + dy * dy));
        z_partials[point_index] += N_cell / (1.f + dx * dx + dy * dy);
        return;
    }
     
    device_compute_partial_forces(nodes, grad_repulsive_x, grad_repulsive_y, z_partials,
                                  point_index, target_x, target_y, root->top_left_child_idx, theta);
    
    device_compute_partial_forces(nodes, grad_repulsive_x, grad_repulsive_y, z_partials,
                                  point_index, target_x, target_y, root->top_right_child_idx, theta);
    device_compute_partial_forces(nodes, grad_repulsive_x, grad_repulsive_y, z_partials,
                                  point_index, target_x, target_y, root->bottom_left_child_idx, theta);
    
   
    device_compute_partial_forces(nodes, grad_repulsive_x, grad_repulsive_y, z_partials,
                                  point_index, target_x, target_y, root->bottom_right_child_idx, theta);
  
 
}

__global__ void kernel_repulsive_forces(QuadTreeNode_t *nodes,
                                        const float *__restrict__ embed_x,
                                        const float *__restrict__ embed_y,
                                        float *__restrict__ grad_repulsive_x,
                                        float *__restrict__ grad_repulsive_y,
                                        float *__restrict__ z_partials,
                                        int num_points, int theta) {
    register int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) {
        return;
    }

    float target_x = embed_x[i];
    float target_y = embed_y[i];

    device_compute_partial_forces(nodes, grad_repulsive_x, grad_repulsive_y, z_partials, i,
                                  target_x, target_y, 0, theta);   

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

void insert_quadtree_node(QuadTreeNode_t *nodes, int node_index,
                          float x, float y, int *next_free_index, int max_depth) {
    QuadTreeNode_t *node = &nodes[node_index];
    if (x < node->top_left.x || y < node->top_left.y ||
        x > node->bottom_right.x || y > node->bottom_right.y) {
        std::cout << "New point does not belong in current quad tree node\n";
        return;
    }

    float2 old_point;
    int N = node->num_points;
    // Immediately occupy empty cell
    if (N == 0) {
        node->center_of_mass.x = x;
        node->center_of_mass.y = y;
        node->num_points++;
        return;
    }

    old_point = node->center_of_mass;
    node->center_of_mass.x = (node->center_of_mass.x * N + x) / (N + 1);
    node->center_of_mass.y = (node->center_of_mass.y * N + y) / (N + 1);
    node->num_points++;
    if (max_depth == 0) {
        return;
    }

    // If cell had one point (leaf cell), split it into four
    float2 box_center = find_box_center(node);
    float2 zeros = make_float2(0.f, 0.f);
    if (N == 1) {
        init_quadtree_node(&nodes[*next_free_index], node->top_left, box_center,
                           zeros, 0);
        init_quadtree_node(&nodes[*next_free_index + 1], make_float2(box_center.x, node->top_left.y),
                           make_float2(node->bottom_right.x, box_center.y), zeros, 0);
        init_quadtree_node(&nodes[*next_free_index + 2], make_float2(node->top_left.x, box_center.y),
                           make_float2(box_center.x, node->bottom_right.y), zeros, 0);
        init_quadtree_node(&nodes[*next_free_index + 3], box_center, node->bottom_right,
                           zeros, 0);
        node->top_left_child_idx = *next_free_index;
        node->top_right_child_idx = *next_free_index + 1;
        node->bottom_left_child_idx = *next_free_index + 2;
        node->bottom_right_child_idx = *next_free_index + 3;
        *next_free_index += 4;

        // Insert new point and re-insert original point (should succeed immediately)
        if (x <= box_center.x && y <= box_center.y) {
            insert_quadtree_node(nodes, node->top_left_child_idx, x, y, next_free_index, max_depth-1);
        }
        else if (x > box_center.x && y <= box_center.y) {
            insert_quadtree_node(nodes, node->top_right_child_idx, x, y, next_free_index, max_depth-1);
        }
        else if (x <= box_center.x && y > box_center.y) {
            insert_quadtree_node(nodes, node->bottom_left_child_idx, x, y, next_free_index, max_depth-1);
        }
        else if (x > box_center.x && y > box_center.y) {
            insert_quadtree_node(nodes, node->bottom_right_child_idx, x, y, next_free_index, max_depth-1);
        }

        if (old_point.x <= box_center.x && old_point.y <= box_center.y) {
            insert_quadtree_node(nodes, node->top_left_child_idx, old_point.x, old_point.y, next_free_index, max_depth-1);
        }
        else if (old_point.x > box_center.x && old_point.y <= box_center.y) {
            insert_quadtree_node(nodes, node->top_right_child_idx, old_point.x, old_point.y, next_free_index, max_depth-1);
        }
        else if (old_point.x <= box_center.x && old_point.y > box_center.y) {
            insert_quadtree_node(nodes, node->bottom_left_child_idx, old_point.x, old_point.y, next_free_index, max_depth-1);
        }
        else if (old_point.x > box_center.x && old_point.y > box_center.y) {
            insert_quadtree_node(nodes, node->bottom_right_child_idx, old_point.x, old_point.y, next_free_index, max_depth-1);
        }
        return;
    }

    // Otherwise, recurse down to children. Node must have 4 children allocated.
    assert (node->top_left_child_idx >= 0 && node->top_right_child_idx >= 0 &&
            node->bottom_left_child_idx >= 0 && node->bottom_right_child_idx >= 0);
    if (x <= box_center.x && y <= box_center.y) {
        insert_quadtree_node(nodes, node->top_left_child_idx, x, y, next_free_index, max_depth-1);
    }
    else if (x > box_center.x && y <= box_center.y) {
        insert_quadtree_node(nodes, node->top_right_child_idx, x, y, next_free_index, max_depth-1);
    }
    else if (x <= box_center.x && y > box_center.y) {
        insert_quadtree_node(nodes, node->bottom_left_child_idx, x, y, next_free_index, max_depth-1);
    }
    else if (x > box_center.x && y > box_center.y) {
        insert_quadtree_node(nodes, node->bottom_right_child_idx, x, y, next_free_index, max_depth-1);
    }
}

void print_quad_tree2(QuadTreeNode* root) {
    if (root->num_points == 0) {
        return;
    }
    
    std::cout << "Center Of Mass: " << root->center_of_mass.x << " " << root->center_of_mass.y <<
                 " Num points: " << root->num_points << std::endl;
    
    std::queue<int> tree_queue;
    
    tree_queue.push(root->top_left_child_idx);
    tree_queue.push(root->top_right_child_idx);
    tree_queue.push(root->bottom_left_child_idx);
    tree_queue.push(root->bottom_right_child_idx);

    while (!tree_queue.empty()) {
        int next_child_idx = tree_queue.front();
        tree_queue.pop();

        if (next_child_idx > -1 && root[next_child_idx].num_points > 0) {
            QuadTreeNode* next_child = &root[next_child_idx];
            std::cout << "Center Of Mass: " << next_child->center_of_mass.x << " " << next_child->center_of_mass.y <<
                         " Num points: " << next_child->num_points << std::endl;

            tree_queue.push(next_child->top_left_child_idx);
            tree_queue.push(next_child->top_right_child_idx);
            tree_queue.push(next_child->bottom_left_child_idx);
            tree_queue.push(next_child->bottom_right_child_idx);
        }
    }

}

void compute_repulsive_forces(thrust::device_vector<float> &embed_x,
                              thrust::device_vector<float> &embed_y,
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

    int max_depth = 8;
    int max_nodes = ((1 << (2 * max_depth + 2)) - 1) / 3; // 1 + 4 + 4^2 + ... + 4^max_depth

    QuadTreeNode_t *nodes = (QuadTreeNode_t *) malloc(max_nodes * sizeof(QuadTreeNode_t));
    init_quadtree_node(&nodes[0], top_left, bottom_right, make_float2(0.f, 0.f), 0);
    int next_free_idx = 1;

    for (int i = 0; i < num_points; i++) {
        insert_quadtree_node(nodes, 0, embed_x[i], embed_y[i], &next_free_idx, max_depth);
    }

    double build_time = duration_cast<dsec>(Clock::now() - build_start).count();

    // Step 2: Memcpy to GPU;

    QuadTreeNode_t *d_nodes;
    cudaMalloc((void **)&d_nodes, max_nodes * sizeof(QuadTreeNode_t));
    cudaMemcpy(d_nodes, nodes, max_nodes * sizeof(QuadTreeNode_t), cudaMemcpyHostToDevice);
    
    thrust::device_vector<float> z_partials(num_points, 0.f);

    // Step 3: Traverse tree
    auto traverse_start = Clock::now();
   
    const int BLOCKSIZE = 1024;
    const int NBLOCKS = (num_points + BLOCKSIZE - 1) / BLOCKSIZE;

    kernel_repulsive_forces<<<NBLOCKS, BLOCKSIZE>>>(d_nodes,
                                                    thrust::raw_pointer_cast(embed_x.data()),
                                                    thrust::raw_pointer_cast(embed_y.data()),
                                                    thrust::raw_pointer_cast(grad_repulsive_x.data()),
                                                    thrust::raw_pointer_cast(grad_repulsive_y.data()),
                                                    thrust::raw_pointer_cast(z_partials.data()),
                                                    num_points, theta);
    
    double traverse_time = duration_cast<dsec>(Clock::now() - traverse_start).count();
    // Step 4: Normalize forces
    auto normalize_start = Clock::now();
     
    float sum_z = thrust::reduce(z_partials.begin(), z_partials.end(), 0.f, thrust::plus<float>());

    kernel_normalize_forces<<<NBLOCKS, BLOCKSIZE>>>(thrust::raw_pointer_cast(grad_repulsive_x.data()),
                                                    thrust::raw_pointer_cast(grad_repulsive_y.data()),
                                                    sum_z, num_points);
    
    double normalize_time = duration_cast<dsec>(Clock::now() - normalize_start).count();

    printf("Building Tree Time: %lf.\n", build_time);
    printf("Traversing Tree Time: %lf.\n", traverse_time);
    printf("Normalizing/Reducing Time: %lf.\n", normalize_time);
    cudaFree(d_nodes);
    free(nodes);    
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

