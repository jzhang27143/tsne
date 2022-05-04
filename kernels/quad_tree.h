#include <cuda.h>

typedef struct QuadTreeNode {
    float2 top_left;
    float2 bottom_right;
    float2 center_of_mass;
    int start;
    int end;
    int num_points;
    bool is_node;
    bool is_leaf;
    /**
     * top left index: 4i + 1
     * top right index: 4i + 2
     * bot left index: 4i + 3
     * bot right index: 4i + 4
     */
} QuadTreeNode_t;

__device__ float2 find_box_center(QuadTreeNode_t *n);

__device__ float find_box_width(QuadTreeNode_t *n);

__device__ float find_box_height(QuadTreeNode_t *n);

__global__ void kernel_center_of_mass(QuadTreeNode_t *nodes, int max_depth);

__global__ void kernel_build_quadtree(float *__restrict__ embed_x_in,
                                      float *__restrict__ embed_y_in,
                                      float *__restrict__ embed_x_out,
                                      float *__restrict__ embed_y_out,
                                      QuadTreeNode_t *nodes,
                                      int remaining_depth,
                                      int node_idx);
