#include <cuda.h>

typedef struct QuadTreeNode {
    float2 top_left;
    float2 bottom_right;
    float2 center_of_mass;
    int num_points;
    int top_left_child_idx;
    int top_right_child_idx;
    int bottom_left_child_idx;
    int bottom_right_child_idx;
} QuadTreeNode_t;

void init_quadtree_node(QuadTreeNode_t *n, float2 top_left, float2 bottom_right,
                        float2 center_of_mass, int num_points) {
    n->top_left = top_left;
    n->bottom_right = bottom_right;
    n->center_of_mass = center_of_mass;
    n->num_points = num_points;
    n->top_left_child_idx = -1;
    n->top_right_child_idx = -1;
    n->bottom_left_child_idx = -1;
    n->bottom_right_child_idx = -1;
}

__host__ __device__ float find_box_width(QuadTreeNode_t *n) {
    return n->bottom_right.x - n->top_left.x; 
}

__host__ __device__ float find_box_height(QuadTreeNode_t *n) {
    return n->bottom_right.y - n->top_left.y;
}

float2 find_box_center(QuadTreeNode_t *n) {
    float center_x = 0.5f * (n->top_left.x + n->bottom_right.x);
    float center_y = 0.5f * (n->top_left.y + n->bottom_right.y);
    return make_float2(center_x, center_y);
}

bool contains(QuadTreeNode_t *n, float2 point) {
    return (point.x >= n->top_left.x && point.x <= n->bottom_right.x &&
            point.y >= n->top_left.y && point.y <= n->bottom_right.y);
}
