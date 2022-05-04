#include <cuda.h>
#include "quad_tree.h"
#include "exclusiveScan.cu_inl"

__device__ float2 find_box_center(QuadTreeNode_t *n) {
    float center_x = 0.5f * (n->top_left.x + n->bottom_right.x);
    float center_y = 0.5f * (n->top_left.y + n->bottom_right.y);
    return make_float2(center_x, center_y);
}

__device__ float find_box_width(QuadTreeNode_t *n) {
    return n->bottom_right.x - n->top_left.x; 
}

__device__ float find_box_height(QuadTreeNode_t *n) {
    return n->bottom_right.y - n->top_left.y;
}

__global__ void kernel_center_of_mass(QuadTreeNode_t *nodes, const int max_depth) {
    register int tid = blockIdx.x * blockDim.x + threadIdx.x;
    register int num_threads = blockDim.x * gridDim.x;

    for (int d = max_depth - 1; d >= 0; d--) {
        register int start = ((1 << (2 * d)) - 1) / 3;
        register int end = ((1 << (2 * d + 2)) - 1) / 3;

        for (int i = start + tid; i < end; i += num_threads) {
            QuadTreeNode_t *node = &nodes[i];
            if (node->is_node && !node->is_leaf) {
                float x = 0.f, y = 0.f;
                int N = node->num_points;
                QuadTreeNode_t *top_left = &nodes[4 * i + 1];
                QuadTreeNode_t *top_right = &nodes[4 * i + 2];
                QuadTreeNode_t *bottom_left = &nodes[4 * i + 3];
                QuadTreeNode_t *bottom_right = &nodes[4 * i + 4];
                if (top_left->is_node) {
                    x += (&nodes[4 * i + 1])->center_of_mass.x * (&nodes[4 * i + 1])->num_points;
                    y += (&nodes[4 * i + 1])->center_of_mass.y * (&nodes[4 * i + 1])->num_points;
                }               
                
                if (top_right->is_node) {
                    x += (&nodes[4 * i + 2])->center_of_mass.x * (&nodes[4 * i + 2])->num_points;
                    y += (&nodes[4 * i + 2])->center_of_mass.y * (&nodes[4 * i + 2])->num_points;
                }
 
                if (bottom_left->is_node) {
                    x += (&nodes[4 * i + 3])->center_of_mass.x * (&nodes[4 * i + 3])->num_points;
                    y += (&nodes[4 * i + 3])->center_of_mass.y * (&nodes[4 * i + 3])->num_points;
                }

                if (bottom_right->is_node) {
                    x += (&nodes[4 * i + 4])->center_of_mass.x * (&nodes[4 * i + 4])->num_points;
                    y += (&nodes[4 * i + 4])->center_of_mass.y * (&nodes[4 * i + 4])->num_points;
                }
                node->center_of_mass = make_float2(x / N, y / N);
            }
        }
        __syncthreads();
    }
}

__global__ void kernel_build_quadtree(float *__restrict__ embed_x_in,
                                      float *__restrict__ embed_y_in,
                                      float *__restrict__ embed_x_out,
                                      float *__restrict__ embed_y_out,
                                      QuadTreeNode_t *nodes,
                                      int remaining_depth,
                                      int node_idx) {

    extern __shared__ uint smem[];
    int root_node_idx = node_idx + blockIdx.x;    

    int start = nodes[root_node_idx].start;
    int end = nodes[root_node_idx].end;

    register int num_threads = blockDim.x;
    uint *bucket_counts_tl = smem;
    uint *bucket_counts_tr = &smem[num_threads];
    uint *bucket_counts_bl = &smem[2 * num_threads];
    uint *bucket_counts_br = &smem[3 * num_threads];
    uint *ex_scan_tl = &smem[4 * num_threads];
    uint *ex_scan_tr = &smem[5 * num_threads];
    uint *ex_scan_bl = &smem[6 * num_threads];
    uint *ex_scan_br = &smem[7 * num_threads];
    uint *ex_scan_scratch = &smem[8 * num_threads];

    // register int tid = blockIdx.x * blockDim.x + threadIdx.x;
    register int tid = threadIdx.x;
    register int num_points = end - start;
    if (num_points == 0) {
        return;
    }

    QuadTreeNode_t *node = &nodes[root_node_idx];
    if (tid == 0) {
        node->num_points = num_points;
        node->is_node = true;
    }

    if (num_points == 1 || remaining_depth == 0) {
        if (tid == 0) {
            float x_center_of_mass = 0.f;
            float y_center_of_mass = 0.f;
            for (int i = start; i < end; i++) {
                x_center_of_mass += embed_x_in[i];
                y_center_of_mass += embed_y_in[i];
            }

            x_center_of_mass /= num_points;
            y_center_of_mass /= num_points;
            node->center_of_mass = make_float2(x_center_of_mass, y_center_of_mass);
            node->is_leaf = true;
        }
        return;
    }

    float2 box_center = find_box_center(node);
    int num_top_left = 0;
    int num_top_right = 0;
    int num_bottom_left = 0;
    int num_bottom_right = 0;

    // Step 1: Each thread counts the number of points per bucket
    for (int i = start + tid; i < end; i += num_threads) {
        float x = embed_x_in[i];
        float y = embed_y_in[i];
        if (x <= box_center.x && y <= box_center.y) {
            num_top_left++;
        }
        else if (x > box_center.x && y <= box_center.y) {
            num_top_right++;
        }
        else if (x <= box_center.x && y > box_center.y) {
            num_bottom_left++;
        }
        else {
            num_bottom_right++;
        }
    }

    bucket_counts_tl[tid] = num_top_left;
    bucket_counts_tr[tid] = num_top_right;
    bucket_counts_bl[tid] = num_bottom_left;
    bucket_counts_br[tid] = num_bottom_right;

    // Step 2: Perform Exclusive Scan for each bucket
    sharedMemExclusiveScan(tid, bucket_counts_tl, ex_scan_tl, ex_scan_scratch, num_threads);
    sharedMemExclusiveScan(tid, bucket_counts_tr, ex_scan_tr, ex_scan_scratch, num_threads);
    sharedMemExclusiveScan(tid, bucket_counts_bl, ex_scan_bl, ex_scan_scratch, num_threads);
    sharedMemExclusiveScan(tid, bucket_counts_br, ex_scan_br, ex_scan_scratch, num_threads);
    __syncthreads();

    // Step 3: Compute offsets for reordered indices
    int offset_tr = start + ex_scan_tl[num_threads - 1] + bucket_counts_tl[num_threads - 1];
    int offset_bl = offset_tr + ex_scan_tr[num_threads - 1] + bucket_counts_tr[num_threads - 1];
    int offset_br = offset_bl + ex_scan_bl[num_threads - 1] + bucket_counts_bl[num_threads - 1];
    __syncthreads();

    ex_scan_tl[tid] += start;
    ex_scan_tr[tid] += offset_tr;
    ex_scan_bl[tid] += offset_bl;
    ex_scan_br[tid] += offset_br;
    __syncthreads();

    // Step 4: Fill in points in bucket sort order
    int fill_idx_tl = ex_scan_tl[tid];
    int fill_idx_tr = ex_scan_tr[tid];
    int fill_idx_bl = ex_scan_bl[tid];
    int fill_idx_br = ex_scan_br[tid];

    for (int i = start + tid; i < end; i += num_threads) {
        float x = embed_x_in[i];
        float y = embed_y_in[i];
        if (x <= box_center.x && y <= box_center.y) {
            embed_x_out[fill_idx_tl] = x;
            embed_y_out[fill_idx_tl++] = y;
        }
        else if (x > box_center.x && y <= box_center.y) {
            embed_x_out[fill_idx_tr] = x;
            embed_y_out[fill_idx_tr++] = y;
        }
        else if (x <= box_center.x && y > box_center.y) {
            embed_x_out[fill_idx_bl] = x;
            embed_y_out[fill_idx_bl++] = y;
        }
        else {
            embed_x_out[fill_idx_br] = x;
            embed_y_out[fill_idx_br++] = y;
        }
    }
    __syncthreads();

    if (tid == 0) {
        int smem_size = 10 * num_threads * sizeof(int);
        float2 top_left = nodes[root_node_idx].top_left;
        float2 bottom_right = nodes[root_node_idx].bottom_right;        

        nodes[4 * root_node_idx + 1].top_left = top_left;
        nodes[4 * root_node_idx + 1].bottom_right = box_center;
        nodes[4 * root_node_idx + 2].top_left = make_float2(box_center.x, top_left.y);
        nodes[4 * root_node_idx + 2].bottom_right = make_float2(bottom_right.x, box_center.y);
        nodes[4 * root_node_idx + 3].top_left = make_float2(top_left.x, box_center.y);
        nodes[4 * root_node_idx + 3].bottom_right = make_float2(box_center.x, bottom_right.y);
        nodes[4 * root_node_idx + 4].top_left = box_center;
        nodes[4 * root_node_idx + 4].bottom_right = bottom_right;

        nodes[4 * root_node_idx + 1].start = start;
        nodes[4 * root_node_idx + 1].end = ex_scan_tr[0];
        nodes[4 * root_node_idx + 2].start = ex_scan_tr[0];
        nodes[4 * root_node_idx + 2].end = ex_scan_bl[0];
        nodes[4 * root_node_idx + 3].start = ex_scan_bl[0];
        nodes[4 * root_node_idx + 3].end = ex_scan_br[0];
        nodes[4 * root_node_idx + 4].start = ex_scan_br[0];
        nodes[4 * root_node_idx + 4].end = end;

        // Launching kernel for all 4 quadrants
        kernel_build_quadtree<<<4, num_threads, smem_size>>>(
            embed_x_out, embed_y_out, embed_x_in, embed_y_in,
            nodes, remaining_depth - 1, 4 * root_node_idx + 1
        );
    }
}
