#include <cuda.h>
#include <queue>
#include <iostream>
#include "quad_tree.h"

QuadTreeNode::QuadTreeNode(float2 top_left_point, float2 bottom_right_point) {
    top_left = top_left_point;
    bottom_right = bottom_right_point;
    center_of_mass = make_float2(0.f, 0.f);
    num_points = 0;
    box_width = bottom_right.x - top_left.x;
    box_height = bottom_right.y - top_left.y;

    top_left_child = NULL;
    top_right_child = NULL;
    bottom_left_child = NULL;
    bottom_right_child = NULL;
}

__host__ __device__ QuadTreeNode::QuadTreeNode(float2 top_left_point, float2 bottom_right_point,
                           float2 new_point) {
    top_left = top_left_point;
    bottom_right = bottom_right_point;
    center_of_mass = new_point;
    num_points = 1;

    top_left_child = NULL;
    top_right_child = NULL;
    bottom_left_child = NULL;
    bottom_right_child = NULL;
}

__host__ __device__ float2 QuadTreeNode::box_center() {
    float center_x = 0.5f * (top_left.x + bottom_right.x);
    float center_y = 0.5f * (top_left.y + bottom_right.y);
    return make_float2(center_x, center_y);
}

__host__ __device__ bool QuadTreeNode::contains(float2 point) {
    return (point.x >= top_left.x && point.x <= bottom_right.x &&
             point.y >= top_left.y && point.y <= bottom_right.y);
}

__host__ __device__ void QuadTreeNode::add_point(float2 new_point) {
    if (new_point.x < top_left.x || new_point.y < top_left.y || 
        new_point.x > bottom_right.x || new_point.y > bottom_right.y) {
        //std::cout << "New point does not belong in current quad tree node" << "\n";
        return;
    }
  
    float2 old_point;    

    if (num_points == 0) {
        center_of_mass.x = new_point.x;
        center_of_mass.y = new_point.y;
    } else {
        old_point = center_of_mass;
        //update center of mass
        center_of_mass.x = (center_of_mass.x * num_points + new_point.x) / (num_points + 1);
        center_of_mass.y = (center_of_mass.y * num_points + new_point.y) / (num_points + 1);
    }

    num_points++;

            
    float2 box_center = QuadTreeNode::box_center();
    // point in top left node
    if (new_point.x <= box_center.x && new_point.y <= box_center.y) {
        if (top_left_child == NULL) {
            top_left_child = new QuadTreeNode(top_left, box_center, new_point);
            if (top_left_child->contains(old_point)) {
                top_left_child->add_point(old_point);
            }
        } else {
            top_left_child->add_point(new_point);
        }
    }
    // top right node
    else if (new_point.x > box_center.x && new_point.y <= box_center.y) {
        if (top_right_child == NULL) {
            top_right_child = new QuadTreeNode(make_float2(box_center.x, top_left.y), 
                                               make_float2(bottom_right.x, box_center.y), new_point);
            if (top_right_child->contains(old_point)) {
                top_right_child->add_point(old_point);
            }
        } else {
            top_right_child->add_point(new_point);
        }
    }
    // bottom left node
    else if (new_point.x <= box_center.x && new_point.y > box_center.y) {
        if (bottom_left_child == NULL) {
            bottom_left_child = new QuadTreeNode(make_float2(top_left.x, box_center.y),
                                                 make_float2(box_center.x, bottom_right.y), new_point);
            if (bottom_left_child->contains(old_point)) {
                bottom_left_child->add_point(old_point);
            }
        } else {
            bottom_left_child->add_point(new_point);
        }
    }
    // bottom right node
    else if (new_point.x > box_center.x && new_point.y > box_center.y) {
        if (bottom_right_child == NULL) {
            bottom_right_child = new QuadTreeNode(box_center, bottom_right, new_point);
            if (bottom_right_child->contains(old_point)) {
                bottom_right_child->add_point(old_point);
            }
        } else {
            bottom_right_child->add_point(new_point);
        }
    } 
    // ILLEGAL STATE
    else {
    }
}

void print_quad_tree(QuadTreeNode* root) {
    if (root == NULL) {
        return;
    }

    std::cout << "Center of mass: " << root->center_of_mass.x << " " << root->center_of_mass.y <<
                 "  Num points: " << root->num_points << std::endl;

    std::queue<QuadTreeNode*> tree_queue;
    tree_queue.push(root->top_left_child);
    tree_queue.push(root->top_right_child);
    tree_queue.push(root->bottom_left_child);
    tree_queue.push(root->bottom_right_child);

    while (!tree_queue.empty()) {
        QuadTreeNode* next_child = tree_queue.front();
        tree_queue.pop();

        if (next_child != NULL) {
            std::cout << "Center of mass: " << next_child->center_of_mass.x << " " << next_child->center_of_mass.y <<
                         "  Num points: " << next_child->num_points << std::endl;

            tree_queue.push(next_child->top_left_child);
            tree_queue.push(next_child->top_right_child);
            tree_queue.push(next_child->bottom_left_child);
            tree_queue.push(next_child->bottom_right_child);
        }
    }     

}
