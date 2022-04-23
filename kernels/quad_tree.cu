#include <thrust/device_vector.h>

class QuadTreeNode {
    const float2 top_left;
    const float2 bottom_right;
    float2 center_of_mass;
    int num_points;

    QuadTreeNode *top_left_child;
    QuadTreeNode *top_right_child;
    QuadTreeNode *bottom_left_child;
    QuadTreeNode *bottom_right_child; 

    public:
 
        __host__ __device__ QuadTreeNode(float2 top_left, float2 bottom_right,
                                         float2 point): {
            top_left = top_left;
            bottom_right = bottom_right;
            center_of_mass = point;
            num_points = 1;

            top_left_child = NULL;
            top_right_child = NULL;
            bottom_left_child = NULL;
            bottom_right_child = NULL;
        }

        __host__ __device__ float2 box_center() {
            float center_x = 0.5f * (top_left.x + bottom_right.x);
            float center_y = 0.5f * (top_left.y + bottom_right.y);
            return make_float2(center_x, center_y);
        }

        __host__ __device__ void add_point(float2 new_point) {
            if (new_point.x < top_left.x || new_point.y < top_left.y || 
                new_point.x > bottom_right.x || new_point.y > bottom_right.y) {
                std::cout << "New point does not belong in current quad tree node" << "\n";
                return;
            }

            //update center of mass
            float new_mass_x = (center_of_mass_x * num_points + new_point.x) / (num_points + 1);
            float new_mass_y = (center_of_mass_y * num_points + new_point.y) / (num_points + 1);
            num_points++;
            
            float2 box_center = box_center();
            // point in top left node
            if (new_point.x <= box_center.x && new_point.y <= box_center.y) {
                if (top_left_child == NULL) {
                    top_left_child = new QuadTreeNode(top_left, box_center, new_point);
                } else {
                    top_left_child.add_point(new_point);
                }
            }
            // top right node
            else if (new_point.x <= box_center.x && new_point.y > box_center.y) {
                if (top_right_child == NULL) {
                    top_right_child = new QuadTreeNode(make_float2(box_center.x, top_left.y), 
                                                       make_float2(bottom_right.x, box_center.y), new_point);
                } else {
                    top_right_child.add_point(new_point);
                }
            }
            // bottom left node
            else if (new_point.x > box_center.x && new_point.y <= box_center.y) {
                if (bottom_left_child == NULL) {
                    bottom_left_child = new QuadTreeNode(make_float2(top_left.x, box_center.y),
                                                         make_float2(box_center.x, bottom_right.y), new_point);
                } else {
                    bottom_left_child.add_point(new_point);
                }
            }
            // bottom right node
            else if (new_point.x > box_center.x && new_point.y > box_center.y) {
                if (bottom_right_child == NULL) {
                    bottom_right_child = new QuadTreeNode(box_center, bottom_right, new_point);
                } else {
                    bottom_right_child.add_point(new_point);
                }
            } 
            // ILLEGAL STATE
            else {
            }
        }
    




}  
