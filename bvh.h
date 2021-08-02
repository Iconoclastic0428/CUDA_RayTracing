#ifndef BVH_H
#define BVH_H

#include <thrust/sort.h>
#include <curand_kernel.h>

#include "hittable.h"


struct box_compare {
    __device__ box_compare(int m) : mode(m) {}
    __device__ bool operator()(hittable* a, hittable* b) const {
        // return true;

        aabb box_left, box_right;
        hittable* ah = a;
        hittable* bh = b;

        if (!ah->bounding_box(0, 0, box_left) || !bh->bounding_box(0, 0, box_right)) {
            return false;
        }

        float val1, val2;
        if (mode == 1) {
            val1 = box_left.min().x();
            val2 = box_right.min().x();
        }
        else if (mode == 2) {
            val1 = box_left.min().y();
            val2 = box_right.min().y();
        }
        else if (mode == 3) {
            val1 = box_left.min().z();
            val2 = box_right.min().z();
        }

        if (val1 - val2 < 0.0) {
            return false;
        }
        else {
            return true;
        }
    }
    // mode: 1, x; 2, y; 3, z
    int mode;
};


class bvhNode : public hittable {
public:
    __device__ bvhNode() {}
    __device__ bvhNode(hittable** l,
        int n,
        float time0,
        float time1,
        curandState* state);

    __device__ virtual bool hit(const ray& r,
        float t_min,
        float t_max,
        hit_record& rec) const;

    __device__ virtual bool bounding_box(float t0,
        float t1,
        aabb& b) const;

    hittable* left;
    hittable* right;
    aabb box;
};


__device__ bvhNode::bvhNode(hittable** l,
    int n,
    float time0,
    float time1,
    curandState* state) {
    int axis = int(3 * curand_uniform(state));
    if (axis == 0) {
        thrust::sort(l, l + n, box_compare(1));
    }
    else if (axis == 1) {
        thrust::sort(l, l + n, box_compare(2));
    }
    else {
        thrust::sort(l, l + n, box_compare(3));
    }

    if (n == 1) {
        left = right = l[0];
    }
    else if (n == 2) {
        left = l[0];
        right = l[1];
    }
    else {
        left = new bvhNode(l, n / 2, time0, time1, state);
        right = new bvhNode(l + n / 2, n - n / 2, time0, time1, state);
    }

    aabb box_left, box_right;
    if (!left->bounding_box(time0, time1, box_left) ||
        !right->bounding_box(time0, time1, box_right)) {
        return;
    }
    box = surrounding_box(box_left, box_right);

}


__device__ bool bvhNode::bounding_box(float t0,
    float t1,
    aabb& b) const {
    b = box;
    return true;
}


__device__ bool bvhNode::hit(const ray& r,
    float t_min,
    float t_max,
    hit_record& rec) const {
    if (box.hit(r, t_min, t_max)) {
        hit_record left_rec, right_rec;
        bool hit_left = left->hit(r, t_min, t_max, left_rec);
        bool hit_right = right->hit(r, t_min, t_max, right_rec);
        if (hit_left && hit_right) {
            if (left_rec.t < right_rec.t) {
                rec = left_rec;
            }
            else {
                rec = right_rec;
            }
            return true;
        }
        else if (hit_left) {
            rec = left_rec;
            return true;
        }
        else if (hit_right) {
            rec = right_rec;
            return true;
        }
        else {
            return false;
        }
    }
    return false;
}

#endif 