#ifndef AABB_H
#define AABB_H

#include "ray.h"
#include "float.h"

__device__ inline float ffmin(float a, float b) { return a < b ? a : b; }
__device__ inline float ffmax(float a, float b) { return a > b ? a : b; }


class aabb {
public:

    __device__ aabb() {
        float minNum = FLT_MIN;
        float maxNum = FLT_MAX;
        _min = vec3(maxNum, maxNum, maxNum);
        _max = vec3(minNum, minNum, minNum);
    }

    // bbox of a point
    __device__ aabb(const vec3& p) : _min(p), _max(p) {}

    // regular constructor
    // TODO: add sanity check
    __device__ aabb(const vec3& p1, const vec3& p2) : _min(p1), _max(p2) {}

    __device__ bool hit(const ray& r,
        float t_min,
        float t_max) const {
        for (int a = 0; a < 3; a++) {
            float t0 = ffmin((_min[a] - r.origin()[a]) / r.direction()[a],
                (_max[a] - r.origin()[a]) / r.direction()[a]);
            float t1 = ffmax((_min[a] - r.origin()[a]) / r.direction()[a],
                (_max[a] - r.origin()[a]) / r.direction()[a]);
            t_min = ffmax(t0, t_min);
            t_max = ffmin(t1, t_max);
            if (t_max <= t_min) return false;
        }
        return true;
    }

    __device__ vec3 min() const { return _min; }
    __device__ vec3 max() const { return _max; }

    vec3 _min, _max;
};


// Get union of two aabb, for temporary use
__device__  aabb surrounding_box(aabb box0, aabb box1) {
    vec3 small(fmin(box0.min().x(), box1.min().x()),
        fmin(box0.min().y(), box1.min().y()),
        fmin(box0.min().z(), box1.min().z()));
    vec3 big(fmax(box0.max().x(), box1.max().x()),
        fmax(box0.max().y(), box1.max().y()),
        fmax(box0.max().z(), box1.max().z()));
    return aabb(small, big);
}


#endif