#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"
#include "aabb.h"
class material;

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
    //float u;
    //float v;
    material* mat_ptr;
};

class hittable {
public:
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const = 0;
    __device__ virtual double pdf_value(const vec3& o, const vec3& v) const {
        return 0.0;
    }

    __device__ virtual vec3 random(const vec3& o, curandState* state) const {
        return vec3(1, 0, 0);
    }
};

class rotate_y : public hittable {
public:
    __device__ rotate_y(hittable* p, float angle);
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const override {
        box = bbox;
        return hasbox;
    }
    hittable* ptr;
    float sin_theta;
    float cos_theta;
    bool hasbox;
    aabb bbox;
};

__device__ rotate_y::rotate_y(hittable* p, float angle) : ptr(p) {
    auto radians = angle / (2 * M_PI);
    sin_theta = sin(radians);
    cos_theta = cos(radians);
    hasbox = ptr->bounding_box(0, 1, bbox);

    vec3 min(FLT_MIN, FLT_MIN, FLT_MIN);
    vec3 max(FLT_MAX, FLT_MAX, FLT_MAX);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                auto x = i * bbox.max().x() + (1 - i) * bbox.min().x();
                auto y = j * bbox.max().y() + (1 - j) * bbox.min().y();
                auto z = k * bbox.max().z() + (1 - k) * bbox.min().z();

                auto newx = cos_theta * x + sin_theta * z;
                auto newz = -sin_theta * x + cos_theta * z;
                vec3 tester(newx, y, newz);
                for (int c = 0; c < 3; ++c) {
                    min[c] = fmin(min[c], tester[c]);
                    max[c] = fmax(max[c], tester[c]);
                }
            }
        }
    }

    bbox = aabb(min, max);
}

__device__ bool rotate_y::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    auto origin = r.origin();
    auto direction = r.direction();

    origin[0] = cos_theta * r.origin()[0] - sin_theta * r.origin()[2];
    origin[2] = sin_theta * r.origin()[0] + cos_theta * r.origin()[2];

    direction[0] = cos_theta * r.direction()[0] - sin_theta * r.direction()[2];
    direction[2] = sin_theta * r.direction()[0] + cos_theta * r.direction()[2];

    ray rotated_r(origin, direction);

    if (!ptr->hit(rotated_r, t_min, t_max, rec))
        return false;

    auto p = rec.p;
    auto normal = rec.normal;

    p[0] = cos_theta * rec.p[0] + sin_theta * rec.p[2];
    p[2] = -sin_theta * rec.p[0] + cos_theta * rec.p[2];

    normal[0] = cos_theta * rec.normal[0] + sin_theta * rec.normal[2];
    normal[2] = -sin_theta * rec.normal[0] + cos_theta * rec.normal[2];

    rec.p = p;
    auto outward_normal = vec3(0, 0, 1);
    bool front_face = dot(r.direction(), outward_normal) < 0;
    rec.normal = front_face ? outward_normal : -outward_normal;

    return true;
}

#endif