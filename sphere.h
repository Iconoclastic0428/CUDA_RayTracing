#ifndef SPHERE_H
#define SPHERE_H

#include "vec3.h"
#include "hittable.h"
#include "material.h"
#include "aabb.h"

class sphere : public hittable {
public:
    __device__ sphere() {}
    __device__ sphere(vec3 cen, float r, material* m) : center(cen), radius(r), mat_ptr(m) {};
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    __device__ virtual bool bounding_box(float t0,
        float t1,
        aabb& box) const;
    vec3 center;
    float radius;
    material* mat_ptr;
    __device__ static void get_sphere_uv(const vec3& p, float& u, float& v) {
        auto theta = acos(-p.y());
        auto phi = atan2(-p.z(), p.x()) + 3.14159265f;
        u = phi / (2 * 3.14159265f);
        v = theta / 3.14159265f;
    }
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            rec.normal = (rec.p - center) / radius;
            //get_sphere_uv(rec.normal, rec.u, rec.v);
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            rec.normal = (rec.p - center) / radius;
            //get_sphere_uv(rec.normal, rec.u, rec.v);
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}

__device__ bool sphere::bounding_box(float t0,
    float t1,
    aabb& box) const {
    box = aabb(center - vec3(radius, radius, radius),
        center + vec3(radius, radius, radius));
    return true;
}


class moving_sphere : public hittable {
public:
    __device__ moving_sphere() {}
    __device__ moving_sphere(vec3 cen0, vec3 cen1, float t0, float t1, float r, material* m) : center0(cen0), center1(cen1), time0(t0), time1(t1), radius(r), mat_ptr(m) {}

    __device__ virtual bool hit(const ray& r,
        float tmin,
        float tmax,
        hit_record& rec) const;
    __device__ virtual bool bounding_box(float t0,
        float t1,
        aabb& box) const;

    __device__ vec3 center(float time) const;
public:
    vec3 center0;
    vec3 center1;
    float time0, time1;
    float radius;
    material* mat_ptr;
};

__device__ bool moving_sphere::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
    vec3 oc = r.origin() - center(r.time());
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
        float sol = (-b - sqrt(discriminant)) / a;
        if (sol < tmax && sol > tmin) {
            rec.t = sol;
            rec.p = r.at(rec.t);
            rec.normal = (rec.p - center(r.time())) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        sol = (-b + sqrt(discriminant)) / a;
        if (sol < tmax && sol > tmin) {
            rec.t = sol;
            rec.p = r.at(rec.t);
            rec.normal = (rec.p - center(r.time())) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}

__device__ bool moving_sphere::bounding_box(float t0, float t1, aabb& box) const {
    aabb box0(center(t0) - vec3(radius, radius, radius), center(t0) + vec3(radius, radius, radius));
    aabb box1(center(t1) - vec3(radius, radius, radius), center(t1) + vec3(radius, radius, radius));
    box = surrounding_box(box0, box1);
    return true;
}

__device__ vec3 moving_sphere::center(float time) const {
    return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
}

#endif