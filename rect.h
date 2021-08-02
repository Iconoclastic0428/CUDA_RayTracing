
#ifndef RECTANGLEH
#define RECTANGLEH

#include "vec3.h"
#include "material.h"
#include "aabb.h"
#include "hittable_list.h"

/**
 * Rectangle along XY axes
 */
class rectangle_xy : public hittable {
public:
    __device__ rectangle_xy() {};
    __device__ rectangle_xy(float _x0, float _x1, float _y0, float _y1, float _k, material* mat):
        x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mat_ptr(mat) {};

    __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const {
        box = aabb(vec3(x0, y0, k - 0.0001), vec3(x1, y1, k + 0.0001));
        return true;
    }

    float x0, x1, y0, y1, k;
    material* mat_ptr;
};


__device__ bool rectangle_xy::hit(const ray& r, float t0, float t1, hit_record& rec) const {
    float t = (k - r.origin().z()) / r.direction().z();
    if (t < t0 || t > t1) return false;

    float x = r.origin().x() + t * r.direction().x();
    float y = r.origin().y() + t * r.direction().y();
    if (x < x0 || x > x1 || y < y0 || y > y1) return false;

    rec.t = t;
    rec.mat_ptr = mat_ptr;
    rec.p = r.at(t);
    auto outward_normal = vec3(0, 0, 1);
    bool front_face = dot(r.direction(), outward_normal) < 0;
    rec.normal = front_face ? outward_normal : -outward_normal;
    return true;
}

/**
 * Rectangle along XZ axes
 */
class rectangle_xz : public hittable
{
public:
    __device__ rectangle_xz() {};
    __device__ rectangle_xz(float _x0, float _x1, float _z0, float _z1, float _k, material* mat) :
        x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mat_ptr(mat) {};

    __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const {
        box = aabb(vec3(x0, k - 0.0001, z0), vec3(x1, k + 0.0001, z1));
        return true;
    }

    float x0, x1, z0, z1, k;
    material* mat_ptr;
};


__device__ bool rectangle_xz::hit(const ray& r, float t0, float t1, hit_record& rec) const {
    float t = (k - r.origin().y()) / r.direction().y();
    if (t < t0 || t > t1) return false;

    float x = r.origin().x() + t * r.direction().x();
    float z = r.origin().z() + t * r.direction().z();
    if (x < x0 || x > x1 || z < z0 || z > z1) return false;

    rec.t = t;
    rec.mat_ptr = mat_ptr;
    rec.p = r.at(t);
    auto outward_normal = vec3(0, 1, 0);
    bool front_face = dot(r.direction(), outward_normal) < 0;
    rec.normal = front_face ? outward_normal : -outward_normal;
    return true;
}


/**
 * Rectangle along YZ axes
 */
class rectangle_yz : public hittable {
public:
    __device__ rectangle_yz() {};
    __device__ rectangle_yz(float _y0, float _y1, float _z0, float _z1, float _k, material* mat) :
        y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mat_ptr(mat) {};

    __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const {
        box = aabb(vec3(k - 0.0001, y0, z0), vec3(k + 0.0001, y1, z1));
        return true;
    }

    float y0, y1, z0, z1, k;
    material* mat_ptr;
};


__device__ bool rectangle_yz::hit(const ray& r, float t0, float t1, hit_record& rec) const {
    float t = (k - r.origin().x()) / r.direction().x();
    if (t < t0 || t > t1) return false;

    float y = r.origin().y() + t * r.direction().y();
    float z = r.origin().z() + t * r.direction().z();
    if (y < y0 || y > y1 || z < z0 || z > z1) return false;

    rec.t = t;
    rec.mat_ptr = mat_ptr;
    rec.p = r.at(t);
    auto outward_normal = vec3(1, 0, 0);
    bool front_face = dot(r.direction(), outward_normal) < 0;
    rec.normal = front_face ? outward_normal : -outward_normal;
    return true;
}

class box : public hittable {
public:
    __device__ box() {}
    __device__ box(const vec3& p0, const vec3& p1, material* ptr);
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const {
        box = aabb(box_min, box_max);
        return true;
    }

    vec3 box_min;
    vec3 box_max;
    hittable* sides;
};

__device__ box::box(const vec3& p0, const vec3& p1, material* ptr) {
    box_min = p0;
    box_max = p1;

    hittable** list = new hittable * [6];
    list[0] = new rectangle_xy(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr);
    list[1] = new rectangle_xy(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr);
    list[2] = new rectangle_xz(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr);
    list[3] = new rectangle_xz(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr);
    list[4] = new rectangle_yz(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr);
    list[5] = new rectangle_yz(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr);
    sides = new hittable_list(list, 6);
}

__device__ bool box::hit(const ray& r,
    float t_min,
    float t_max,
    hit_record& rec) const {
    return sides->hit(r, t_min, t_max, rec);
}

#endif