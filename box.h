#ifndef BOX_H
#define BOX_H

#include "rect.h"
#include "hittable_list.h"
#include "material.h"

class box : public hittable{
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