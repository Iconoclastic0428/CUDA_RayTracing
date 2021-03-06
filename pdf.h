#ifndef PDF_H
#define PDF_H

#include "vec3.h"
#include "onb.h"
#include "hittable.h"

class pdf {
public:
	__device__ virtual ~pdf() {}
	__device__ virtual float value(const vec3& direction) const = 0;
	__device__ virtual vec3 generate() const = 0;
};

class cosine_pdf : public pdf {
public:
	__device__ cosine_pdf(const vec3& w) {
		uvw.build_from_w(w);
	}
	__device__ virtual float value(const vec3& direction) const override {
		auto cosine = dot(unit_vector(direction), uvw.w());
		return (cosine <= 0) ? 0 : cosine / M_PI;
	}
	__device__ virtual vec3 generate() const override {
		return uvw.local(random_cosine_direction());
	}

	onb uvw;
};

class hittable_pdf : public pdf {
public:
	__device__ hittable_pdf(hittable* p, const vec3& origin) : ptr(p), o(origin) {}

	__device__ virtual double value(const vec3& direction) const override {
		return ptr->pdf_value(o, direction);
	}

	__device__ virtual vec3 generate() const override {
		return ptr->random(o);
	}

public:
	vec3 o;
	hittable* ptr;
};

#endif