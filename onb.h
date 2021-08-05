#ifndef ONB_H
#define ONB_H

#include "vec3.h"

class onb {
public:
    __device__ onb() {}

    __device__ inline vec3 operator[](int i) const { return axis[i]; }

    __device__ vec3 u() const { return axis[0]; }
    __device__ vec3 v() const { return axis[1]; }
    __device__ vec3 w() const { return axis[2]; }

    __device__ vec3 local(double a, double b, double c) const {
        return a * u() + b * v() + c * w();
    }

    __device__ vec3 local(const vec3& a) const {
        return a.x() * u() + a.y() * v() + a.z() * w();
    }

    __device__ void build_from_w(const vec3&);

public:
    vec3 axis[3];
};


__device__ void onb::build_from_w(const vec3& n) {
    axis[2] = unit_vector(n);
    vec3 a = (fabs(w().x()) > 0.9) ? vec3(0, 1, 0) : vec3(1, 0, 0);
    axis[1] = unit_vector(cross(w(), a));
    axis[0] = cross(w(), v());
}

__device__ vec3 random_cosine_direction(curandState* local_rand_state) {
    auto r1 = curand_uniform(local_rand_state);
    auto r2 = curand_uniform(local_rand_state);
    auto z = sqrt(1 - r2);

    auto phi = 2 * M_PI * r1;
    auto x = cos(phi) * sqrt(r2);
    auto y = sin(phi) * sqrt(r2);

    return vec3(x, y, z);
}

#endif