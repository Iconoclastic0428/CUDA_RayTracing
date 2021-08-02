#include <iostream>
#include <time.h>
#include <fstream>
#include <curand_kernel.h>

#include "ray.h"
#include "vec3.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"
#include "rect.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

__device__ vec3 color(const ray& r,
    hittable** world,
    curandState* state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.f, 1.f, 1.f);

    for (int i = 0; i < 50; ++i) {
        hit_record rec;
        if (!((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec))) {
            return vec3(0.f, 0.f, 0.f);
        }
        ray scattered;
        vec3 attenuation;
        vec3 emitted = rec.mat_ptr->emitted(0., 0., rec.p);
        if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, state)) {
            cur_attenuation *= attenuation;
            cur_ray = scattered;
        }
        else {
            cur_attenuation = emitted * cur_attenuation;
            return cur_attenuation;
        }
    }
    return vec3(0.f, 0.f, 0.f);
}

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* fb, int max_x, int max_y, int ns, camera** cam, hittable** world, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
            new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f) {
                    auto center2 = center + vec3(0, RND * RND, 0);
                    d_list[i++] = new moving_sphere(center, center2, 0.f, 1.f, 0.2,
                        new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
                }
                else if (choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                        new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world = new hittable_list(d_list, 22 * 22 + 1 + 3);

        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0; (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            30.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus,
            0.f,
            1.f);
    }
}

__global__ void free_world(hittable** d_list, hittable** d_world, camera** d_cam) {
    for (int i = 0; i < 8; ++i) {
        delete* (d_list + i);
    }
    delete *(d_world);
    delete* (d_cam);
}

__global__ void simple_light(hittable** d_list, hittable** d_world, camera** d_cam, int nx, int ny) {

    d_list[0] = new rectangle_xz(-10, 10, -10, 10, 0, new lambertian(vec3(0.5, 0.5, 0.5)));
    d_list[1] = new sphere(vec3(0, 2, 0), 2, new lambertian(vec3(0.4, 0.2, 0.1)));
    d_list[2] = new rectangle_xy(3, 5, 1, 3, -2, new diffuse_light(vec3(4, 4, 4)));
    d_list[3] = new sphere(vec3(0, 7, 0), 2, new diffuse_light(vec3(4, 4, 4)));
    *d_world = new hittable_list(d_list, 4);
    *d_cam = new camera(vec3(26, 3, 6), vec3(0, 2, 0), vec3(0, 1, 0), 30.f, float(nx) / float(ny), 0., 10., 0.f, 0.f);
}

__global__ void cornell_box(hittable** d_list, hittable** d_world, camera** d_cam, int nx, int ny) {

    d_list[0] = new rectangle_yz(0, 555, 0, 555, 555, new lambertian(vec3(.12, .45, .15)));
    d_list[1] = new rectangle_yz(0, 555, 0, 555, 0, new lambertian(vec3(.65, .05, .05)));
    d_list[2] = new rectangle_xz(213, 343, 227, 332, 554, new diffuse_light(vec3(15, 15, 15)));
    d_list[3] = new rectangle_xz(0, 555, 0, 555, 0, new lambertian(vec3(0.73, 0.73, 0.73)));
    d_list[4] = new rectangle_xz(0, 555, 0, 555, 555, new lambertian(vec3(0.73, 0.73, 0.73)));
    d_list[5] = new rectangle_xy(0, 555, 0, 555, 555, new lambertian(vec3(0.73, 0.73, 0.73)));
    d_list[6] = new box(vec3(130, 0, 65), vec3(295, 165, 230), new lambertian(vec3(0.73, 0.73, 0.73)));
    d_list[7] = new box(vec3(265, 0, 295), vec3(430, 330, 460), new lambertian(vec3(0.73, 0.73, 0.73)));
    *d_world = new hittable_list(d_list, 8);
    *d_cam = new camera(vec3(278, 278, -800), vec3(278, 278, 0), vec3(0, 1, 0), 40.f, float(nx) / float(ny), 0.f, 10.f, 0.f, 0.f);

}

int main() {
    int nx = 600;
    int ny = 600;
    int tx = 8;
    int ty = 8;
    int ns = 400;

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    curandState* d_rand_state;
    checkCudaErrors(cudaMallocManaged((void**)&d_rand_state, num_pixels * sizeof(curandState)));


    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init << <blocks, threads >> > (nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // allocate FB
    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));
    hittable** d_list;
    checkCudaErrors(cudaMallocManaged((void**)&d_list, 8 * sizeof(hittable*)));
    hittable** d_world;
    checkCudaErrors(cudaMallocManaged((void**)&d_world, sizeof(hittable*)));
    camera** d_camera;
    checkCudaErrors(cudaMallocManaged((void**)&d_camera, sizeof(camera*)));
    cornell_box << <1, 1 >> > (d_list, d_world, d_camera, nx, ny);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    clock_t start, stop;
    start = clock();
    // Render our buffer
    render << <blocks, threads >> > (fb, nx, ny, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::ofstream image("image.ppm");
    // Output FB as Image
    image << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            int ir = int(255.99 * fb[pixel_index].r());
            int ig = int(255.99 * fb[pixel_index].g());
            int ib = int(255.99 * fb[pixel_index].b());
            image << ir << " " << ig << " " << ib << "\n";
        }
    }
    image.close();

    checkCudaErrors(cudaDeviceSynchronize());
    free_world << <1, 1 >> > (d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(fb));
    cudaDeviceReset();

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";
}