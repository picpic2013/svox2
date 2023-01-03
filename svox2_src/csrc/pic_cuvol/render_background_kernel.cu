#include "pic_cuvol_render.cuh"
#include "render_util.cuh"

namespace {
__device__ __inline__ void render_background_forward(
            const svox2::device::PackedSparseGridSpec& __restrict__ grid,
            svox2::device::SingleRaySpec& __restrict__ ray,
            const RenderOptions& __restrict__ opt,
            float log_transmit,
            float* __restrict__ out
        ) {

    svox2::device::ConcentricSpheresIntersector csi(ray.origin, ray.dir);

    const float inner_radius = fmaxf(_dist_ray_to_origin(ray.origin, ray.dir) + 1e-3f, 1.f);
    float t, invr_last = 1.f / inner_radius;
    const int n_steps = int(grid.background_nlayers / opt.step_size) + 2;

    // csi.intersect(inner_radius, &t_last);

    float outv[3] = {0.f, 0.f, 0.f};
    for (int i = 0; i < n_steps; ++i) {
        // Between 1 and infty
        float r = n_steps / (n_steps - i - 0.5);
        if (r < inner_radius || !csi.intersect(r, &t)) continue;

#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
        }
        const float invr_mid = _rnorm(ray.pos);
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] *= invr_mid;
        }
        // NOTE: reusing ray.pos (ok if you check _unitvec2equirect)
        svox2::device::_unitvec2equirect(ray.pos, grid.background_reso, ray.pos);
        ray.pos[2] = fminf(fmaxf((1.f - invr_mid) * grid.background_nlayers - 0.5f, 0.f),
                       grid.background_nlayers - 1);
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.l[j] = (int) ray.pos[j];
        }
        ray.l[0] = min(ray.l[0], grid.background_reso * 2 - 1);
        ray.l[1] = min(ray.l[1], grid.background_reso - 1);
        ray.l[2] = min(ray.l[2], grid.background_nlayers - 2);
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] -= ray.l[j];
        }

        float sigma = svox2::device::trilerp_bg_one(
                grid.background_links,
                grid.background_data,
                grid.background_reso,
                grid.background_nlayers,
                4,
                ray.l,
                ray.pos,
                3);

        // if (i == n_steps - 1) {
        //     ray.world_step = 1e9;
        // }
        // if (opt.randomize && opt.random_sigma_std_background > 0.0)
        //     sigma += ray.rng.randn() * opt.random_sigma_std_background;
        if (sigma > 0.f) {
            const float pcnt = (invr_last - invr_mid) * ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;
#pragma unroll 3
            for (int i = 0; i < 3; ++i) {
                // Not efficient
                const float color = svox2::device::trilerp_bg_one(
                        grid.background_links,
                        grid.background_data,
                        grid.background_reso,
                        grid.background_nlayers,
                        4,
                        ray.l,
                        ray.pos,
                        i) * svox2::device::C0;  // Scale by SH DC factor to help normalize lrs
                outv[i] += weight * fmaxf(color + 0.5f, 0.f);  // Clamp to [+0, infty)
            }
            if (_EXP(log_transmit) < opt.stop_thresh) {
                break;
            }
        }
        invr_last = invr_mid;
    }
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        out[i] += outv[i] + _EXP(log_transmit) * opt.background_brightness;
    }
}

} // end of namespace

__launch_bounds__(svox2::TRACE_RAY_BG_CUDA_THREADS, svox2::MIN_BG_BLOCKS_PER_SM)
__global__ void pic::render_background_kernel(
        svox2::device::PackedSparseGridSpec grid,
        svox2::device::PackedRaysSpec rays,
        RenderOptions opt,
        const float* __restrict__ log_transmit,
        // Outputs
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out) {
    CUDA_GET_THREAD_ID(ray_id, int(rays.origins.size(0)));
    if (log_transmit[ray_id] < -25.f) return;
    svox2::device::SingleRaySpec ray_spec(rays.origins[ray_id].data(), rays.dirs[ray_id].data());
    svox2::device::ray_find_bounds_bg(ray_spec, grid, opt, ray_id);
    render_background_forward(
        grid,
        ray_spec,
        opt,
        log_transmit[ray_id],
        out[ray_id].data());
}