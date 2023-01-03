#include "pic_cuvol_render.cuh"
#include "render_util.cuh"

namespace {
// * For ray rendering
__device__ __inline__ void trace_ray_cuvol(
        const svox2::device::PackedSparseGridSpec& __restrict__ grid,
        svox2::device::SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        uint32_t lane_id,
        float* __restrict__ sphfunc_val,
        svox2::WarpReducef::TempStorage& __restrict__ temp_storage,
        float* __restrict__ out,
        float* __restrict__ out_log_transmit) {
    const uint32_t lane_colorgrp_id = lane_id % grid.basis_dim;
    const uint32_t lane_colorgrp = lane_id / grid.basis_dim;

    if (ray.tmin > ray.tmax) {
        out[lane_colorgrp] = (grid.background_nlayers == 0) ? opt.background_brightness : 0.f;
        if (out_log_transmit != nullptr) {
            *out_log_transmit = 0.f;
        }
        return;
    }

    float t = ray.tmin;
    float outv = 0.f;

    float log_transmit = 0.f;
    // printf("tmin %f, tmax %f \n", ray.tmin, ray.tmax);

    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }

        const float skip = compute_skip_dist(ray,
                       grid.links, grid.stride_x,
                       grid.size[2], 0);

        if (skip >= opt.step_size) {
            // For consistency, we skip the by step size
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }
        float sigma = svox2::device::trilerp_cuvol_one(
                grid.links, grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);
        if (opt.last_sample_opaque && t + opt.step_size > ray.tmax) {
            ray.world_step = 1e9;
        }
        // if (opt.randomize && opt.random_sigma_std > 0.0) sigma += ray.rng.randn() * opt.random_sigma_std;

        if (sigma > opt.sigma_thresh) {
            float lane_color = svox2::device::trilerp_cuvol_one(
                            grid.links,
                            grid.sh_data,
                            grid.stride_x,
                            grid.size[2],
                            grid.sh_data_dim,
                            ray.l, ray.pos, lane_id);
            lane_color *= sphfunc_val[lane_colorgrp_id]; // bank conflict

            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            float lane_color_total = svox2::WarpReducef(temp_storage).HeadSegmentedSum(
                                           lane_color, lane_colorgrp_id == 0);
            outv += weight * fmaxf(lane_color_total + 0.5f, 0.f);  // Clamp to [+0, infty)
            if (_EXP(log_transmit) < opt.stop_thresh) {
                log_transmit = -1e3f;
                break;
            }
        }
        t += opt.step_size;
    }

    if (grid.background_nlayers == 0) {
        outv += _EXP(log_transmit) * opt.background_brightness;
    }
    if (lane_colorgrp_id == 0) {
        if (out_log_transmit != nullptr) {
            *out_log_transmit = log_transmit;
        }
        out[lane_colorgrp] = outv;
    }
}

} // end of namespace

__launch_bounds__(svox2::TRACE_RAY_CUDA_THREADS, svox2::MIN_BLOCKS_PER_SM)
__global__ void pic::render_ray_kernel(
        svox2::device::PackedSparseGridSpec grid,
        svox2::device::PackedRaysSpec rays,
        RenderOptions opt,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out,
        float* __restrict__ log_transmit_out) {
    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * svox2::WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;

    if (lane_id >= grid.sh_data_dim)  // Bad, but currently the best way due to coalesced memory access
        return;

    __shared__ float sphfunc_val[svox2::TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
    __shared__ svox2::device::SingleRaySpec ray_spec[svox2::TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    __shared__ typename svox2::WarpReducef::TempStorage temp_storage[
        svox2::TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
            rays.dirs[ray_id].data());
    svox2::device::calc_sphfunc(grid, lane_id,
                 ray_id,
                 ray_spec[ray_blk_id].dir,
                 sphfunc_val[ray_blk_id]);
    svox2::device::ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
    __syncwarp((1U << grid.sh_data_dim) - 1);

    trace_ray_cuvol(
        grid,
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        out[ray_id].data(),
        log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);
}