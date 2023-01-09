#include "pic_cuvol_render.cuh"
#include "render_util.cuh"

namespace {

template<class data_type_t, class voxel_index_t>
__device__ __inline__ void trilerp_backward_counter(
    data_type_t* __restrict__ counter_data,
    int offx, int offy, 
    const voxel_index_t* __restrict__ l,
    const float* __restrict__ pos,
    float rayLoss
) {
    const float ay = 1.f - pos[1], az = 1.f - pos[2];
    float xo = (1.0f - pos[0]) * rayLoss;

    data_type_t* __restrict__ link_ptr = counter_data + (offx * l[0] + offy * l[1] + l[2]);

#define MAYBE_ADD_LINK(u, val) atomicAdd(&link_ptr[u], val)
    MAYBE_ADD_LINK(0, ay * az * xo);
    MAYBE_ADD_LINK(1, ay * pos[2] * xo);
    MAYBE_ADD_LINK(offy, pos[1] * az * xo);
    MAYBE_ADD_LINK(offy + 1, pos[1] * pos[2] * xo);

    xo = pos[0] * rayLoss;
    MAYBE_ADD_LINK(offx + 0, ay * az * xo);
    MAYBE_ADD_LINK(offx + 1, ay * pos[2] * xo);
    MAYBE_ADD_LINK(offx + offy, pos[1] * az * xo);
    MAYBE_ADD_LINK(offx + offy + 1, pos[1] * pos[2] * xo);
#undef MAYBE_ADD_LINK
}

__device__ __inline__ void trace_ray_cuvol_backward(
    const svox2::device::PackedSparseGridSpec& __restrict__ grid,
    const float* __restrict__ grad_output,
    const float* __restrict__ color_cache,
    svox2::device::SingleRaySpec& __restrict__ ray,
    const RenderOptions& __restrict__ opt,
    uint32_t lane_id,
    const float* __restrict__ sphfunc_val,
    float* __restrict__ grad_sphfunc_val,
    svox2::WarpReducef::TempStorage& __restrict__ temp_storage,
    float log_transmit_in,
    float beta_loss,
    float sparsity_loss,
    svox2::device::PackedGridOutputGrads& __restrict__ grads,
    float* __restrict__ counterOutput,
    float* __restrict__ grad_color_out,
    int rayLossSpreadType, 
    float rayLoss, 
    float* __restrict__ accum_out,
    float* __restrict__ log_transmit_out
) {
    const uint32_t lane_colorgrp_id = lane_id % grid.basis_dim;
    const uint32_t lane_colorgrp = lane_id / grid.basis_dim;
    const uint32_t leader_mask = 1U | (1U << grid.basis_dim) | (1U << (2 * grid.basis_dim));

    float accum = fmaf(color_cache[0], grad_output[0],
                      fmaf(color_cache[1], grad_output[1],
                           color_cache[2] * grad_output[2]));

    if (beta_loss > 0.f) {
        const float transmit_in = _EXP(log_transmit_in);
        beta_loss *= (1 - transmit_in / (1 - transmit_in + 1e-3)); // d beta_loss / d log_transmit_in
        accum += beta_loss;
        // Interesting how this loss turns out, kinda nice?
    }

    if (ray.tmin > ray.tmax) {
        if (accum_out != nullptr) { *accum_out = accum; }
        if (log_transmit_out != nullptr) { *log_transmit_out = 0.f; }
        // printf("accum_end_fg_fast=%f\n", accum);
        return;
    }
    float t = ray.tmin;

    const float gout = grad_output[lane_colorgrp];

    float log_transmit = 0.f;

    // remat samples
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
                grid.links,
                grid.density_data,
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
            float weighted_lane_color = lane_color * sphfunc_val[lane_colorgrp_id];

            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            const float lane_color_total = svox2::WarpReducef(temp_storage).HeadSegmentedSum(
                                           weighted_lane_color, lane_colorgrp_id == 0) + 0.5f;
            float total_color = fmaxf(lane_color_total, 0.f);
            float color_in_01 = total_color == lane_color_total;
            total_color *= gout; // Clamp to [+0, infty)

            float total_color_c1 = __shfl_sync(leader_mask, total_color, grid.basis_dim);
            total_color += __shfl_sync(leader_mask, total_color, 2 * grid.basis_dim);
            total_color += total_color_c1;

            color_in_01 = __shfl_sync((1U << grid.sh_data_dim) - 1, color_in_01, lane_colorgrp * grid.basis_dim);
            const float grad_common = weight * color_in_01 * gout;
            const float curr_grad_color = sphfunc_val[lane_colorgrp_id] * grad_common;

            if (grid.basis_type != BASIS_TYPE_SH) {
                float curr_grad_sphfunc = lane_color * grad_common;
                const float curr_grad_up2 = __shfl_down_sync((1U << grid.sh_data_dim) - 1,
                        curr_grad_sphfunc, 2 * grid.basis_dim);
                curr_grad_sphfunc += __shfl_down_sync((1U << grid.sh_data_dim) - 1,
                        curr_grad_sphfunc, grid.basis_dim);
                curr_grad_sphfunc += curr_grad_up2;
                if (lane_id < grid.basis_dim) {
                    grad_sphfunc_val[lane_id] += curr_grad_sphfunc;
                }
            }

            accum -= weight * total_color;
            float curr_grad_sigma = ray.world_step * (
                    total_color * _EXP(log_transmit) - accum);
            if (sparsity_loss > 0.f) {
                // Cauchy version (from SNeRG)
                curr_grad_sigma += sparsity_loss * (4 * sigma / (1 + 2 * (sigma * sigma)));

                // Alphs version (from PlenOctrees)
                // curr_grad_sigma += sparsity_loss * _EXP(-pcnt) * ray.world_step;
            }
            svox2::device::trilerp_backward_cuvol_one(grid.links, grads.grad_sh_out,
                    grid.stride_x,
                    grid.size[2],
                    grid.sh_data_dim,
                    ray.l, ray.pos,
                    curr_grad_color, lane_id);
            if (lane_id == 0) {
                svox2::device::trilerp_backward_cuvol_one_density(
                        grid.links,
                        grads.grad_density_out,
                        grads.mask_out,
                        grid.stride_x,
                        grid.size[2],
                        ray.l, ray.pos, curr_grad_sigma);
            
            
                // spread ray loss to grid counter tensor
                if (rayLossSpreadType) {
                    trilerp_backward_counter(counterOutput, grid.stride_x, grid.size[2], ray.l, ray.pos, weight);
                    trilerp_backward_counter(grad_color_out, grid.stride_x, grid.size[2], ray.l, ray.pos, weight * rayLoss);
                }
            }
            if (_EXP(log_transmit) < opt.stop_thresh) {
                break;
            }
        }
        t += opt.step_size;
    }
    if (lane_id == 0) {
        if (accum_out != nullptr) {
            // Cancel beta loss out in case of background
            accum -= beta_loss;
            *accum_out = accum;
        }
        if (log_transmit_out != nullptr) { *log_transmit_out = log_transmit; }
        // printf("accum_end_fg=%f\n", accum);
        // printf("log_transmit_fg=%f\n", log_transmit);
    }
}

} // end of namespace

__launch_bounds__(svox2::TRACE_RAY_BKWD_CUDA_THREADS, svox2::MIN_BLOCKS_PER_SM)
__global__ void pic::render_ray_backward_kernel(
    svox2::device::PackedSparseGridSpec grid,
    const float* __restrict__ grad_output, // GT
    const float* __restrict__ color_cache, // OUT
    svox2::device::PackedRaysSpec rays,
    RenderOptions opt,
    bool grad_out_is_rgb,
    const float* __restrict__ log_transmit_in,
    float beta_loss,
    float sparsity_loss,
    svox2::device::PackedGridOutputGrads grads,
    float* __restrict__ counterOutput,
    float* __restrict__ grad_color_out,
    int rayLossSpreadType, 
    float* __restrict__ accum_out,
    float* __restrict__ log_transmit_out) {
    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * svox2::WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;

    if (lane_id >= grid.sh_data_dim)
        return;

    __shared__ float sphfunc_val[svox2::TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK][9];
    __shared__ float grad_sphfunc_val[svox2::TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
    __shared__ svox2::device::SingleRaySpec ray_spec[svox2::TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
    __shared__ typename svox2::WarpReducef::TempStorage temp_storage[
        svox2::TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                             rays.dirs[ray_id].data());
    const float vdir[3] = {ray_spec[ray_blk_id].dir[0],
                     ray_spec[ray_blk_id].dir[1],
                     ray_spec[ray_blk_id].dir[2] };
    if (lane_id < grid.basis_dim) {
        grad_sphfunc_val[ray_blk_id][lane_id] = 0.f;
    }
    calc_sphfunc(grid, lane_id,
                 ray_id,
                 vdir, sphfunc_val[ray_blk_id]);
    if (lane_id == 0) {
        ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
    }

    float grad_out[3];
    float rayMSELoss = 0.;
    if (grad_out_is_rgb) {
        const float norm_factor = 2.f / (3 * int(rays.origins.size(0)));
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            const float resid = color_cache[ray_id * 3 + i] - grad_output[ray_id * 3 + i];
            grad_out[i] = resid * norm_factor;
            rayMSELoss += resid * resid;
        }
    } else {
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            const float resid = color_cache[ray_id * 3 + i] - grad_output[ray_id * 3 + i];
            grad_out[i] = grad_output[ray_id * 3 + i];
            rayMSELoss += resid * resid;
        }
    }

    __syncwarp((1U << grid.sh_data_dim) - 1);
    trace_ray_cuvol_backward(
        grid,
        grad_out,
        color_cache + ray_id * 3,
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        grad_sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        log_transmit_in == nullptr ? 0.f : log_transmit_in[ray_id],
        beta_loss,
        sparsity_loss,
        grads,
        counterOutput, 
        grad_color_out, 
        rayLossSpreadType, 
        rayMSELoss, 
        accum_out == nullptr ? nullptr : accum_out + ray_id,
        log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);
    svox2::device::calc_sphfunc_backward(
                 grid, lane_id,
                 ray_id,
                 vdir,
                 sphfunc_val[ray_blk_id],
                 grad_sphfunc_val[ray_blk_id],
                 grads.grad_basis_out);
}