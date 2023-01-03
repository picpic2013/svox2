#pragma once

#include "data_spec.hpp"
#include "data_spec_packed.cuh"
#include "cuda_util.cuh"
#include <torch/extension.h>

#include "config.cuh"

namespace pic {

__launch_bounds__(svox2::TRACE_RAY_CUDA_THREADS, svox2::MIN_BLOCKS_PER_SM)
__global__ void render_ray_kernel(
    svox2::device::PackedSparseGridSpec grid,
    svox2::device::PackedRaysSpec rays,
    RenderOptions opt,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out,
    float* __restrict__ log_transmit_out = nullptr);

__launch_bounds__(svox2::TRACE_RAY_BG_CUDA_THREADS, svox2::MIN_BG_BLOCKS_PER_SM)
__global__ void render_background_kernel(
    svox2::device::PackedSparseGridSpec grid,
    svox2::device::PackedRaysSpec rays,
    RenderOptions opt,
    const float* __restrict__ log_transmit,
    // Outputs
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out);

__launch_bounds__(svox2::TRACE_RAY_BKWD_CUDA_THREADS, svox2::MIN_BLOCKS_PER_SM)
__global__ void render_ray_backward_kernel(
    svox2::device::PackedSparseGridSpec grid,
    const float* __restrict__ grad_output,
    const float* __restrict__ color_cache,
    svox2::device::PackedRaysSpec rays,
    RenderOptions opt,
    bool grad_out_is_rgb,
    const float* __restrict__ log_transmit_in,
    float beta_loss,
    float sparsity_loss,
    svox2::device::PackedGridOutputGrads grads,
    float* __restrict__ accum_out = nullptr,
    float* __restrict__ log_transmit_out = nullptr);

__launch_bounds__(svox2::TRACE_RAY_BG_CUDA_THREADS, svox2::MIN_BG_BLOCKS_PER_SM)
__global__ void render_background_backward_kernel(
    svox2::device::PackedSparseGridSpec grid,
    const float* __restrict__ grad_output,
    const float* __restrict__ color_cache,
    svox2::device::PackedRaysSpec rays,
    RenderOptions opt,
    const float* __restrict__ log_transmit,
    const float* __restrict__ accum,
    bool grad_out_is_rgb,
    float sparsity_loss,
    // Outputs
    svox2::device::PackedGridOutputGrads grads);

torch::Tensor _get_empty_1d(const torch::Tensor& origins);

} // namespace pic

Tensor volume_render_pic_cuvol(SparseGridSpec &, RaysSpec &, RenderOptions &);
Tensor volume_render_pic_cuvol_image(SparseGridSpec &, CameraSpec &, RenderOptions &);
void volume_render_pic_cuvol_backward(SparseGridSpec &, RaysSpec &, RenderOptions &, Tensor, Tensor, GridOutputGrads &);
void volume_render_pic_cuvol_fused(SparseGridSpec &, RaysSpec &, RenderOptions &, Tensor, float, float, Tensor, GridOutputGrads &);