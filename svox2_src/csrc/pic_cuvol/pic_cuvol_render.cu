#include "pic_cuvol_render.cuh"
#include "render_util.cuh"
#include "svox2.cuh"

#include <cstdio>

torch::Tensor volume_render_pic_cuvol(SparseGridSpec& grid, RaysSpec& rays, RenderOptions& opt) {
    // printf("volume_render_pic_cuvol");
    return volume_render_cuvol(grid, rays, opt);
}

torch::Tensor volume_render_pic_cuvol_image(SparseGridSpec& grid, CameraSpec& cam, RenderOptions& opt) {
    // printf("volume_render_pic_cuvol_image");
    return volume_render_cuvol_image(grid, cam, opt);
}

void volume_render_pic_cuvol_backward(
        SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        torch::Tensor grad_out,
        torch::Tensor color_cache,
        GridOutputGrads& grads) {
    
    // printf("volume_render_pic_cuvol_backward");
    volume_render_cuvol_backward(grid, rays, opt, grad_out, color_cache, grads);
}

__host__ void volume_render_pic_cuvol_fused(
    SparseGridSpec& grid,
    RaysSpec& rays,
    RenderOptions& opt,
    torch::Tensor rgb_gt,
    float beta_loss,
    float sparsity_loss,
    torch::Tensor rgb_out,
    GridOutputGrads& grads) {
    
    DEVICE_GUARD(grid.sh_data);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(rgb_out);
    grid.check();
    rays.check();
    grads.check();
    const auto Q = rays.origins.size(0);

    bool use_background = grid.background_links.defined() &&
                          grid.background_links.size(0) > 0;
    bool need_log_transmit = use_background || beta_loss > 0.f;
    torch::Tensor log_transmit, accum;
    if (need_log_transmit) {
        log_transmit = pic::_get_empty_1d(rays.origins);
    }
    if (use_background) {
        accum = pic::_get_empty_1d(rays.origins);
    }

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * svox2::WARP_SIZE, svox2::TRACE_RAY_CUDA_THREADS);
        pic::render_ray_kernel<<<blocks, svox2::TRACE_RAY_CUDA_THREADS>>>(
                grid, rays, opt,
                // Output
                rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                need_log_transmit ? log_transmit.data_ptr<float>() : nullptr);
    }

    if (use_background) {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, svox2::TRACE_RAY_BG_CUDA_THREADS);
        pic::render_background_kernel<<<blocks, svox2::TRACE_RAY_BG_CUDA_THREADS>>>(
                grid,
                rays,
                opt,
                log_transmit.data_ptr<float>(),
                rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    }

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * svox2::WARP_SIZE, svox2::TRACE_RAY_BKWD_CUDA_THREADS);
        pic::render_ray_backward_kernel<<<blocks, svox2::TRACE_RAY_BKWD_CUDA_THREADS>>>(
                grid,
                rgb_gt.data_ptr<float>(),
                rgb_out.data_ptr<float>(),
                rays, opt,
                true,
                beta_loss > 0.f ? log_transmit.data_ptr<float>() : nullptr,
                beta_loss / Q,
                sparsity_loss,
                // Output
                grads,
                use_background ? accum.data_ptr<float>() : nullptr,
                nullptr);
    }

    if (use_background) {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, svox2::TRACE_RAY_BG_CUDA_THREADS);
        pic::render_background_backward_kernel<<<blocks, svox2::TRACE_RAY_BG_CUDA_THREADS>>>(
                grid,
                rgb_gt.data_ptr<float>(),
                rgb_out.data_ptr<float>(),
                rays,
                opt,
                log_transmit.data_ptr<float>(),
                accum.data_ptr<float>(),
                true,
                sparsity_loss,
                // Output
                grads);
    }

    CUDA_CHECK_ERRORS;

    // volume_render_cuvol_fused(grid, rays, opt, rgb_gt, beta_loss, sparsity_loss, rgb_out, grads);

    return;
}