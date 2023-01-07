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

    volume_render_cuvol_fused(grid, rays, opt, rgb_gt, beta_loss, sparsity_loss, rgb_out, grads);

    return;
}