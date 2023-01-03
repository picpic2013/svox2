#pragma once

#include "data_spec.hpp"
#include <cstdint>
#include <torch/extension.h>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> sample_grid(SparseGridSpec &, torch::Tensor,
                                                     bool);
void sample_grid_backward(SparseGridSpec &, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                          torch::Tensor, bool);

// ** NeRF rendering formula (trilerp)
torch::Tensor volume_render_cuvol(SparseGridSpec &, RaysSpec &, RenderOptions &);
torch::Tensor volume_render_cuvol_image(SparseGridSpec &, CameraSpec &,
                                 RenderOptions &);
void volume_render_cuvol_backward(SparseGridSpec &, RaysSpec &, RenderOptions &,
                                  torch::Tensor, torch::Tensor, GridOutputGrads &);
void volume_render_cuvol_fused(SparseGridSpec &, RaysSpec &, RenderOptions &,
                               torch::Tensor, float, float, torch::Tensor, GridOutputGrads &);
// Expected termination (depth) rendering
torch::Tensor volume_render_expected_term(SparseGridSpec &, RaysSpec &,
                                          RenderOptions &);
// Depth rendering based on sigma-threshold as in Dex-NeRF
torch::Tensor volume_render_sigma_thresh(SparseGridSpec &, RaysSpec &,
                                         RenderOptions &, float);

// ** NV rendering formula (trilerp)
torch::Tensor volume_render_nvol(SparseGridSpec &, RaysSpec &, RenderOptions &);
void volume_render_nvol_backward(SparseGridSpec &, RaysSpec &, RenderOptions &,
                                 torch::Tensor, torch::Tensor, GridOutputGrads &);
void volume_render_nvol_fused(SparseGridSpec &, RaysSpec &, RenderOptions &,
                              torch::Tensor, float, float, torch::Tensor, GridOutputGrads &);

// ** NeRF rendering formula (nearest-neighbor, infinitely many steps)
torch::Tensor volume_render_svox1(SparseGridSpec &, RaysSpec &, RenderOptions &);
void volume_render_svox1_backward(SparseGridSpec &, RaysSpec &, RenderOptions &,
                                  torch::Tensor, torch::Tensor, GridOutputGrads &);
void volume_render_svox1_fused(SparseGridSpec &, RaysSpec &, RenderOptions &,
                               torch::Tensor, float, float, torch::Tensor, GridOutputGrads &);

// Tensor volume_render_cuvol_image(SparseGridSpec &, CameraSpec &,
//                                  RenderOptions &);
//
// void volume_render_cuvol_image_backward(SparseGridSpec &, CameraSpec &,
//                                         RenderOptions &, Tensor, Tensor,
//                                         GridOutputGrads &);

// Misc
torch::Tensor dilate(torch::Tensor);
void accel_dist_prop(torch::Tensor);
void grid_weight_render(torch::Tensor, CameraSpec &, float, float, bool, torch::Tensor,
                        torch::Tensor, torch::Tensor);
// void sample_cubemap(Tensor, Tensor, bool, Tensor);

// Loss
torch::Tensor tv(torch::Tensor, torch::Tensor, int, int, bool, float, bool, float, float);
void tv_grad(torch::Tensor, torch::Tensor, int, int, float, bool, float, bool, float, float,
             torch::Tensor);
void tv_grad_sparse(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, float, bool,
                    float, bool, bool, float, float, torch::Tensor);
void msi_tv_grad_sparse(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float, torch::Tensor);
void lumisphere_tv_grad_sparse(SparseGridSpec &, torch::Tensor, torch::Tensor, torch::Tensor, float,
                               float, float, float, GridOutputGrads &);

// Optim
void rmsprop_step(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float, float, float,
                  float);
void sgd_step(torch::Tensor, torch::Tensor, torch::Tensor, float, float);