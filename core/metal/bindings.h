#ifndef MSPLAT_BINDINGS_H
#define MSPLAT_BINDINGS_H

#include <tuple>
#include "metal_tensor.hpp"

// Release all cached GPU tensors (call before exit to prevent GPU memory leak)
void cleanup_msplat_metal();

// Returns the Metal device used by the msplat context (void* in C++, id<MTLDevice> in ObjC++)
#ifdef __OBJC__
id<MTLDevice> msplat_device();
#else
void* msplat_device();
#endif

// GPU tensor allocation (callable from C++ — delegates to Metal device)
MTensor gpu_zeros(std::vector<int64_t> shape, DType dtype);
MTensor gpu_empty(std::vector<int64_t> shape, DType dtype);

// Commit current command buffer (non-blocking)
void msplat_commit();

// Synchronize (commit + wait for completion)
void msplat_gpu_sync();

// GPU timing — non-invasive, uses completion handlers on committed CBs
void msplat_enable_gpu_timing(bool enable);
// Drains accumulated GPU times (ms per CB) into the provided vector. Thread-safe.
void msplat_drain_gpu_times(std::vector<double>& out);
// Drains per-stage GPU times. stage_times must be an array of N_STAGES vectors.
void msplat_drain_stage_times(std::vector<double> stage_times[], int max_stages, int& n_stages,
                              const char** stage_names);

// Render-only forward pass (no loss computation)
// Returns: out_img (H, W, 3) as MTensor
MTensor msplat_render(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background,
    MTensor &filter_3d
);

// Fused forward + backward + Adam + grad_stats in one encoder
// Returns: (radii [N], loss_value float)
std::tuple<MTensor, float> msplat_train_step(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background,
    MTensor &gt, MTensor &window2d, float ssim_weight,
    float loss_inv_n, int features_rest_bases,
    int num_adam_groups,
    MTensor adam_params[], MTensor adam_exp_avg[], MTensor adam_exp_avg_sq[],
    float adam_step_sizes[], float adam_bc2_sqrts[],
    float adam_beta1, float adam_beta2, float adam_eps,
    MTensor &vis_counts, MTensor &xys_grad_norm, MTensor &max_2d_size,
    float inv_max_dim,
    MTensor &filter_3d
);

// Compute per-Gaussian 3D smoothing filter (max focal/depth across views)
void msplat_compute_3d_filter(int N, MTensor &means3d, MTensor &viewmat,
                              float fx, float fy, MTensor &filter_3d);

// MCMC: Langevin noise injection on Gaussian means
void msplat_sgld_noise(int N, MTensor &means, MTensor &scales, MTensor &quats,
                       MTensor &opacities, MTensor &rng_states,
                       float noise_lr, float xyz_lr);

// MCMC: classify dead Gaussians and compute alive opacity
void msplat_mcmc_classify_dead(int N, MTensor &opacities, MTensor &dead_flag,
                               MTensor &alive_opacity, float dead_thresh);

// MCMC: relocate dead Gaussians by sampling from alive set
void msplat_mcmc_relocate(int N, MTensor &dead_flag, MTensor &alive_prefix_sum,
                          float total_alive_weight,
                          MTensor &means_buf, MTensor &scales_buf,
                          MTensor &quats_buf, MTensor &featuresDc_buf,
                          MTensor &featuresRest_buf, MTensor &opacities_buf,
                          int fr_stride, MTensor &rng_states,
                          MTensor adam_ea[], MTensor adam_es[]);

// MCMC: L1 regularization loss on scales and opacities
void msplat_mcmc_reg_loss(int N, MTensor &scales, MTensor &opacities,
                          MTensor &reg_out);

// Bilateral grid: forward slice (apply per-pixel affine color correction)
void msplat_bilateral_slice_forward(MTensor &rendered_img, MTensor &grid,
                                    MTensor &corrected_img,
                                    unsigned width, unsigned height,
                                    int grid_X, int grid_Y, int grid_W);

// Bilateral grid: backward slice + grid gradient scatter
void msplat_bilateral_slice_backward(MTensor &rendered_img, MTensor &grid,
                                     MTensor &dL_d_corrected,
                                     MTensor &dL_d_rendered, MTensor &dL_d_grid,
                                     unsigned width, unsigned height,
                                     int grid_X, int grid_Y, int grid_W);

// Bilateral grid: total variation regularization loss + gradient
void msplat_bilateral_tv_loss(MTensor &grid, MTensor &tv_loss,
                              MTensor &dL_d_grid,
                              int grid_X, int grid_Y, int grid_W,
                              float tv_weight);

int msplat_densify(
    int N, int buf_capacity,
    float grad_thresh, float size_thresh, float screen_thresh, int check_screen,
    float cull_alpha_thresh, float cull_scale_thresh, float cull_screen_size, int check_huge,
    MTensor &xys_grad_norm, MTensor &vis_counts, MTensor &max_2d_size,
    float half_max_dim,
    MTensor &means_buf, MTensor &scales_buf, MTensor &quats_buf,
    MTensor &featuresDc_buf, MTensor &featuresRest_buf, MTensor &opacities_buf,
    int fr_stride,
    MTensor adam_exp_avg_buf[], MTensor adam_exp_avg_sq_buf[],
    MTensor &split_flag, MTensor &dup_flag,
    MTensor &split_prefix, MTensor &dup_prefix,
    MTensor &keep_flag, MTensor &keep_prefix,
    MTensor &block_totals, MTensor &compact_scratch,
    MTensor &random_samples
);

#endif
