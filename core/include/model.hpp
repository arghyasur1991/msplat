#ifndef MODEL_H
#define MODEL_H

#include <random>
#include "metal_tensor.hpp"
#include "ssim.hpp"
#include "input_data.hpp"

int numShBases(int degree);
float psnr(const MTensor& rendered, const MTensor& gt);
float l1_loss(const MTensor& rendered, const MTensor& gt);

enum class TrainingStrategy { Default, MCMC };

struct Model{
  Model(const InputData &inputData, int numCameras,
        int numDownscales, int resolutionSchedule, int shDegree, int shDegreeInterval,
        int refineEvery, int warmupLength, int resetAlphaEvery, float densifyGradThresh, float densifySizeThresh, int stopScreenSizeAt, float splitScreenSize,
        int maxSteps, bool keepCrs,
        bool randomBackground = true, int stopSplitAtOverride = -1,
        TrainingStrategy strategy = TrainingStrategy::Default,
        const float* bgColor = nullptr);

  ~Model(){ releaseOptimizers(); }

  void setupOptimizers();
  void releaseOptimizers();

  void schedulersStep(int step);
  int getDownscaleFactor(int step);
  void afterTrain(int step);
  void afterTrainMCMC(int step);
  void initMCMCBuffers();
  void save(const std::string &filename, int step);
  void savePly(const std::string &filename, int step);
  void saveSplat(const std::string &filename);
  int loadPly(const std::string &filename);
  void saveCheckpoint(const std::string &filename, int step);
  int loadCheckpoint(const std::string &filename);
  struct CamSetup {
    float fx, fy, cx, cy;
    int height, width, degree, degreesToUse;
    std::tuple<int,int,int> tileBounds;
    float cam_pos[3];
  };
  CamSetup prepareCam(Camera& cam, int step);
  void fullIteration(Camera& cam, int step, MTensor &gt, float ssimWeight);
  MTensor render(Camera& cam, int step);
  void compute3DFilter(std::vector<Camera>& cameras);

  MTensor means;
  MTensor scales;
  MTensor quats;
  MTensor featuresDc;
  MTensor featuresRest;
  MTensor opacities;

  static constexpr int N_ADAM_GROUPS = 6;
  MTensor adam_exp_avg[N_ADAM_GROUPS];
  MTensor adam_exp_avg_sq[N_ADAM_GROUPS];
  int adam_step_count = 0;
  float adam_lr[N_ADAM_GROUPS] = {};
  float adam_beta1 = 0.9f, adam_beta2 = 0.999f, adam_eps = 1e-8f;
  float means_lr_init = 0, means_lr_final = 0;

  MTensor means_buf, scales_buf, quats_buf, featuresDc_buf, featuresRest_buf, opacities_buf;
  MTensor adam_exp_avg_buf[N_ADAM_GROUPS], adam_exp_avg_sq_buf[N_ADAM_GROUPS];
  int num_active = 0, buf_capacity = 0;
  void refreshViews();
  void ensureCapacity(int needed);

  MTensor densify_split_flag, densify_dup_flag;
  MTensor densify_split_prefix, densify_dup_prefix;
  MTensor densify_keep_flag, densify_keep_prefix;
  MTensor densify_block_totals;
  MTensor densify_compact_scratch;
  MTensor densify_random_samples;

  MTensor radii;
  int lastHeight;
  int lastWidth;

  MTensor xysGradNorm;
  MTensor visCounts;
  MTensor max2DSize;

  MTensor backgroundColor;
  MTensor window2d;  // SSIM window (11,11) f32

  int numCameras;
  int numDownscales;
  int resolutionSchedule;
  int shDegree;
  int shDegreeInterval;
  int refineEvery;
  int warmupLength;
  int resetAlphaEvery;
  int stopSplitAt;
  float densifyGradThresh;
  float densifySizeThresh;
  int stopScreenSizeAt;
  float splitScreenSize;
  int maxSteps;
  bool keepCrs;
  bool randomBackground = true;

  TrainingStrategy strategy = TrainingStrategy::Default;
  float mcmc_noise_lr = 5e5f;
  float mcmc_dead_thresh = 0.005f;
  float mcmc_scale_reg = 0.01f;
  float mcmc_opacity_reg = 0.01f;
  int mcmc_cap_max = 1000000;

  // MCMC state buffers
  MTensor rng_states;      // [N] uint2 per-gaussian RNG
  MTensor dead_flag;       // [N] int32
  MTensor alive_opacity;   // [N] float — for multinomial sampling
  MTensor alive_prefix_sum; // [N] float — prefix sum of alive opacities
  MTensor reg_out;          // [2] float — scale_reg_sum, opacity_reg_sum

  bool use_3d_filter = false;
  MTensor filter_3d;
  MTensor filter_3d_buf;

  bool use_bilateral_grid = false;
  int grid_X = 16, grid_Y = 16, grid_W = 8;
  float bilateral_tv_weight = 10.0f;
  float bilateral_lr = 1e-3f;
  std::vector<MTensor> bilateral_grids;
  std::vector<MTensor> bilateral_adam_m;
  std::vector<MTensor> bilateral_adam_v;
  int bilateral_adam_step = 0;

  void initBilateralGrids();

  std::mt19937 bgRng{42};

  float scale;
  float translation[3] = {};
};

#endif
