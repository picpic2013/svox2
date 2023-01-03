#pragma once

namespace svox2 {

const int WARP_SIZE = 32;

const int TRACE_RAY_CUDA_THREADS = 128;
const int TRACE_RAY_CUDA_RAYS_PER_BLOCK = TRACE_RAY_CUDA_THREADS / WARP_SIZE;

const int TRACE_RAY_BKWD_CUDA_THREADS = 128;
const int TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK = TRACE_RAY_BKWD_CUDA_THREADS / WARP_SIZE;

const int MIN_BLOCKS_PER_SM = 8;

const int TRACE_RAY_BG_CUDA_THREADS = 128;
const int MIN_BG_BLOCKS_PER_SM = 8;
typedef cub::WarpReduce<float> WarpReducef;

}