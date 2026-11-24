#include <cuda_runtime.h>

constexpr int TILE = 4;
constexpr int KVTILE = 16;
constexpr int D = 128;

__global__ void attention(const float* Q, const float* K, const float* V, float* output, int M, int N,
                           int dims) {
    int row = blockIdx.y * TILE;
    auto lane = threadIdx.x % 32;

    __shared__ float Qs[TILE][D];
    __shared__ float Ks[KVTILE][D];
    __shared__ float Vs[KVTILE][D];

    // 32 threads
    // 4 tiles

    float invSqrtDims = rsqrtf(dims);

    for (int r = threadIdx.y; r < TILE; r += blockDim.y) {
        for (auto d = threadIdx.x; d < dims; d += blockDim.x) {
            if (row + r < M) {
                Qs[r][d] = Q[(row + r) * dims + d];
            } else {
                Qs[r][d] = 0.0;
            }
        }
    }

    int numTiles = (N + KVTILE - 1) / KVTILE;

    __shared__ float accs[TILE][D];
    __shared__ float sums[TILE];
    __shared__ float maxs[TILE];
    for (int r = threadIdx.y; r < TILE; r += blockDim.y) {
        for (auto d = threadIdx.x; d < dims; d += blockDim.x) {
            accs[r][d] = 0.0;
        }
    }
    
    for (int r = threadIdx.y; r < TILE; r += blockDim.y) {
        maxs[r] = -INFINITY;
        sums[r] = 0.0;
    }
    __syncthreads();

    // Load K/V tiles
    for (int t = 0; t < numTiles; t++) {
        for (int r = threadIdx.y; r < KVTILE; r += blockDim.y) {
            for (int d = threadIdx.x; d < dims; d += blockDim.x) {    
                if (KVTILE * t + r < N) {
                    Ks[r][d] = K[(KVTILE * t + r) * dims + d];
                    Vs[r][d] = V[(KVTILE * t + r) * dims + d];
                } else {
                    Ks[r][d] = 0.0;
                    Vs[r][d] = 0.0;
                }
            }
        }
        __syncthreads();

        // Calculate QKT
        float scores[TILE][KVTILE] = {0.0};    
        for (int r = threadIdx.y; r < TILE; r += blockDim.y) {
            auto maxF = maxs[r];
            for (int c = threadIdx.x; c < KVTILE; c += blockDim.x) {
                if (t * KVTILE + c >= N) {
                    continue;
                }
                for (int d = 0; d < dims; d++) {
                    scores[r][c] += Qs[r][d] * Ks[c][d];
                }
                scores[r][c] *= invSqrtDims;
                maxF = max(maxF, scores[r][c]);
            }
            for (auto offset = 16; offset > 0; offset >>= 1) {
                maxF = max(maxF, __shfl_down_sync(0xffffffff, maxF, offset));
            }
            maxF = __shfl_sync(0xffffffff, maxF, 0);

            auto total = 0.0;
            for (int c = threadIdx.x; c < KVTILE; c += blockDim.x) {
                if (t * TILE + c >= N) {
                    scores[r][c] = 0.0f;
                    continue;
                }
                auto score = expf(scores[r][c] - maxF);
                scores[r][c] = score;
                total += score;
            }
            for (auto offset = 16; offset > 0; offset >>= 1) {
                total += __shfl_down_sync(0xffffffff, total, offset);
            }
            
            auto scale = expf(maxs[r] - maxF);
            if (lane == 0) {
                sums[r] *= scale;
                sums[r] += total;
                maxs[r] = maxF;
            }
            for (int d = threadIdx.x; d < dims; d += blockDim.x) {
                accs[r][d] *= scale;
            }
            __syncthreads();
        }

        // Calculate (QKT) @ V
        for (int r = threadIdx.y; r < TILE; r += blockDim.y) {
            for (int d = 0; d < dims; d++) {
                auto localScore = 0.0f;
                for (int c = threadIdx.x; c < KVTILE; c += blockDim.x) {
                    localScore += scores[r][c] * Vs[c][d];
                }
                for (auto offset = 16; offset > 0; offset >>= 1) {
                    localScore += __shfl_down_sync(0xffffffff, localScore, offset);
                }

                if (lane == 0) {
                    accs[r][d] += localScore;
                }
            }
        }
        __syncthreads();
    }

    for (int r = threadIdx.y; r < TILE; r += blockDim.y) {
        for (auto d = threadIdx.x; d < dims; d += blockDim.x) {
            if (row + r < M) {
                output[(row + r) * dims + d] = accs[r][d] / sums[r];
            }
        }
    }

}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d) {
    // Q: M x d
    // K: N x d
    // V: N x d
    //
    // Attention(Q, K, V): M x d
    dim3 threads(32, TILE);
    dim3 blocks(1, (M + TILE - 1) / TILE);

    attention<<<blocks, threads>>>(Q, K, V, output, M, N, d);
}
