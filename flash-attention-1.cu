#include <cuda_runtime.h>

constexpr int TILE = 4;
constexpr int D = 128;

__global__ void attention(const float* Q, const float* K, const float* V, float* output, int M, int N,
                          int dims) {
    int row = blockIdx.y * TILE + threadIdx.y;

    __shared__ float Qs[TILE][D];
    __shared__ float Ks[TILE][D];
    __shared__ float Vs[TILE][D];

    float invSqrtDims = rsqrtf(dims);

    for (int r = 0; r < TILE; r++) {
        for (auto d = threadIdx.x; d < dims; d += blockDim.x) {
            if (row + r < M) {
                Qs[r][d] = Q[(row + r) * dims + d];
            } else {
                Qs[r][d] = 0.0;
            }
        }
    }

    int numTiles = (N + TILE - 1) / TILE;

    __shared__ float accs[TILE][D];
    __shared__ float sums[TILE];
    __shared__ float maxs[TILE];
    for (int r = 0; r < TILE; r++) {
        for (auto d = threadIdx.x; d < dims; d += blockDim.x) {
            accs[r][d] = 0.0;
        }
    }

    for (int r = threadIdx.x; r < TILE; r += blockDim.x) {
        maxs[r] = -INFINITY;
        sums[r] = 0.0;
    }
    __syncthreads();

    // Load K/V tiles
    for (int t = 0; t < numTiles; t++) {
        for (int r = 0; r < TILE; r++) {
            for (int d = threadIdx.x; d < dims; d += blockDim.x) {
                if (TILE * t + r < N) {
                    Ks[r][d] = K[(TILE * t + r) * dims + d];
                    Vs[r][d] = V[(TILE * t + r) * dims + d];
                } else {
                    Ks[r][d] = 0.0;
                    Vs[r][d] = 0.0;
                }
            }
        }

        // Calculate QKT
        float scores[TILE][TILE] = {0.0};
        for (int r = threadIdx.x; r < TILE; r += blockDim.x) {
            auto maxF = maxs[r];
            for (int c = 0; c < TILE; c++) {
                if (t * TILE + c >= N) {
                    continue;
                }
                for (int d = 0; d < dims; d++) {
                    scores[r][c] += Qs[r][d] * Ks[c][d];
                }
                scores[r][c] *= invSqrtDims;
                maxF = max(maxF, scores[r][c]);
            }

            auto total = 0.0;
            for (int c = 0; c < TILE; c++) {
                if (t * TILE + c >= N) {
                    scores[r][c] = 0.0f;
                    continue;
                }
                auto score = expf(scores[r][c] - maxF);
                scores[r][c] = score;
                total += score;
            }
            auto scale = expf(maxs[r] - maxF);
            sums[r] *= scale;
            sums[r] += total;
            maxs[r] = maxF;
            for (int d = 0; d < dims; d++) {
                accs[r][d] *= scale;
            }
        }

        // Calculate (QKT) @ V
        for (int r = threadIdx.x; r < TILE; r += blockDim.x) {
            for (int c = 0; c < TILE; c++) {
                for (int d = 0; d < dims; d++) {
                    accs[r][d] += scores[r][c] * Vs[c][d];
                }
            }
        }
    }
    __syncthreads();

    for (int r = 0; r < TILE; r++) {
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
    dim3 threads(32, 1);
    dim3 blocks(1, (M + TILE - 1) / TILE);

    attention<<<blocks, threads>>>(Q, K, V, output, M, N, d);
}
