#include <metal_stdlib>
using namespace metal;

constant int n = 256;

int IX(int x, int y) { return x + y * n; }

kernel void linSolveKernel(device float *x, device const float *x0,
                           device const float *a, device float *c,
                           uint2 index [[thread_position_in_grid]]) {
    int i = index.x + 1;
    int j = index.y + 1;
    x[IX(i, j)] = (x0[IX(i, j)] +
                   (*a) * (x[IX(i + 1, j)] + x[IX(i - 1, j)] + x[IX(i, j + 1)] +
                           x[IX(i, j - 1)] + x[IX(i, j)] + x[IX(i, j)])) *
                  (1.0 / (*c));
}

kernel void projectOneKernel(device float *velocX, device float *velocY,
                             device float *p, device float *div,
                             uint2 index [[thread_position_in_grid]]) {
    int i = index.x + 1;
    int j = index.y + 1;
    div[IX(i, j)] = (-0.5 * (velocX[IX(i + 1, j)] - velocX[IX(i - 1, j)] +
                             velocY[IX(i, j + 1)] - velocY[IX(i, j - 1)])) /
                    n;
    p[IX(i, j)] = 0;
}

kernel void projectTwoKernel(device float *velocX, device float *velocY,
                             device float *p, device float *div,
                             uint2 index [[thread_position_in_grid]]) {
    int i = index.x + 1;
    int j = index.y + 1;
    velocX[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) * n;
    velocY[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) * n;
}

kernel void advectKernel(device float *d, device float *d0,
                         device float *velocX, device float *velocY,
                         device float *dt,
                         uint2 index [[thread_position_in_grid]]) {
    int i = index.x + 1;
    int j = index.y + 1;

    float x = i - (*dt) * velocX[IX(i, j)];
    float y = j - (*dt) * velocY[IX(i, j)];
    
    x = fmin(fmax(x, 0.5f), n + 0.5f);
    int i0 = (int)x;
    int i1 = i0 + 1;
    y = fmin(fmax(y, 0.5f), n + 0.5f);
    int j0 = (int)y;
    int j1 = j0 + 1;

    float s1 = x - i0;
    float s0 = 1 - s1;
    float t1 = y - j0;
    float t0 = 1 - t1;

    d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                  s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
}

kernel void setBndKernel(device int *b, device float *x,
                         uint2 index [[thread_position_in_grid]]) {
    int i = index.x + 1;

    x[IX(i, 0)] = (*b) == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
    x[IX(i, n - 1)] = (*b) == 2 ? -x[IX(i, n - 2)] : x[IX(i, n - 2)];

    x[IX(0, i)] = (*b) == 1 ? -x[IX(1, i)] : x[IX(1, i)];
    x[IX(n - 1, i)] = (*b) == 1 ? -x[IX(n - 2, i)] : x[IX(n - 2, i)];
}
