#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <simd/simd.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <vector>

// width & height of fluid
const int n = 256;
// Size of fluid data structures
const int arraySize = sizeof(float) * n * n;

class Fluid {
  private:
    // GPU
    MTL::Device *device;

    // Compute pipelines generated from the kernels in the shader.metal file
    MTL::ComputePipelineState *linSolvePSO;
    MTL::ComputePipelineState *projectOnePSO;
    MTL::ComputePipelineState *projectTwoPSO;
    MTL::ComputePipelineState *advectPSO;
    MTL::ComputePipelineState *setBndPSO;

    // Queue for passing commands to the device
    MTL::CommandQueue *commandQueue;

    // Simulation parameters
    float dt;
    float diff;
    float visc;

    // Buffers hold data on the device
    MTL::Buffer *bufferDt;
    MTL::Buffer *bufferDiff;
    MTL::Buffer *bufferVisc;
    MTL::Buffer *bufferS;
    MTL::Buffer *bufferDensity;
    MTL::Buffer *bufferVx;
    MTL::Buffer *bufferVy;
    MTL::Buffer *bufferVx0;
    MTL::Buffer *bufferVy0;

    // Arrays for copying changed values back to CPU after each step
    float densityResult[n * n];
    float feedbackDensity[n * n];
    float feedbackVx[n * n];
    float feedbackVy[n * n];
    float feedbackX[n * n];

    // Profiling, for calculating average step time 
    std::vector<int> executionTimes;

  public:
    Fluid(MTL::Device *pDevice, float pdt, float pDiff, float pVisc);

    void step();

    void addDensity(int x, int y, float amount);
    void addVelocity(int x, int y, float xAmount, float yAmount);

    void diffuse(int b, MTL::Buffer *x, MTL::Buffer *x0, float diff, float dt);
    void sendLinSolveComand(int b, MTL::Buffer *x, MTL::Buffer *x0, float a,
                            float c);
    void encodeLinSolveCommand(MTL::ComputeCommandEncoder *computeEncoder,
                               MTL::Buffer *xBuff, MTL::Buffer *x0Buff,
                               MTL::Buffer *bufferA, MTL::Buffer *bufferC);

    void sendProjectCommand(MTL::Buffer *velocX, MTL::Buffer *velocY,
                            MTL::Buffer *p, MTL::Buffer *div);
    void encodeProjectOneCommand(MTL::ComputeCommandEncoder *computeEncoder,
                                 MTL::Buffer *velocX, MTL::Buffer *velocY,
                                 MTL::Buffer *p, MTL::Buffer *div);
    void encodeProjectTwoCommand(MTL::ComputeCommandEncoder *computeEncoder,
                                 MTL::Buffer *velocX, MTL::Buffer *velocY,
                                 MTL::Buffer *p, MTL::Buffer *div);

    void sendAdvectCommand(int b, MTL::Buffer *d, MTL::Buffer *d0,
                           MTL::Buffer *velocX, MTL::Buffer *velocY,
                           MTL::Buffer *dt);
    void encodeAdvectCommand(MTL::ComputeCommandEncoder *computeEncoder,
                             MTL::Buffer *d, MTL::Buffer *d0,
                             MTL::Buffer *velocX, MTL::Buffer *velocY,
                             MTL::Buffer *dt);

    void sendBndCommand(int b, MTL::Buffer *x);
    void encodeBndCommand(MTL::ComputeCommandEncoder *computeEncoder,
                          MTL::Buffer *b, MTL::Buffer *x);

    float *getDensity();
};
