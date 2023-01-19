#include "Fluid.hpp"

// Utility function for flattening 2D index into 1D
int IX(int x, int y) { return x + y * n; }

// Returns raw string of shader file
std::string loadShader() {
    std::string file_path = __FILE__;
    std::regex target("Fluid.cpp");
    file_path = std::regex_replace(file_path, target, "");
    file_path += "shader.metal";

    std::string shaderStr;
    std::fstream shaderFile(file_path, std::ios_base::in);

    if (shaderFile) {
        shaderFile.seekg(0, std::ios::end);
        shaderStr.resize(shaderFile.tellg());
        shaderFile.seekg(0, std::ios::beg);
        shaderFile.read(&shaderStr[0], (long)shaderStr.size());
    }
    shaderFile.close();

    return shaderStr;
}

Fluid::Fluid(MTL::Device *pDevice, float pdt, float pDiff, float pVisc) {
    device = pDevice;

    auto UTF8StringEncoding = NS::StringEncoding::UTF8StringEncoding;

    NS::Error *error = nullptr;

    // Load shader file
    MTL::Library *defaultLibrary = device->newLibrary(
        NS::String::string(loadShader().c_str(), UTF8StringEncoding), nullptr,
        &error);

    // Create new compute pipeline state objects from shader functions
    MTL::Function *linSolveFunction = defaultLibrary->newFunction(
        NS::String::string("linSolveKernel", UTF8StringEncoding));
    linSolvePSO = device->newComputePipelineState(linSolveFunction, &error);

    MTL::Function *projectFunctionOne = defaultLibrary->newFunction(
        NS::String::string("projectOneKernel", UTF8StringEncoding));
    projectOnePSO = device->newComputePipelineState(projectFunctionOne, &error);

    MTL::Function *projectFunctionTwo = defaultLibrary->newFunction(
        NS::String::string("projectTwoKernel", UTF8StringEncoding));
    projectTwoPSO = device->newComputePipelineState(projectFunctionTwo, &error);

    MTL::Function *advectFunction = defaultLibrary->newFunction(
        NS::String::string("advectKernel", UTF8StringEncoding));
    advectPSO = device->newComputePipelineState(advectFunction, &error);

    MTL::Function *setBndFunction = defaultLibrary->newFunction(
        NS::String::string("setBndKernel", UTF8StringEncoding));
    setBndPSO = device->newComputePipelineState(setBndFunction, &error);

    commandQueue = device->newCommandQueue();

    dt = pdt;
    diff = pDiff;
    visc = pVisc;

    // Allocate buffers
    bufferDt = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
    bufferDiff =
        device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
    bufferVisc =
        device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
    bufferS = device->newBuffer(arraySize, MTL::ResourceStorageModeShared);
    bufferDensity =
        device->newBuffer(arraySize, MTL::ResourceStorageModeShared);
    bufferVx = device->newBuffer(arraySize, MTL::ResourceStorageModeShared);
    bufferVy = device->newBuffer(arraySize, MTL::ResourceStorageModeShared);
    bufferVx0 = device->newBuffer(arraySize, MTL::ResourceStorageModeShared);
    bufferVy0 = device->newBuffer(arraySize, MTL::ResourceStorageModeShared);

    // Copy simulation parameters onto device
    memcpy(bufferDt->contents(), &pdt, sizeof(float));
    memcpy(bufferDiff->contents(), &pDiff, sizeof(float));
    memcpy(bufferVisc->contents(), &pVisc, sizeof(float));
}

void Fluid::step() {
    auto start = std::chrono::high_resolution_clock::now();

    diffuse(1, bufferVx0, bufferVx, visc, dt);

    diffuse(2, bufferVy0, bufferVy, visc, dt);

    sendProjectCommand(bufferVx0, bufferVy0, bufferVx, bufferVy);

    sendAdvectCommand(1, bufferVx, bufferVx0, bufferVx0, bufferVy0, bufferDt);
    sendAdvectCommand(2, bufferVy, bufferVy0, bufferVx0, bufferVy0, bufferDt);

    sendProjectCommand(bufferVx, bufferVy, bufferVx0, bufferVy0);
    diffuse(0, bufferS, bufferDensity, diff, dt);
    sendAdvectCommand(0, bufferDensity, bufferS, bufferVx, bufferVy, bufferDt);

    // Profiling
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << duration.count() << std::endl;
}

// Copy data from device to cpu, alter on cpu, then copy back
void Fluid::addDensity(int x, int y, float amount) {
    memcpy(feedbackDensity, bufferDensity->contents(), arraySize);
    feedbackDensity[x + y * n] += amount;
    memcpy(bufferDensity->contents(), feedbackDensity, arraySize);
}

void Fluid::addVelocity(int x, int y, float xAmount, float yAmount) {
    memcpy(feedbackVx, bufferVx->contents(), arraySize);
    feedbackVx[x + y * n] += xAmount;
    memcpy(bufferVx->contents(), feedbackVx, arraySize);

    memcpy(feedbackVy, bufferVy->contents(), arraySize);
    feedbackVy[x + y * n] += yAmount;
    memcpy(bufferVy->contents(), feedbackVy, arraySize);
}

void Fluid::diffuse(int b, MTL::Buffer *x, MTL::Buffer *x0, float diff,
                    float dt) {
    auto a = dt * diff * (n - 2) * (n - 2);
    sendLinSolveComand(b, x, x0, a, 1 + 6 * a);
}

void Fluid::sendLinSolveComand(int b, MTL::Buffer *x, MTL::Buffer *x0, float a,
                               float c) {
    MTL::Buffer *bufferA =
        device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer *bufferC =
        device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    memcpy(bufferA->contents(), &a, sizeof(float));
    memcpy(bufferC->contents(), &c, sizeof(float));

    for (int i = 0; i < 16; i++) {
        // Command buffer holds commands
        MTL::CommandBuffer *commandBuffer = commandQueue->commandBuffer();

        // Start a compute pass
        MTL::ComputeCommandEncoder *computeEncoder =
            commandBuffer->computeCommandEncoder();

        encodeLinSolveCommand(computeEncoder, x, x0, bufferA, bufferC);

        computeEncoder->endEncoding();
        // Execute the command
        commandBuffer->commit();

        sendBndCommand(b, x);
    }
}

void Fluid::encodeLinSolveCommand(MTL::ComputeCommandEncoder *computeEncoder,
                                  MTL::Buffer *xBuff, MTL::Buffer *x0Buff,
                                  MTL::Buffer *bufferA, MTL::Buffer *bufferC) {
    // Encode the pipeline state objext and its parameters
    computeEncoder->setComputePipelineState(linSolvePSO);

    computeEncoder->setBuffer(xBuff, 0, 0);
    computeEncoder->setBuffer(x0Buff, 0, 1);
    computeEncoder->setBuffer(bufferA, 0, 2);
    computeEncoder->setBuffer(bufferC, 0, 3);

    MTL::Size gridSize{n - 2, n - 2, 1};
    // Calculate a thread group size
    MTL::Size threadGroupSize{linSolvePSO->threadExecutionWidth(),
                              linSolvePSO->maxTotalThreadsPerThreadgroup() /
                                  linSolvePSO->threadExecutionWidth(),
                              1};

    // Encode the compute command
    computeEncoder->dispatchThreads(gridSize, threadGroupSize);
}

void Fluid::sendProjectCommand(MTL::Buffer *velocX, MTL::Buffer *velocY,
                               MTL::Buffer *p, MTL::Buffer *div) {
    MTL::CommandBuffer *commandBuffer = commandQueue->commandBuffer();

    MTL::ComputeCommandEncoder *computeEncoder =
        commandBuffer->computeCommandEncoder();

    encodeProjectOneCommand(computeEncoder, velocX, velocY, p, div);

    computeEncoder->endEncoding();
    commandBuffer->commit();

    sendBndCommand(0, div);
    sendBndCommand(0, p);
    sendLinSolveComand(0, p, div, 1, 6);

    MTL::CommandBuffer *commandBufferTwo = commandQueue->commandBuffer();

    MTL::ComputeCommandEncoder *computeEncoderTwo =
        commandBufferTwo->computeCommandEncoder();

    encodeProjectTwoCommand(computeEncoderTwo, velocX, velocY, p, div);

    computeEncoderTwo->endEncoding();
    commandBufferTwo->commit();

    sendBndCommand(1, velocX);
    sendBndCommand(2, velocY);
}

void Fluid::encodeProjectOneCommand(MTL::ComputeCommandEncoder *computeEncoder,
                                    MTL::Buffer *velocX, MTL::Buffer *velocY,
                                    MTL::Buffer *p, MTL::Buffer *div) {
    computeEncoder->setComputePipelineState(projectOnePSO);

    computeEncoder->setBuffer(velocX, 0, 0);
    computeEncoder->setBuffer(velocY, 0, 1);
    computeEncoder->setBuffer(p, 0, 2);
    computeEncoder->setBuffer(div, 0, 3);

    MTL::Size gridSize{n - 2, n - 2, 1};
    MTL::Size threadGroupSize{linSolvePSO->threadExecutionWidth(),
                              linSolvePSO->maxTotalThreadsPerThreadgroup() /
                                  linSolvePSO->threadExecutionWidth(),
                              1};

    computeEncoder->dispatchThreads(gridSize, threadGroupSize);
}

void Fluid::encodeProjectTwoCommand(MTL::ComputeCommandEncoder *computeEncoder,
                                    MTL::Buffer *velocX, MTL::Buffer *velocY,
                                    MTL::Buffer *p, MTL::Buffer *div) {
    computeEncoder->setComputePipelineState(projectTwoPSO);

    computeEncoder->setBuffer(velocX, 0, 0);
    computeEncoder->setBuffer(velocY, 0, 1);
    computeEncoder->setBuffer(p, 0, 2);
    computeEncoder->setBuffer(div, 0, 3);

    MTL::Size gridSize{n - 2, n - 2, 1};
    MTL::Size threadGroupSize{linSolvePSO->threadExecutionWidth(),
                              linSolvePSO->maxTotalThreadsPerThreadgroup() /
                                  linSolvePSO->threadExecutionWidth(),
                              1};

    computeEncoder->dispatchThreads(gridSize, threadGroupSize);
}

void Fluid::sendAdvectCommand(int b, MTL::Buffer *d, MTL::Buffer *d0,
                              MTL::Buffer *velocX, MTL::Buffer *velocY,
                              MTL::Buffer *dt) {
    MTL::CommandBuffer *commandBuffer = commandQueue->commandBuffer();

    MTL::ComputeCommandEncoder *computeEncoder =
        commandBuffer->computeCommandEncoder();

    encodeAdvectCommand(computeEncoder, d, d0, velocX, velocY, dt);

    computeEncoder->endEncoding();
    commandBuffer->commit();

    sendBndCommand(b, d);
}

void Fluid::encodeAdvectCommand(MTL::ComputeCommandEncoder *computeEncoder,
                                MTL::Buffer *d, MTL::Buffer *d0,
                                MTL::Buffer *velocX, MTL::Buffer *velocY,
                                MTL::Buffer *dt) {
    computeEncoder->setComputePipelineState(advectPSO);

    computeEncoder->setBuffer(d, 0, 0);
    computeEncoder->setBuffer(d0, 0, 1);
    computeEncoder->setBuffer(velocX, 0, 2);
    computeEncoder->setBuffer(velocY, 0, 3);
    computeEncoder->setBuffer(dt, 0, 4);

    MTL::Size gridSize{n - 2, n - 2, 1};
    MTL::Size threadGroupSize{linSolvePSO->threadExecutionWidth(),
                              linSolvePSO->maxTotalThreadsPerThreadgroup() /
                                  linSolvePSO->threadExecutionWidth(),
                              1};

    computeEncoder->dispatchThreads(gridSize, threadGroupSize);
}

void Fluid::sendBndCommand(int b, MTL::Buffer *x) {
    MTL::Buffer *bufferB =
        device->newBuffer(sizeof(int), MTL::ResourceStorageModeShared);
    memcpy(bufferB->contents(), &b, sizeof(int));

    MTL::CommandBuffer *commandBuffer = commandQueue->commandBuffer();

    MTL::ComputeCommandEncoder *computeEncoder =
        commandBuffer->computeCommandEncoder();

    encodeBndCommand(computeEncoder, bufferB, x);

    computeEncoder->endEncoding();
    commandBuffer->commit();
    // setBnd is the last stage of each step, so we must wait for the GPU to finish processing before continuing
    commandBuffer->waitUntilCompleted();

    memcpy(feedbackX, x->contents(), arraySize);
    feedbackX[0] = 0.5 * (feedbackX[1] + feedbackX[n]);
    feedbackX[(n - 1) * n] = 0.5 * (feedbackX[1 + (n - 1) * n] + feedbackX[(n - 2) * n]);
    feedbackX[n - 1] = 0.5 * (feedbackX[n - 2] + feedbackX[(n - 1) + n]);
    feedbackX[(n - 1) + (n - 1) * n] =
        0.5 * (feedbackX[(n - 2) + (n - 1) * n] + feedbackX[(n - 1) + (n - 2) * n]);
    memcpy(x->contents(), feedbackX, arraySize);
}

void Fluid::encodeBndCommand(MTL::ComputeCommandEncoder *computeEncoder,
                             MTL::Buffer *b, MTL::Buffer *x) {
    computeEncoder->setComputePipelineState(setBndPSO);

    computeEncoder->setBuffer(b, 0, 0);
    computeEncoder->setBuffer(x, 0, 1);

    MTL::Size gridSize{n - 2, 1, 1};
    MTL::Size threadGroupSize{linSolvePSO->maxTotalThreadsPerThreadgroup(), 1,
                              1};

    computeEncoder->dispatchThreads(gridSize, threadGroupSize);
}

float *Fluid::getDensity() {
    memcpy(densityResult, bufferDensity->contents(), arraySize);
    return densityResult;
}
