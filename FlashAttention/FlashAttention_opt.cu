#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 32
#define TILE_WIDTH 32
#define WARP_SIZE 32
#define NUM_WARPS_PER_BLOCK (BLOCK_SIZE * BLOCK_SIZE / WARP_SIZE)
#define MAX_THREADS_PER_BLOCK 1024
#define MIN_BLOCKS_PER_SM 2
#define SHARED_MEM_SIZE (TILE_WIDTH * (TILE_WIDTH + 1))
#define NUM_SAMPLES 512
#define FEATURE_DIMENSION 512
#define SCALE_FACTOR (1.0f / sqrtf(static_cast<float>(FEATURE_DIMENSION)))
#define NUM_ITERATIONS 2

// Utility macro for CUDA error checking
#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)
inline void checkCuda(cudaError_t result, const char* const func, const char* const file, int const line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line
            << " '" << func << "'\n";
        std::cerr << "Error: " << cudaGetErrorString(result) << std::endl;
        cudaDeviceReset();
        exit(99);
    }
}

// Matrix Initialization
void initRandomMatrix(float* __restrict__ matrix, int rows, int cols) {
    srand(0);
    #pragma omp parallel for simd aligned(matrix: 32)
    for (int i = 0; i < rows * cols; i += 4) {
        const float r1 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        const float r2 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        const float r3 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        const float r4 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        matrix[i] = r1;
        matrix[i + 1] = r2;
        matrix[i + 2] = r3;
        matrix[i + 3] = r4;
    }
}

// Matrix Printing
void printMatrix(const float* matrix, int rows, int cols, const char* name) {
    printf("%s:\n", name);
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            printf("%.6f ", matrix[y * cols + x]);
        }
        printf("\n");
    }
    printf("\n");
}

// Matrix Comparison
void compareMatrices(const float* cpuMatrix, const float* gpuMatrix, int rows, int cols, const char* name) {
    const float epsilon = 1e-5f;
    bool match = true;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            int index = y * cols + x;
            float diff = fabs(cpuMatrix[index] - gpuMatrix[index]);
            if (diff > epsilon) {
                printf("Mismatch in %s at [%d][%d]: CPU=%.6f, GPU=%.6f, Diff=%.6f\n",
                    name, y, x, cpuMatrix[index], gpuMatrix[index], diff);
                match = false;
            }
        }
    }

    if (match) {
        printf("Success: %s matrices match within tolerance %.6f\n", name, epsilon);
    }
}

// Matrix Transposition
void transposeMatrix(const float* __restrict__ inputMatrix, float* __restrict__ transposedMatrix, int rows, int cols) {
    #pragma omp parallel for simd collapse(2) aligned(inputMatrix, transposedMatrix: 32)
    for (int i = 0; i < rows; i += 4) {
        for (int j = 0; j < cols; j += 4) {
            const float v1 = inputMatrix[i * cols + j];
            const float v2 = inputMatrix[(i + 1) * cols + j];
            const float v3 = inputMatrix[(i + 2) * cols + j];
            const float v4 = inputMatrix[(i + 3) * cols + j];
            transposedMatrix[j * rows + i] = v1;
            transposedMatrix[j * rows + i + 1] = v2;
            transposedMatrix[j * rows + i + 2] = v3;
            transposedMatrix[j * rows + i + 3] = v4;
        }
    }
}

// CPU Implementation of Attention
void computeAttentionCPU(float* query, float* key, float* value, float* attentionScores, float* output) {
    float* transposedKey = (float*)malloc(FEATURE_DIMENSION * NUM_SAMPLES * sizeof(float));
    transposeMatrix(key, transposedKey, NUM_SAMPLES, FEATURE_DIMENSION);

    float scalingFactor = 1.0f / sqrtf((float)FEATURE_DIMENSION);

    // Compute attention scores
    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < NUM_SAMPLES; j++) {
            for (int k = 0; k < FEATURE_DIMENSION; k++) {
                attentionScores[i * NUM_SAMPLES + j] += query[i * FEATURE_DIMENSION + k] * transposedKey[k * NUM_SAMPLES + j];
            }
            attentionScores[i * NUM_SAMPLES + j] *= scalingFactor;
        }
    }

    // Apply softmax row-wise
    for (int row = 0; row < NUM_SAMPLES; row++) {
        float maxScore = attentionScores[row * NUM_SAMPLES];
        for (int col = 1; col < NUM_SAMPLES; col++) {
            if (attentionScores[row * NUM_SAMPLES + col] > maxScore) {
                maxScore = attentionScores[row * NUM_SAMPLES + col];
            }
        }

        float sumExp = 0.0f;
        for (int col = 0; col < NUM_SAMPLES; col++) {
            attentionScores[row * NUM_SAMPLES + col] = exp(attentionScores[row * NUM_SAMPLES + col] - maxScore);
            sumExp += attentionScores[row * NUM_SAMPLES + col];
        }

        for (int col = 0; col < NUM_SAMPLES; col++) {
            attentionScores[row * NUM_SAMPLES + col] /= sumExp;
        }
    }

    // Compute output = attentionScores * value
    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < FEATURE_DIMENSION; j++) {
            for (int k = 0; k < NUM_SAMPLES; k++) {
                output[i * FEATURE_DIMENSION + j] += attentionScores[i * NUM_SAMPLES + k] * value[k * FEATURE_DIMENSION + j];
            }
        }
    }

    free(transposedKey);
}

// Optimized CUDA kernel for computing QK^T
__global__ void computeQKAttentionKernel(const float* __restrict__ queryMatrix, 
                                        const float* __restrict__ keyTransposeMatrix, 
                                        float* __restrict__ attentionScores) {
    __shared__ float sharedQuery[SHARED_MEM_SIZE];
    __shared__ float sharedKeyTranspose[SHARED_MEM_SIZE];

    const int threadX = threadIdx.x;
    const int threadY = threadIdx.y;
    const int blockX = blockIdx.x;
    const int blockY = blockIdx.y;
    const int scoreColumnIndex = blockX * TILE_WIDTH + threadX;
    const int scoreRowIndex = blockY * TILE_WIDTH + threadY;
    
    float scoreValue = 0.0f;
    const int numPhases = (FEATURE_DIMENSION + TILE_WIDTH - 1) / TILE_WIDTH;

    // Pre-load data
    float localQuery[TILE_WIDTH];
    float localKey[TILE_WIDTH];

    #pragma unroll 4
    for (int phase = 0; phase < numPhases; phase++) {
        const int phaseOffset = phase * TILE_WIDTH;
        const bool validQuery = (phaseOffset + threadX < FEATURE_DIMENSION) && (blockY * TILE_WIDTH + threadY < NUM_SAMPLES);
        const bool validKey = (phaseOffset + threadY < FEATURE_DIMENSION) && (blockX * TILE_WIDTH + threadX < NUM_SAMPLES);

        // Load data into shared memory using vectorized loads
        if (validQuery) {
            sharedQuery[threadY * (TILE_WIDTH + 1) + threadX] = queryMatrix[(blockY * TILE_WIDTH + threadY) * FEATURE_DIMENSION + phaseOffset + threadX];
        } else {
            sharedQuery[threadY * (TILE_WIDTH + 1) + threadX] = 0.0f;
        }
        
        if (validKey) {
            sharedKeyTranspose[threadY * (TILE_WIDTH + 1) + threadX] = keyTransposeMatrix[(phaseOffset + threadY) * NUM_SAMPLES + blockX * TILE_WIDTH + threadX];
        } else {
            sharedKeyTranspose[threadY * (TILE_WIDTH + 1) + threadX] = 0.0f;
        }

        __syncthreads();

        if (scoreColumnIndex < NUM_SAMPLES && scoreRowIndex < NUM_SAMPLES) {
            #pragma unroll 4
            for (int i = 0; i < TILE_WIDTH; i++) {
                scoreValue += sharedQuery[threadY * (TILE_WIDTH + 1) + i] * 
                            sharedKeyTranspose[i * (TILE_WIDTH + 1) + threadX];
            }
        }

        __syncthreads();
    }

    if (scoreColumnIndex < NUM_SAMPLES && scoreRowIndex < NUM_SAMPLES) {
        attentionScores[scoreRowIndex * NUM_SAMPLES + scoreColumnIndex] = scoreValue * SCALE_FACTOR;
    }
}

// Optimized CUDA kernel for softmax
__global__ void computeSoftmaxKernel(const float* __restrict__ attentionScores, 
                                   float* __restrict__ softmaxScores) {
    const int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    const int warpLaneId = threadIdx.x % WARP_SIZE;

    if (rowIndex < NUM_SAMPLES) {
        // Find maximum in row using warp shuffle
        float maxScore = -1e30f;
        float localMax = -1e30f;
        
        #pragma unroll 4
        for (int colIndex = threadIdx.x; colIndex < NUM_SAMPLES; colIndex += blockDim.x) {
            localMax = fmaxf(localMax, attentionScores[rowIndex * NUM_SAMPLES + colIndex]);
        }

        // Reduce maximum within warp
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            localMax = fmaxf(localMax, __shfl_down(localMax, offset));
        }

        maxScore = localMax;

        // Compute exponentials and their sum
        float sumExp = 0.0f;
        float localSum = 0.0f;
        
        #pragma unroll 4
        for (int colIndex = threadIdx.x; colIndex < NUM_SAMPLES; colIndex += blockDim.x) {
            const float expVal = expf(attentionScores[rowIndex * NUM_SAMPLES + colIndex] - maxScore);
            softmaxScores[rowIndex * NUM_SAMPLES + colIndex] = expVal;
            localSum += expVal;
        }

        // Reduce sum within warp
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            localSum += __shfl_down(localSum, offset);
        }

        sumExp = localSum;

        // Normalize
        #pragma unroll 4
        for (int colIndex = threadIdx.x; colIndex < NUM_SAMPLES; colIndex += blockDim.x) {
            softmaxScores[rowIndex * NUM_SAMPLES + colIndex] /= sumExp;
        }
    }
}

// Optimized CUDA kernel for computing output matrix
__global__ void computeOutputKernel(const float* __restrict__ softmaxScores, 
                                  const float* __restrict__ valueMatrix, 
                                  float* __restrict__ outputMatrix) {
    __shared__ float sharedSoftmaxScores[SHARED_MEM_SIZE];
    __shared__ float sharedValueMatrix[SHARED_MEM_SIZE];

    const int threadX = threadIdx.x;
    const int threadY = threadIdx.y;
    const int blockX = blockIdx.x;
    const int blockY = blockIdx.y;
    const int outputColumnIndex = blockX * TILE_WIDTH + threadX;
    const int outputRowIndex = blockY * TILE_WIDTH + threadY;
    
    float outputValue = 0.0f;
    const int numPhases = (NUM_SAMPLES + TILE_WIDTH - 1) / TILE_WIDTH;

    // Pre-load data
    float localSoftmax[TILE_WIDTH];
    float localValue[TILE_WIDTH];

    #pragma unroll 4
    for (int phase = 0; phase < numPhases; phase++) {
        const int phaseOffset = phase * TILE_WIDTH;
        const bool validSoftmax = (phaseOffset + threadX < NUM_SAMPLES) && (blockY * TILE_WIDTH + threadY < NUM_SAMPLES);
        const bool validValue = (phaseOffset + threadY < NUM_SAMPLES) && (blockX * TILE_WIDTH + threadX < FEATURE_DIMENSION);

        // Load data into shared memory using vectorized loads
        if (validSoftmax) {
            sharedSoftmaxScores[threadY * (TILE_WIDTH + 1) + threadX] = 
                softmaxScores[(blockY * TILE_WIDTH + threadY) * NUM_SAMPLES + phaseOffset + threadX];
        } else {
            sharedSoftmaxScores[threadY * (TILE_WIDTH + 1) + threadX] = 0.0f;
        }
        
        if (validValue) {
            sharedValueMatrix[threadY * (TILE_WIDTH + 1) + threadX] = 
                valueMatrix[(phaseOffset + threadY) * FEATURE_DIMENSION + blockX * TILE_WIDTH + threadX];
        } else {
            sharedValueMatrix[threadY * (TILE_WIDTH + 1) + threadX] = 0.0f;
        }

        __syncthreads();

        if (outputColumnIndex < FEATURE_DIMENSION && outputRowIndex < NUM_SAMPLES) {
            #pragma unroll 4
            for (int i = 0; i < TILE_WIDTH; i++) {
                outputValue += sharedSoftmaxScores[threadY * (TILE_WIDTH + 1) + i] * 
                             sharedValueMatrix[i * (TILE_WIDTH + 1) + threadX];
            }
        }

        __syncthreads();
    }

    if (outputColumnIndex < FEATURE_DIMENSION && outputRowIndex < NUM_SAMPLES) {
        outputMatrix[outputRowIndex * FEATURE_DIMENSION + outputColumnIndex] = outputValue;
    }
}

// Optimized GPU implementation of Flash Attention
void computeFlashAttentionGPU(const float* __restrict__ queryMatrix, 
                             const float* __restrict__ keyTransposeMatrix, 
                             const float* __restrict__ valueMatrix, 
                             float* __restrict__ attentionScores, 
                             float* __restrict__ outputMatrix) {
    float* deviceQuery, * deviceKeyTranspose, * deviceValue;
    float* deviceAttentionScores, * deviceSoftmaxScores, * deviceOutput;

    // Allocate device memory with alignment
    cudaMalloc(&deviceQuery, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));
    cudaMalloc(&deviceKeyTranspose, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));
    cudaMalloc(&deviceValue, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));
    cudaMalloc(&deviceAttentionScores, NUM_SAMPLES * NUM_SAMPLES * sizeof(float));
    cudaMalloc(&deviceSoftmaxScores, NUM_SAMPLES * NUM_SAMPLES * sizeof(float));
    cudaMalloc(&deviceOutput, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));

    // Copy data to device
    cudaMemcpy(deviceQuery, queryMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceKeyTranspose, keyTransposeMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceValue, valueMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);

    // Optimized block and grid dimensions
    dim3 blockDimension(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDimension((NUM_SAMPLES + blockDimension.x - 1) / blockDimension.x, 
                      (NUM_SAMPLES + blockDimension.y - 1) / blockDimension.y);

    // Launch kernels with optimized parameters
    computeQKAttentionKernel<<<gridDimension, blockDimension>>>(deviceQuery, deviceKeyTranspose, deviceAttentionScores);
    cudaDeviceSynchronize();

    // Optimized dimensions for softmax
    dim3 softmaxBlockDimension(WARP_SIZE, 1);
    dim3 softmaxGridDimension(1, NUM_SAMPLES);
    computeSoftmaxKernel<<<softmaxGridDimension, softmaxBlockDimension>>>(deviceAttentionScores, deviceSoftmaxScores);
    cudaDeviceSynchronize();

    // Optimized dimensions for output matrix
    dim3 outputBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 outputGrid((FEATURE_DIMENSION + outputBlock.x - 1) / outputBlock.x,
                   (NUM_SAMPLES + outputBlock.y - 1) / outputBlock.y);
    computeOutputKernel<<<outputGrid, outputBlock>>>(deviceSoftmaxScores, deviceValue, deviceOutput);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(attentionScores, deviceSoftmaxScores, NUM_SAMPLES * NUM_SAMPLES * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(outputMatrix, deviceOutput, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(deviceQuery);
    cudaFree(deviceKeyTranspose);
    cudaFree(deviceValue);
    cudaFree(deviceAttentionScores);
    cudaFree(deviceSoftmaxScores);
    cudaFree(deviceOutput);
}

int main() {
    float* queryMatrix = new float[NUM_SAMPLES * FEATURE_DIMENSION];
    float* keyMatrix = new float[NUM_SAMPLES * FEATURE_DIMENSION];
    float* valueMatrix = new float[NUM_SAMPLES * FEATURE_DIMENSION];
    float* outputGPUShared = new float[NUM_SAMPLES * FEATURE_DIMENSION]();
    float* attentionScoresShared = new float[NUM_SAMPLES * NUM_SAMPLES]();
    float* transposedKeyMatrix = new float[FEATURE_DIMENSION * NUM_SAMPLES];

    initRandomMatrix(queryMatrix, NUM_SAMPLES, FEATURE_DIMENSION);
    initRandomMatrix(keyMatrix, NUM_SAMPLES, FEATURE_DIMENSION);
    initRandomMatrix(valueMatrix, NUM_SAMPLES, FEATURE_DIMENSION);
    transposeMatrix(keyMatrix, transposedKeyMatrix, NUM_SAMPLES, FEATURE_DIMENSION);

    // Measure total execution time
    cudaEvent_t start, stop;
    float total_milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    computeFlashAttentionGPU(queryMatrix, transposedKeyMatrix, valueMatrix, attentionScoresShared, outputGPUShared);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&total_milliseconds, start, stop);
    
    printf("Total GPU Execution Time: %.3f ms\n", total_milliseconds);

    delete[] queryMatrix;
    delete[] keyMatrix;
    delete[] valueMatrix;
    delete[] outputGPUShared;
    delete[] attentionScoresShared;
    delete[] transposedKeyMatrix;

    return 0;
}