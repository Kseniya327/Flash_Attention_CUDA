#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Constants for CUDA kernel configuration
#define BLOCK_SIZE 32
#define MEM_WIDTH 32
#define TILE_WIDTH 32

// Model configuration parameters
#define NUM_SAMPLES 512
#define FEATURE_DIMENSION 512

// CUDA error handling utility
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

// Initialize matrix with random values
void initializeMatrixWithRandomValues(float* matrix, int rows, int cols) {
    srand(0);
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

// Display matrix contents
void displayMatrix(const float* matrix, int rows, int cols, const char* name) {
    printf("%s:\n", name);
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            printf("%.6f ", matrix[y * cols + x]);
        }
        printf("\n");
    }
    printf("\n");
}

// Compare matrices and report differences
void compareMatrixResults(const float* cpuMatrix, const float* gpuMatrix, int rows, int cols, const char* name) {
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

// Transpose matrix operation
void transposeMatrixOperation(const float* inputMatrix, float* transposedMatrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transposedMatrix[j * rows + i] = inputMatrix[i * cols + j];
        }
    }
}

// CPU implementation of attention mechanism
void computeAttentionOnCPU(float* queryMatrix, float* keyMatrix, float* valueMatrix, float* attentionScores, float* outputMatrix) {
    float* transposedKeyMatrix = (float*)malloc(FEATURE_DIMENSION * NUM_SAMPLES * sizeof(float));
    transposeMatrixOperation(keyMatrix, transposedKeyMatrix, NUM_SAMPLES, FEATURE_DIMENSION);

    float scalingFactor = 1.0f / sqrtf((float)FEATURE_DIMENSION);

    // Calculate attention scores
    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < NUM_SAMPLES; j++) {
            for (int k = 0; k < FEATURE_DIMENSION; k++) {
                attentionScores[i * NUM_SAMPLES + j] += queryMatrix[i * FEATURE_DIMENSION + k] * transposedKeyMatrix[k * NUM_SAMPLES + j];
            }
            attentionScores[i * NUM_SAMPLES + j] *= scalingFactor;
        }
    }

    // Apply softmax normalization
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

    // Compute final attention output
    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < FEATURE_DIMENSION; j++) {
            for (int k = 0; k < NUM_SAMPLES; k++) {
                outputMatrix[i * FEATURE_DIMENSION + j] += attentionScores[i * NUM_SAMPLES + k] * valueMatrix[k * FEATURE_DIMENSION + j];
            }
        }
    }

    free(transposedKeyMatrix);
}

// CUDA kernel for Flash Attention implementation
__global__ void flashAttentionKernel(
    float* queryMatrix,    // [NUM_SAMPLES, FEATURE_DIMENSION]
    float* keyMatrix,      // [NUM_SAMPLES, FEATURE_DIMENSION]
    float* valueMatrix,    // [NUM_SAMPLES, FEATURE_DIMENSION]
    float* outputMatrix    // [NUM_SAMPLES, FEATURE_DIMENSION]
) {
    __shared__ float sharedQueryTile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedKeyTile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedValueTile[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float localSum = 0.0f;
    float localMax = -INFINITY;
    float localSoftmaxScores[BLOCK_SIZE];
    
    // Process data in tiles using shared memory
    for (int tile = 0; tile < (NUM_SAMPLES + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        // Load query and key tiles into shared memory
        if (row < NUM_SAMPLES && tile * BLOCK_SIZE + threadIdx.x < NUM_SAMPLES) {
            sharedQueryTile[threadIdx.y][threadIdx.x] = queryMatrix[row * FEATURE_DIMENSION + tile * BLOCK_SIZE + threadIdx.x];
            sharedKeyTile[threadIdx.y][threadIdx.x] = keyMatrix[(tile * BLOCK_SIZE + threadIdx.y) * FEATURE_DIMENSION + col];
        }
        __syncthreads();
        
        // Calculate attention scores for current tile
        if (row < NUM_SAMPLES && col < NUM_SAMPLES) {
            float attentionScore = 0.0f;
            for (int i = 0; i < BLOCK_SIZE; i++) {
                attentionScore += sharedQueryTile[threadIdx.y][i] * sharedKeyTile[i][threadIdx.x];
            }
            attentionScore /= sqrtf(FEATURE_DIMENSION);
            localMax = fmaxf(localMax, attentionScore);
            localSoftmaxScores[tile] = attentionScore;
        }
        __syncthreads();
    }
    
    // Apply softmax normalization
    if (row < NUM_SAMPLES && col < NUM_SAMPLES) {
        float sumExp = 0.0f;
        for (int i = 0; i < (NUM_SAMPLES + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
            localSoftmaxScores[i] = expf(localSoftmaxScores[i] - localMax);
            sumExp += localSoftmaxScores[i];
        }
        for (int i = 0; i < (NUM_SAMPLES + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
            localSoftmaxScores[i] /= sumExp;
        }
    }
    
    // Calculate final attention output
    if (row < NUM_SAMPLES && col < FEATURE_DIMENSION) {
        float result = 0.0f;
        for (int tile = 0; tile < (NUM_SAMPLES + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
            // Load value tile into shared memory
            if (tile * BLOCK_SIZE + threadIdx.x < NUM_SAMPLES) {
                sharedValueTile[threadIdx.y][threadIdx.x] = valueMatrix[(tile * BLOCK_SIZE + threadIdx.y) * FEATURE_DIMENSION + col];
            }
            __syncthreads();
            
            // Compute output for current tile
            for (int i = 0; i < BLOCK_SIZE; i++) {
                result += localSoftmaxScores[tile] * sharedValueTile[i][threadIdx.x];
            }
            __syncthreads();
        }
        outputMatrix[row * FEATURE_DIMENSION + col] = result;
    }
}

// GPU implementation of Flash Attention
void computeFlashAttentionOnGPU(float* queryMatrix, float* keyMatrix, float* valueMatrix, float* outputMatrix) {
    float* deviceQueryMatrix, * deviceKeyMatrix, * deviceValueMatrix, * deviceOutputMatrix;

    // Setup CUDA timing events
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float executionTime = 0;

    // Start timing
    cudaEventRecord(startEvent);

    // Allocate GPU memory
    cudaMalloc(&deviceQueryMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));
    cudaMalloc(&deviceKeyMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));
    cudaMalloc(&deviceValueMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));
    cudaMalloc(&deviceOutputMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));

    // Transfer data to GPU
    cudaMemcpy(deviceQueryMatrix, queryMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceKeyMatrix, keyMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceValueMatrix, valueMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((FEATURE_DIMENSION + blockDim.x - 1) / blockDim.x, (NUM_SAMPLES + blockDim.y - 1) / blockDim.y);

    // Launch Flash Attention kernel
    flashAttentionKernel<<<gridDim, blockDim>>>(deviceQueryMatrix, deviceKeyMatrix, deviceValueMatrix, deviceOutputMatrix);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(outputMatrix, deviceOutputMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate and display execution time
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&executionTime, startEvent, stopEvent);
    printf("Flash Attention GPU execution time: %.3f ms\n", executionTime);

    // Free GPU memory
    cudaFree(deviceQueryMatrix);
    cudaFree(deviceKeyMatrix);
    cudaFree(deviceValueMatrix);
    cudaFree(deviceOutputMatrix);

    // Cleanup CUDA events
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

int main() {
    // Allocate host memory
    float* hostQuery = (float*)malloc(NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));
    float* hostKey = (float*)malloc(NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));
    float* hostValue = (float*)malloc(NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));
    float* hostOutput = (float*)malloc(NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));

    // Initialize input matrices
    initializeMatrixWithRandomValues(hostQuery, NUM_SAMPLES, FEATURE_DIMENSION);
    initializeMatrixWithRandomValues(hostKey, NUM_SAMPLES, FEATURE_DIMENSION);
    initializeMatrixWithRandomValues(hostValue, NUM_SAMPLES, FEATURE_DIMENSION);

    // Initialize output matrix
    memset(hostOutput, 0, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));

    // Execute Flash Attention on GPU
    printf("Executing Flash Attention on GPU...\n");
    computeFlashAttentionOnGPU(hostQuery, hostKey, hostValue, hostOutput);

    // Free host memory
    free(hostQuery);
    free(hostKey);
    free(hostValue);
    free(hostOutput);

    return 0;
}
