/*
    Matrix.cu
*/
#include "../include/Matrix.cuh"

void printMatrix(int m, int n, float*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            float Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
            //printf("%s[%d] = %f\n", name, row + col*lda, Areg);
        }
    }
}

__global__ void MatrixSetUpLargeIdentityMatrix(float *IdMat, int Ydimention)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int id = iy * Ydimention + ix;

    if(ix == iy)
    {
        IdMat[id] = 1.0f;
    }else{
        IdMat[id] = 0.0f;
    }
    // __syncthreads();
}

__global__ void MatrixSetUpSmallIdentityMatrix(float *IdMat)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(threadIdx.x == blockIdx.x)
    {
        IdMat[id] = 1.0f;
    }else{
        IdMat[id] = 0.0f;
    }
    __syncthreads();
}

__global__ void MatrixMultiplyOperation(float *RetMat, float multiplyValue, float *OriginMat)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    RetMat[id] = multiplyValue * OriginMat[id];
    __syncthreads();
}