/*
 Matrix.cuh
*/
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include "params.cuh"
#include "DataStructure.cuh"
#include "dynamics.cuh"


void printMatrix(int m, int n, float*A, int lda, const char* name);


__global__ void MatrixSetUpLargeIdentityMatrix(float *IdMat, int Ydimention); //1024×1024以上の行列に対応　Ydimentionは行(又は列)数
__global__ void MatrixSetUpSmallIdentityMatrix(float *IdMat); //1024×1024以下の行列で推奨

__global__ void MatrixMultiplyOperation(float *RetMat, float multiplyValue, float *OriginMat);