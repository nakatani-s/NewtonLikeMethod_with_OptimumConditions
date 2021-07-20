/*
    NewtonLikeMethod.cuh cosist following part of Algorithm
    1. get tensor vector for generate regular matrix which is used in least squere mean
    2.
*/ 
#include<cuda.h>
#include<curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include "params.cuh"
#include "DataStructure.cuh"
#include "dynamics.cuh"


/* ------------ global functions are defined below -------------*/
__global__ void NewtonLikeMethodGetTensorVector(QHP *Out, SampleInfo *In, int *indices);
__global__ void NewtonLikeMethodGenNormalizationMatrix(float *Mat, QHP *elements, int SAMPLE_SIZE, int Ydimention);
__global__ void NewtonLikeMethodGenNormalizationVector(float *Vec, QHP *elements, int SAMPLE_SIZE);
__global__ void NewtonLikeMethodGenNormalEquation(float *Mat, float *Vec, QHP *elements, int SAMPLE_SIZE, int Ydimention);

// 最小二乗法の結果からヘシアンだけ取り出すための関数群
__global__ void NewtonLikeMethodGetHessianElements(float *HessElement, float *ansVec);
__global__ void NewtonLikeMethodGetHessianOriginal(float *Hessian, float *HessianElements);

__global__ void NewtonLikeMethodGetLowerTriangle(float *LowerTriangle, float *UpperTriangle);
__global__ void NewtonLikeMethodGetFullHessianLtoU(float *FullHessian, float *LowerTriangle);
__global__ void NewtonLikeMethodGetFullHessianUtoL(float *FullHessian, float *UpperTriangle);


// 最小二乗法の結果から勾配相当のベクトルを取り出すための関数
__global__ void NewtonLikeMethodGetGradient(float *Gradient, float *elements, int index);