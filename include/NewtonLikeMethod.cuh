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


void NewtonLikeMethodInputSaturation(float *In, float Umax, float Umin);
void NewtonLikeMethodGetIterResult(SampleInfo *RetInfo, float costValue, float *InputSeq);

__global__ void NewtonLikeMethodGetTensorVectorNoIndex(QHP *Out, SampleInfo *Info);

/* ------------ global functions are defined below -------------*/
__global__ void NewtonLikeMethodGetTensorVector(QHP *Out, SampleInfo *In, int *indices);
__global__ void NewtonLikeMethodGenNormalizationMatrix(float *Mat, QHP *elements, int SAMPLE_SIZE, int Ydimention);
__global__ void NewtonLikeMethodGenNormalizationVector(float *Vec, QHP *elements, int SAMPLE_SIZE);
__global__ void NewtonLikeMethodGenNormalEquation(float *Mat, float *Vec, QHP *elements, int SAMPLE_SIZE, int Ydimention);

__global__ void NewtonLikeMethodGetRegularMatrix(float *Mat, QHP *element, int Sample_size);
__global__ void NewtonLikeMethodGetRegularVector(float *Vec, QHP *element, int Sample_size);

// 最小二乗法の結果からヘシアンだけ取り出すための関数群
__global__ void NewtonLikeMethodGetHessianElements(float *HessElement, float *ansVec);
__global__ void NewtonLikeMethodGetHessianOriginal(float *Hessian, float *HessianElements);

__global__ void NewtonLikeMethodGetLowerTriangle(float *LowerTriangle, float *UpperTriangle);
__global__ void NewtonLikeMethodGetFullHessianLtoU(float *FullHessian, float *LowerTriangle);
__global__ void NewtonLikeMethodGetFullHessianUtoL(float *FullHessian, float *UpperTriangle);


// 最小二乗法の結果から勾配相当のベクトルを取り出すための関数
__global__ void NewtonLikeMethodGetGradient(float *Gradient, float *elements, int index);

// ベクトルのコピー，あとの処理を円滑に行うために
__global__ void NewtonLikeMethodCopyVector(float *Out, float *In);