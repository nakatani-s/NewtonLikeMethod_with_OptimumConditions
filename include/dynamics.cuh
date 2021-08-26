/*
 dynamics.cuh 
*/
#include<math.h>
#include<cuda.h>
#include "params.cuh"
#include "DataStructure.cuh"

#ifndef DYNAMICS_CUH
#define DYNAMICS_CUH

// System Dynamics
__host__ __device__ float Cart_type_Pendulum_ddx(float u, float x, float theta, float dx, float dtheta, SystemControlVariable *SCV);
__host__ __device__ float Cart_type_Pendulum_ddtheta(float u, float x,  float theta, float dx, float dtheta, SystemControlVariable *SCV);

//KKT conditions dynamics
__host__ __device__ void get_Lx_Cart_and_SinglePole(float *Lx, Tolerance *current, SystemControlVariable *SCV);
__host__ __device__ void get_LFx_Cart_and_SinglePole(float *LFx, Tolerance *current, Tolerance *later, SystemControlVariable *SCV, float t_delta);
__host__ __device__ void get_LFx_Using_M_Cart_and_SinglePole(float *LFx, Tolerance *current, Tolerance *later, SystemControlVariable *SCV, float t_delta); //2021.8.25 add
__host__ __device__ void get_dHdu_Cart_and_SinglePole(Tolerance *current, Tolerance *later, SystemControlVariable *SCV, float t_delta);

void Runge_Kutta45_for_SecondaryOderSystem(SystemControlVariable *SCV, float input, float t_delta);  //１入力系（SISO！？）
#endif