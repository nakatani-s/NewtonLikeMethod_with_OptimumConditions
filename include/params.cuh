/*
params.cuh

*/
#include <math.h>
#ifndef PARAMS_CUH
#define PARAMS_CUH

#define InputSaturation
#define WRITE_MATRIX_INFORMATION
#define USING_WEIGHTED_LEAST_SQUARES

#define SIM_TIME 1000
#define ITERATIONS_MAX 1000
#define ITERATIONS 4
#define HORIZON 35

#define DIM_OF_PARAMETERS 7
#define DIM_OF_STATES 4
#define NUM_OF_CONSTRAINTS 4
#define DIM_OF_WEIGHT_MATRIX 5
#define DIM_OF_INPUT 1

#define NUM_OF_SAMPLES 10000
#define NUM_OF_ELITES 200
#define THREAD_PER_BLOCKS 10

const float predictionInterval = 0.7f;
const float interval = 0.01f; // control cycle for plant
const int NUM_OF_PARABOLOID_COEFFICIENT = 666;
const int MAX_DIVISOR = 666;  //Require divisor of "NUM_OF_PARABOLOID_COEFFICIENT" less than 1024 
const int addTermForLSM = 1034;
const float neighborVar = 0.5f;
const float variance = 2.0f; // variance used for seaching initial solution by MCMPC with Geometric Cooling
const float Rho = 1e-6; // inverse constant values for Barrier term
const float sRho = 1e-4;
const float mic = 0.1f;
const float hdelta = 0.1f;
const float zeta = 1e-6;

// Parameters for cuBlas
const float alpha = 1.0f;
const float beta = 0.0f;

#endif