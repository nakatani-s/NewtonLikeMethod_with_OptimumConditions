/*
 DataStructure.cuh
*/

#include <curand_kernel.h>
#include "params.cuh"

#ifndef DATASTRUCTURE_CUH
#define DATASTRUCTURE_CUH

typedef struct{
    float L;
    float W;
    float WHM;
    float Input[HORIZON];
    float tolerance[HORIZON];
    float dHdu[HORIZON];
    float dHdx[DIM_OF_STATES][HORIZON];
}SampleInfo;

typedef struct{
    float tensor_vector[NUM_OF_PARABOLOID_COEFFICIENT];
    float column_vector[NUM_OF_PARABOLOID_COEFFICIENT];
    float QplaneValue;
}QHP; // QHP := Quadratic Hyper Plane()


typedef struct{
    float params[DIM_OF_PARAMETERS];
    float state[DIM_OF_STATES];
    float constraints[NUM_OF_CONSTRAINTS];
    float weightMatrix[DIM_OF_WEIGHT_MATRIX];
}SystemControlVariable;

typedef struct{
    float Input[HORIZON];
}InputData;

typedef struct{
    float Input[DIM_OF_INPUT];
    float lambda[DIM_OF_STATES];
    float dHdu[DIM_OF_INPUT];
    float state[DIM_OF_STATES];
    float dstate[DIM_OF_STATES];
}Tolerance;

#endif