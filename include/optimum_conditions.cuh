/*
    optimum conditions consited with
    * cost function
    * dH/du 
    * dH/dx
    ...etc 
*/
#include <math.h>
#include <stdio.h>
#include "params.cuh"
#include "DataStructure.cuh"
#include "dynamics.cuh"

float calc_cost_Cart_and_SinglePole(float *U, SystemControlVariable SCV);
float calc_tolerance_Cart_and_SinglePole(float *U, SystemControlVariable SCV);

void calc_OC_for_Cart_and_SinglePole_hostF(float *Ans, float *U, SystemControlVariable *SCV, Tolerance *Tol);
