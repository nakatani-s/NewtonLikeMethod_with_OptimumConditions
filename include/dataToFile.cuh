#include <stdio.h>
#include "params.cuh"
#include "DataStructure.cuh"
// #include "dynamics.cuh"

typedef struct{
    char name[20];
    char inputfile[35];
    int dimSize;
}dataName;

void get_timeParam(int *tparam,int month, int day, int hour, int min, int step);
void write_Matrix_Information(float *data, dataName *d_name, int *timeparam);

void resd_InitSolution_Input(float *input, dataName *d_name);