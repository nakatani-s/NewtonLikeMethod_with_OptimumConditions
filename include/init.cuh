/*
 initialize all param for dynamical systems and setting of MCMPC
*/

#include <math.h>
#include "params.cuh"
#include "DataStructure.cuh"

void init_host_vector(float *params, float *states, float *constraints, float *matrix_elements);
void init_variables(SystemControlVariable *ret);
unsigned int countBlocks(unsigned int a, unsigned int b);