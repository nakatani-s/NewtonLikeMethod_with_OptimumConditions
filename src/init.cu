/*
 initialize all parameter for System model and cost function
*/ 
#include "../include/init.cuh"
void init_params( float *a )
{
    // params for simple nonlinear systems
    // for Simple Nonlinear System
    /*a[0] = 1.0f;
    a[1] = 1.0f;*/

    // FOR CART AND POLE
    a[0] = 0.1f;
    a[1] = 0.024f;
    a[2] = 0.2f;
    a[3] = a[1] * powf(a[2],2) /3;
    a[4] = 1.265f;
    a[5] = 0.0000001;
    a[6] = 9.81f;
}

void init_state( float *a )
{
    // FOR CART AND POLE
    a[0] = 0.0f; //x
    a[1] = M_PI + 0.001f; //theta
    a[2] = 0.0f; //dx
    a[3] = 0.0f; //dth
}

void init_constraint( float *a )
{
    // FOR CONTROL CART AND POLE
    // For Quadric Fitting Superior Constraints parameters
    a[0] = -3.0f;
    a[1] = 3.0f;
    a[2] = -0.5f;
    a[3] = 0.5f;

    // For MC superior Parameter
    /*a[0] = -1.0f;
    a[1] = 1.0f;
    a[2] = -0.5f;
    a[3] = 0.5f;*/
}

void init_matrix( float *a )
{
    // FOR CAONTROL CART AND POLE
    // For Quadric Fitting Superior Weight Parameter
    /*a[0] = 3.0f;
    a[1] = 3.5f;
    a[2] = 0.0f;
    a[3] = 0.0f;
    a[4] = 1.0f;*/

    // For MC superior Parameter
    a[0] = 3.0f;
    a[1] = 10.0f;
    a[2] = 0.05f;
    a[3] = 0.01f;
    a[4] = 0.5f;
}

void init_host_vector(float *params, float *states, float *constraints, float *matrix_elements)
{
    init_params(params);
    init_state(states);
    init_constraint(constraints);
    init_matrix(matrix_elements);
}

void init_variables(SystemControlVariable *ret)
{
    float param[DIM_OF_PARAMETERS], state[DIM_OF_STATES], constraints[NUM_OF_CONSTRAINTS], weightMatrix[DIM_OF_WEIGHT_MATRIX];
    init_params(param);
    init_state(state);
    init_constraint(constraints);
    init_matrix(weightMatrix);
    for(int i = 0; i < DIM_OF_PARAMETERS; i++)
    {
        ret->params[i] = param[i];
    }
    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        ret->state[i] = state[i];
    }
    for(int i = 0; i < NUM_OF_CONSTRAINTS; i++)
    {
        ret->constraints[i] = constraints[i];
    }
    for(int i = 0; i < DIM_OF_WEIGHT_MATRIX; i++)
    {
        ret->weightMatrix[i] = weightMatrix[i];
    }
}
unsigned int countBlocks(unsigned int a, unsigned int b) {
	unsigned int num;
	num = a / b;
	if (a < b || a % b > 0)
		num++;
	return num;
}