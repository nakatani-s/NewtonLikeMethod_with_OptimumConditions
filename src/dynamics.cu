/*
 dynamics.cu
*/
#include "../include/dynamics.cuh"

__host__ __device__ float Cart_type_Pendulum_ddx(float u, float x, float theta, float dx, float dtheta, SystemControlVariable *SCV)
{
    float a[10];
    /*a[0] = state[3] + powf(state[2], 2) * state[1];		//J+l^2*mp
	a[1] = u - dx * state[4]
			+ powf(dtheta, 2) * state[2] * state[1] * sinf(theta);//u-dx*myuc+dtheta^2*l*mp*sin
	a[2] = cosf(theta) * state[2] * state[1];						//cos*l*mp
	a[3] = dtheta * state[5] - state[6] * state[2] * state[1] * sinf(theta);//dtheta*myup-g*l*mp*sin
	a[4] = -(a[0] * a[1] + a[2] * a[3]);

	a[5] = powf(cosf(theta), 2) * powf(state[2], 2) * powf(state[1], 2);//cos^2*l^2*mp^2
	a[6] = state[0] + state[1];		//mc+mp
	a[7] = state[3] + powf(state[2], 2) * state[1];		//J+l^2*mp
	a[8] = a[5] - (a[6] * a[7]);*/

    a[0] = SCV->params[3] + powf(SCV->params[2], 2) * SCV->params[1]; // J + mplp^2
    a[1] = u - dx * SCV->params[4] + powf(dtheta, 2) * SCV->params[2] * SCV->params[1] * sinf(theta); // u - dx * myu_c *dtheta^2 * mp * lp * sin(theta)
    a[2] = cosf(theta) * SCV->params[2] * SCV->params[1]; // mp * lp * cos(theta)
    a[3] = dtheta * SCV->params[5] - SCV->params[6] * SCV->params[2] * SCV->params[1] * sinf(theta); //myu_p * dtheta - mp * g * lp * sin(theta)
    a[4] = -(a[0] * a[1] + a[2] * a[3]);

    a[5] = powf(cosf(theta), 2) * powf(SCV->params[2], 2) * powf(SCV->params[1], 2); //mp^2 * lp^2 * cos^2(theta)
    a[6] = SCV->params[0] + SCV->params[1]; // mp + mc
    a[7] = SCV->params[3] + powf(SCV->params[2],2) * SCV->params[1];
    a[8] = a[5] - (a[6] * a[7]);

	return a[4] / a[8];
}

__host__ __device__ float Cart_type_Pendulum_ddtheta(float u, float x,  float theta, float dx, float dtheta, SystemControlVariable *SCV)
{
    float a[10];
    /*a[0] = cosf(theta) * state[2] * state[1];		//cos*l*mp
	a[1] = u - dx * state[4]
			+ powf(dtheta, 2) * state[2] * state[1] * sinf(theta);//u-dx*myuc+dtheta^2*l*mp*sin
	a[2] = state[0] + state[1];		//mc+mp
	a[3] = dtheta * state[5] - state[6] * state[2] * state[1] * sinf(theta);//dtheta*myup-g*l*mp*sin
	a[4] = -(a[0] * a[1] + a[2] * a[3]);

	a[5] = state[3] * (state[0] + state[1]);		//J(mc+mp)
	a[6] = powf(state[2], 2) * state[1];		//l^2*mp
	a[7] = state[0] + state[1] - powf(cosf(theta), 2) * state[1];//mc+mp-cos^2*mp
	a[8] = a[5] + a[6] * a[7];*/

    a[0] = cosf(theta) * SCV->params[2] * SCV->params[1]; //mp * lp * cos(theta)
    a[1] = u - dx * SCV->params[4] + powf(dtheta, 2) * SCV->params[2] * SCV->params[1] * sinf(theta); //u - myu_c * dx + mp * lp * dtheta^2 *sin(theta)
    a[2] = SCV->params[0] + SCV->params[1]; //mc + mp
    a[3] = dtheta * SCV->params[5] - SCV->params[6] * SCV->params[2] * SCV->params[1] * sinf(theta); // myu_p * dtheta - mp * lp * sin(theta)
    a[4] = -(a[0] * a[1] + a[2] * a[3]);

    a[5] = SCV->params[3] * (SCV->params[0] + SCV->params[1]);
    a[6] = powf(SCV->params[2], 2) * SCV->params[1];
    a[7] = SCV->params[0] + SCV->params[1] - powf(cosf(theta), 2) * SCV->params[1];
    a[8] = a[5] + a[6] * a[7];

	return a[4] / a[8];
}

__host__ __device__ void get_Lx_Cart_and_SinglePole(float *Lx, Tolerance *prev, SystemControlVariable *SCV)
{
    // float temp_Lx[DIM_OF_STATES] = { };
    Lx[0] = SCV->weightMatrix[0] * prev->state[0];
    Lx[1] = SCV->weightMatrix[1] * cosf(prev->state[1] / 2) * sinf(prev->state[1] / 2) * 0.5f;
    Lx[2] = SCV->weightMatrix[2] * prev->state[2];
    Lx[3] = SCV->weightMatrix[3] * prev->state[3];
}

__host__ __device__ void get_LFx_Cart_and_SinglePole(float *LFx, Tolerance *current, Tolerance *later, SystemControlVariable *SCV, float t_delta)
{
    float a[36] = { };
    float /*x,*/ th, dx, dtheta;
    // x = current->state[0];
    th = current->state[1];
    dx = current->state[2];
    dtheta = current->state[3];

    a[0] = - later->lambda[0]; //lambda^T * Fx(:,1) := LFx[0]

    a[1] = -powf(SCV->params[1] * SCV->params[2] * cosf(th), 2) + powf(SCV->params[1] * SCV->params[2], 2)
            + SCV->params[0] * SCV->params[1] * powf(SCV->params[2], 2) + SCV->params[3] * (SCV->params[0] + SCV->params[1]); // -(Mp*lp*cos(th))^2 + (Mp*lp)^2 + Mc*Mp*lp^2 + Jp * (Mp + Mc)
    a[2] = (SCV->params[1] * powf(SCV->params[2], 2) + SCV->params[3]) * SCV->params[1] * powf(dtheta, 2) * SCV->params[2] * cosf(th); // (Jp + Mp*lp^2) * Mp * lp * dth^2 * cos(th)
    a[3] = powf(SCV->params[1] * SCV->params[2] * cosf(th), 2) * SCV->params[6]; //(Mp*lp*cos(th))^2 * g
    a[4] = (SCV->params[5] * dtheta - SCV->params[1] * SCV->params[6] * SCV->params[2] * sinf(th)); // (mup * dtheta - Mp * lp * g * sin(th) )
    a[5] = SCV->params[1] * SCV->params[2] * sinf(th); //Mp * lp * sin(th)
    a[6] = a[4] * a[5]; //Mp * lp * sin(th) *(mup * dtheta - Mp * lp * g * sin(th) )
    a[7] = 2.0f * powf(SCV->params[1] * SCV->params[2], 3) * powf(cosf(th), 2) * sinf(th); //2 * Mp^3 * lp^3 * cos^2(th) * sin(th)
    a[8] = 2.0f * powf(SCV->params[1] * SCV->params[2], 2) * cosf(th) * sinf(th); //2 * Mp^2 * lp^2 * cos(th) * sin(th)
    a[9] = SCV->params[3] + SCV->params[1] * powf(SCV->params[2], 2); // Jp + Mp * lp^2
    a[10] = SCV->params[1] * SCV->params[2] * powf(dtheta, 2) * sinf(th) + current->Input[0] - SCV->params[4] * dx; // [Mp * lp * dtheta^2 * sinf(th) + U - muc * dx]
    a[11] = SCV->params[1] * SCV->params[2]; //Mp * lp
    a[12] = 2.0f * SCV->params[1] * powf( SCV->params[2], 2) * dtheta * sinf(th) + SCV->params[5] * cosf(th) + 2.0f * SCV->params[3] * sinf(th); //2 * Mp * lp^2 *dtheta * sin(th) + mup * cos(th) + 2 * Jp * dtheta * sin(th)
    
    a[13] = a[2] / a[1]; //[Fx;32]_1
    a[14] = a[3] / a[1]; //[Fx;32]_2
    a[15] = a[6] / a[1]; //[Fx;32]_3
    a[16] = a[1] * a[1]; //a[1]^2
    a[17] = (a[7] * a[4]) / a[16]; //[Fx;32]_4
    a[18] = (a[8] * a[9] * a[10]) / a[16]; //[Fx;32]_5
    a[19] = (a[13] - a[14] - a[15] - a[17] - a[18]) * t_delta; // Fx;32
    
    a[20] = -(SCV->params[4] * a[9]) / a[1]; //[Fx;33] = -myuc * (Jp + Mp * lp^2) / a[1]
    a[21] = a[20] * t_delta - 1.0f; // Fx;33

    a[22] = SCV->params[1] * SCV->params[2] * a[12] / a[1]; //[Fx;34] = Mp * lp * (2 * Mp * lp^2 * dtheta * sin(th) + myup * cos(th) + 2 * Jp * dtheta * sinf(th) ) / a[1]
    a[23] = a[22] * t_delta; //Fx;34

    a[24] = (a[5] * a[10]) / a[1]; //[Fx;42]_1
    a[25] =  powf(SCV->params[1] * SCV->params[2] * cosf(th), 2); // (Mp*lp*cos(th))^2
    a[26] = a[22] * powf(dtheta, 2) / a[1]; //[Fx;42]_2
    a[27] = a[11] * SCV->params[6] * cosf(th) * (SCV->params[0] + SCV->params[1]) / a[1]; //[Fx;42]_3
    a[28] = (a[7] * a[10]) / a[16]; //[Fx;42]_4
    a[29] = (a[8] * (SCV->params[0] + SCV->params[1]) * a[4]) / a[16]; //[Fx;42]_5
    a[30] = (a[24] - a[26] + a[27] + a[28] + a[29]) * t_delta; // Fx;42 

    a[31] = SCV->params[5] * SCV->params[1] * SCV->params[2] * cosf(th) / a[1]; //[Fx;43]
    a[32] = a[31] * t_delta; //Fx;43

    a[33] = -SCV->params[5] * (SCV->params[0] + SCV->params[1]) / a[1]; //[Fx;44]_1
    a[34] = -a[8] / a[1]; //[Fx;44]_2
    a[35] = (a[29] + a[30]) * t_delta - 1.0f; //Fx;44

    // 行列Fxの面倒臭い要素の計算まで”一応終了”<--- 2021.07.12

    LFx[0] = a[0];
    LFx[1] = -later->lambda[1] + a[19] * later->lambda[2] + a[30] * later->lambda[3];
    LFx[2] = (t_delta * later->lambda[0]) + (a[21] * later->lambda[2]) + (a[32] * later->lambda[3]);
    LFx[3] = (t_delta * later->lambda[1]) + (a[23] * later->lambda[2]) + (a[35] * later->lambda[3]); 
}

__host__ __device__ void get_dHdu_Cart_and_SinglePole(Tolerance *current, Tolerance *later, SystemControlVariable *SCV, float t_delta)
{
    float temp_Lu[DIM_OF_INPUT] = { };
    float temp_LBu[DIM_OF_INPUT] = { };
    float temp_Fu[DIM_OF_STATES] = { };
    float temp_LamFu[DIM_OF_INPUT] = { };

    float o[10] = { };
    temp_Lu[0] = SCV->weightMatrix[4] * current->Input[0];
    
    o[0] = powf(current->Input[0], 2) - powf(SCV->constraints[1], 2); // (U^2 - U_max^2)
    o[1] = 2.0f * current->Input[0]; // 2 * U
    o[2] = o[1] / o[0]; //LBu = 2U / (U^2 - U_max^2)

    o[3] = SCV->params[3] + SCV->params[1] * powf(SCV->params[2], 2); //Jp + Mp * lp^2
    o[4] = -SCV->params[1] * SCV->params[2] * cosf(current->state[1]); // -Mp * lp * cos(th)
    
    o[5] = -powf(SCV->params[1] * SCV->params[2] * cosf(current->state[1]), 2); //- Mp^2 * lp^2 * cos(th)^2
    o[6] = powf(SCV->params[1] * SCV->params[2], 2); // Mp^2 * lp^2
    o[7] = SCV->params[0] * SCV->params[1] * powf(SCV->params[2], 2); // Mc * Mp * lp^2
    o[8] = SCV->params[3] * (SCV->params[0] + SCV->params[1]); // Jp * (Mp + Mc)
    o[9] = o[5] + o[6] + o[7] + o[8];

    temp_LBu[0] = o[2];

    temp_Fu[2] = (o[3] * t_delta) / o[9];
    temp_Fu[3] = (o[4] * t_delta) / o[9];

    for(int i = 0; i < DIM_OF_INPUT; i++)
    {
        for(int k = 0; k < DIM_OF_STATES; k++)
        {
            temp_LamFu[i] += later->lambda[k] * temp_Fu[k];
        }
    }

    for(int i = 0; i < DIM_OF_INPUT; i++)
    {
        current->dHdu[i] = temp_Lu[i] + Rho * temp_LBu[i] + temp_LamFu[i];
    }
}

void get_curent_diff_state_Cart_and_SinglePole(float *diffState, float *state, float input, SystemControlVariable *SCV)
{
    diffState[0] = state[2]; //dx
    diffState[1] = state[3]; //dtheta
    diffState[2] = Cart_type_Pendulum_ddx(input, state[0], state[1], state[2], state[3], SCV);
    diffState[3] = Cart_type_Pendulum_ddtheta(input, state[0], state[1], state[2], state[3], SCV);
}

void Eular_integrator(float *yp_vector, float t_delta, float *diffState)
{
    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        yp_vector[i] = diffState[i] * t_delta;
    }
}

void Runge_Kutta45_for_SecondaryOderSystem(SystemControlVariable *SCV, float input, float t_delta)
{
    float state[DIM_OF_STATES], diff_state[DIM_OF_STATES], yp_1[DIM_OF_STATES], next_state[DIM_OF_STATES];
    // float params[DIM_OF_PARAMETERS] = { };
    // copyStateFromDataStructure(state, SCV);
    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        state[i] = SCV->state[i];
    }
    /*for(int i = 0; i < DIM_OF_PARAMETERS; i++){
        params[i] = SCV->params[i];
    }*/
    get_curent_diff_state_Cart_and_SinglePole( diff_state, state, input, SCV);
    Eular_integrator(yp_1, t_delta, diff_state);
    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        next_state[i] = state[i] + yp_1[i] / 2;
    }

    float yp_2[DIM_OF_STATES] = { };
    get_curent_diff_state_Cart_and_SinglePole( diff_state, next_state, input, SCV);
    Eular_integrator(yp_2, t_delta, diff_state);
    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        next_state[i] = state[i] + yp_2[i] / 2;
    }

    float yp_3[DIM_OF_STATES] = { };
    get_curent_diff_state_Cart_and_SinglePole( diff_state, next_state, input, SCV);
    Eular_integrator(yp_3, t_delta, diff_state);
    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        next_state[i] = state[i] + yp_3[i];
    }

    float yp_4[DIM_OF_STATES] = { };
    get_curent_diff_state_Cart_and_SinglePole( diff_state, next_state, input, SCV);
    Eular_integrator(yp_4, t_delta, diff_state);

    for(int i = 0; i < DIM_OF_STATES; i++)
    {
        SCV->state[i] = state[i] + (yp_1[i] + 2 * yp_2[i] + 2 * yp_3[i] + yp_4[i]) / 6.0f;
    }
}