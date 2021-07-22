/*

*/
#include "../include/NewtonLikeMethod.cuh"

void NewtonLikeMethodInputSaturation(float *In, float Umax, float Umin)
{
    for(int i = 0; i < HORIZON; i++)
    {
        if(In[i] > Umax)
        {
            In[i] = Umax -zeta;
        }
        if(In[i] < Umin)
        {
            In[i] = Umin + zeta;
        }
    }
}


__global__ void NewtonLikeMethodGetTensorVector(QHP *Out, SampleInfo *In, int *indices)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    int next_indices = 0;

    for(int i = 0; i < HORIZON; i++)
    {
        for(int j = i; j < HORIZON; j++)
        {
#ifdef USING_WEIGHTED_LEAST_SQUARES
            Out[id].tensor_vector[next_indices] = In[indices[id]].Input[i] * In[indices[id]].Input[j] * sqrtf( In[indices[id]].WHM );

            Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[i] * In[indices[id]].Input[j] * In[indices[id]].WHM;

#else
            Out[id].tensor_vector[next_indices] = In[indices[id]].Input[i] * In[indices[id]].Input[j];

            Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[i] * In[indices[id]].Input[j];

#endif
            next_indices += 1;
        }
    }
    for(int i = 0; i < HORIZON; i++)
    {
#ifdef USING_WEIGHTED_LEAST_SQUARES
        Out[id].tensor_vector[next_indices] = In[indices[id]].Input[i] * sqrtf( In[indices[id]].WHM );
        Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[i] * In[indices[id]].WHM;

#else
        Out[id].tensor_vector[next_indices] = In[indices[id]].Input[i];
        Out[id].column_vector[next_indices] = In[indices[id]].L * In[indices[id]].Input[i];
#endif
        next_indices += 1;
    }

#ifdef USING_WEIGHTED_LEAST_SQUARES
    Out[id].tensor_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = 1.0f * sqrtf( In[indices[id]].WHM );
    Out[id].column_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = In[indices[id]].L * In[indices[id]].WHM;
#else
    Out[id].tensor_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = 1.0f;
    Out[id].column_vector[NUM_OF_PARABOLOID_COEFFICIENT - 1] = In[indices[id]].L; 
#endif
    __syncthreads();
}


// 正規方程式に関するベクトルGを計算するやつ
__global__ void NewtonLikeMethodGenNormalizationMatrix(float *Mat, QHP *elements, int SAMPLE_SIZE, int Ydimention)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * Ydimention + ix;

    Mat[idx] = 0.0f; //initialization
    for(int index = 0; index < SAMPLE_SIZE; index++)
    {
        Mat[idx] += elements[index].tensor_vector[ix] * elements[index].tensor_vector[iy];
    }
    __syncthreads();
}


__global__ void NewtonLikeMethodGenNormalizationVector(float *Vec, QHP *elements, int SAMPLE_SIZE)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    Vec[id] = 0.0f; //initialization
    for(int index = 0; index < SAMPLE_SIZE; index++)
    {
        Vec[id] += elements[index].column_vector[id];
    }
    __syncthreads( );
}

__global__ void NewtonLikeMethodGenNormalEquation(float *Mat, float *Vec, QHP *elements, int SAMPLE_SIZE, int Ydimention)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * Ydimention + ix;
    Mat[idx] = 0.0f;
    if(idx < Ydimention)
    {
        for(int index = 0; index < SAMPLE_SIZE; index++)
        {
            Mat[idx] += elements[index].tensor_vector[ix] * elements[index].tensor_vector[iy];
            Vec[idx] += elements[index].column_vector[idx];
        }
    }else{
        for(int index = 0; index < SAMPLE_SIZE; index++)
        {
            Mat[idx] += elements[index].tensor_vector[ix] * elements[index].tensor_vector[iy];
        }
    }
}

__global__ void NewtonLikeMethodGetRegularMatrix(float *Mat, QHP *element, int Sample_size)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    Mat[id] = 0.0f;
    for(int index = 0; index < Sample_size; index++)
    {
        Mat[id] += element[index].tensor_vector[threadIdx.x] * element[index].tensor_vector[blockIdx.x];
    }
    __syncthreads();
}

__global__ void NewtonLikeMethodGetRegularVector(float *Vec, QHP *element, int Sample_size)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    Vec[id] = 0.0f;
    for(int index = 0; index < Sample_size; index++)
    {
        Vec[id] += element[index].column_vector[id];
    }
    __syncthreads();
}



__global__ void NewtonLikeMethodGetHessianElements(float *HessElement, float *ansVec)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    HessElement[id] = ansVec[id];
    // __syncthreads( ); 
}

__global__ void NewtonLikeMethodGetHessianOriginal(float *Hessian, float *HessianElements)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    float temp_here;

    int vector_id = blockIdx.x;
    if(threadIdx.x <= blockIdx.x)
    {
        for(int t_id = 0; t_id < threadIdx.x; t_id++)
        {
            int sum_a = t_id + 1;
            vector_id += (HORIZON - sum_a);
        }
        temp_here = HessianElements[vector_id];
    }else{
        temp_here = 0.0f;
    }

    if(threadIdx.x != blockIdx.x)
    {
        if(isnan(temp_here))
        {
            Hessian[id] = Hessian[id];
        }else{
            Hessian[id] = temp_here / 2;
        }
    }else{
        if(isnan(temp_here))
        {
            Hessian[id] = 1.0f;
        }else{
            Hessian[id] = temp_here;
        }
    }
    // __syncthreads();
}

__global__ void NewtonLikeMethodGetLowerTriangle(float *LowerTriangle, float *UpperTriangle)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int t_id = blockIdx.x + threadIdx.x * blockDim.x;

    LowerTriangle[id] = UpperTriangle[t_id];
}

__global__ void NewtonLikeMethodGetFullHessian(float *FullHessian, float *LowerTriangle)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

    //この条件文が反対の可能性もある。2021.7.16(NAK)
    if(blockIdx.x < threadIdx.x )
    {
        if(!FullHessian[id] == LowerTriangle[id])
        {
            FullHessian[id] = LowerTriangle[id];
        }
    }
}

__global__ void NewtonLikeMethodGetFullHessianUtoL(float *FullHessian, float *UpperTriangle)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(blockIdx.x > threadIdx.x)
    {
        if(!FullHessian[id] == UpperTriangle[id])
        {
            FullHessian[id] = UpperTriangle[id];
        }
    }
}

__global__ void NewtonLikeMethodGetGradient(float *Gradient, float *elements, int index)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    Gradient[id] = elements[index + id];
    __syncthreads( );
}