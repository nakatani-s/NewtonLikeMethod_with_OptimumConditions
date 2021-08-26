/*
*/
#include<iostream>
#include <stdio.h>
#include <fstream>
#include <math.h>
#include <cuda.h>
#include <time.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <iomanip>

#include "include/params.cuh"
#include "include/init.cuh"
#include "include/Matrix.cuh"
#include "include/DataStructure.cuh"
#include "include/MCMPC.cuh"
#include "include/NewtonLikeMethod.cuh"
#include "include/optimum_conditions.cuh"
#include "include/dataToFile.cuh"
// #include "include/cudaErrorCheck.cuh"

#define CHECK(call)                                                  \
{                                                                    \
    const cudaError_t error = call;                                  \
    if (error != cudaSuccess)                                        \
    {                                                                \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                \
        printf("code:%d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                          \
        exit(1);                                                     \
    }                                                                \
}
#define CHECK_CUBLAS(call,str)                                                        \
{                                                                                     \
    if ( call != CUBLAS_STATUS_SUCCESS)                                               \
    {                                                                                 \
        printf("CUBLAS Error: %s : %s %d\n", str, __FILE__, __LINE__);                \
        exit(1);                                                                      \
    }                                                                                 \
}
#define CHECK_CUSOLVER(call,str)                                                      \
{                                                                                     \
    if ( call != CUSOLVER_STATUS_SUCCESS)                                             \
    {                                                                                 \
        printf("CUBLAS Error: %s : %s %d\n", str, __FILE__, __LINE__);                \
        exit(1);                                                                      \
    }                                                                                 \
}


int main(int argc, char **argv)
{
    /* 行列演算ライブラリ用に宣言する変数群 */
    cusolverDnHandle_t cusolverH = NULL;
    // cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    CHECK_CUSOLVER( cusolverDnCreate(&cusolverH),"Failed to Create cusolver handle");

    cublasHandle_t handle_cublas = 0;
    cublasCreate(&handle_cublas);

    /* メインの実験データ書き込み用ファイルの宣言　*/
    FILE *fp, *opco;
    time_t timeValue;
    struct tm *timeObject;
    time( &timeValue );
    timeObject = localtime( &timeValue );
    char filename1[35], filename2[40];
    sprintf(filename1,"data_system_%d%d_%d%d.txt",timeObject->tm_mon + 1, timeObject->tm_mday, timeObject->tm_hour,timeObject->tm_min);
    sprintf(filename2,"optimum_condition_%d%d_%d%d.txt", timeObject->tm_mon + 1, timeObject->tm_mday, timeObject->tm_hour, timeObject->tm_min);
    fp = fopen(filename1,"w");
    opco = fopen(filename2,"w");

    /* ホスト・デバイス双方で使用するベクトルの宣言 */
    // float hostParams[DIM_OF_PARAMETERS], hostState[DIM_OF_STATES], hostConstraint[NUM_OF_CONSTRAINTS], hostWeightMatrix[DIM_OF_WEIGHT_MATRIX];
    SystemControlVariable *hostSCV, *deviceSCV;
    hostSCV = (SystemControlVariable*)malloc(sizeof(SystemControlVariable));
    init_variables( hostSCV );
    CHECK( cudaMalloc(&deviceSCV, sizeof(SystemControlVariable)) );
    CHECK( cudaMemcpy(deviceSCV, hostSCV, sizeof(SystemControlVariable), cudaMemcpyHostToDevice) );
    

    /* GPUの設定用パラメータ */
    unsigned int numBlocks, /*randomBlocks,*/ randomNums, /*Blocks,*/ dimHessian, numUnknownParamQHP, numUnknownParamHessian;
    unsigned int paramsSizeQuadHyperPlane;
    randomNums = NUM_OF_SAMPLES * (DIM_OF_INPUT + 1) * HORIZON;
    // randomBlocks = countBlocks(randomNums, THREAD_PER_BLOCKS);
    numBlocks = countBlocks(NUM_OF_SAMPLES, THREAD_PER_BLOCKS);
    // Blocks = numBlocks;
    dimHessian = HORIZON * HORIZON;

    numUnknownParamQHP = NUM_OF_PARABOLOID_COEFFICIENT;
    numUnknownParamHessian = numUnknownParamQHP - (HORIZON + 1);
    paramsSizeQuadHyperPlane = numUnknownParamQHP; //ホライズンの大きさに併せて、局所サンプルのサイズを決定
    paramsSizeQuadHyperPlane = paramsSizeQuadHyperPlane + addTermForLSM;
    // dim3 block(MAX_DIVISOR,1);
    dim3 block(1,1);
    dim3 grid((numUnknownParamQHP + block.x - 1)/ block.x, (numUnknownParamQHP + block.y -1) / block.y);
    printf("#NumBlocks = %d\n", numBlocks);
    printf("#NumBlocks = %d\n", numUnknownParamQHP);

#ifdef WRITE_MATRIX_INFORMATION
    float *WriteHessian, *WriteRegular;
    WriteHessian = (float *)malloc(sizeof(float)*dimHessian);
    WriteRegular = (float *)malloc(sizeof(float)* NUM_OF_PARABOLOID_COEFFICIENT * NUM_OF_PARABOLOID_COEFFICIENT);
    int timerParam[5] = { };
    dataName *name;
    name = (dataName*)malloc(sizeof(dataName)*3);
#endif

    /* MCMPC用の乱数生成用のseedを生成する */
    curandState *deviceRandomSeed;
    cudaMalloc((void **)&deviceRandomSeed, randomNums * sizeof(curandState));
    setup_kernel<<<NUM_OF_SAMPLES, (DIM_OF_INPUT + 1) * HORIZON>>>(deviceRandomSeed, rand());
    cudaDeviceSynchronize();
    
    /* 入力・コスト・最適性残差等の情報をまとめた構造体の宣言 */
    SampleInfo *deviceSampleInfo, *hostSampleInfo, *hostEliteSampleInfo, *deviceEliteSampleInfo;
    hostSampleInfo = (SampleInfo *)malloc(sizeof(SampleInfo) * NUM_OF_SAMPLES);
    hostEliteSampleInfo = (SampleInfo*)malloc(sizeof(SampleInfo) * NUM_OF_ELITES);
    cudaMalloc(&deviceSampleInfo, sizeof(SampleInfo) * NUM_OF_SAMPLES);
    cudaMalloc(&deviceEliteSampleInfo, sizeof(SampleInfo) * NUM_OF_ELITES);
    /*SampleInfo *TemporarySampleInfo, *deviceTempSampleInfo;
    TemporarySampleInfo = (SampleInfo *)malloc(sizeof(SampleInfo) * paramsSizeQuadHyperPlane);
    CHECK(cudaMalloc(&deviceTempSampleInfo, sizeof(SampleInfo) * paramsSizeQuadHyperPlane) );*/

    Tolerance *hostTol;
    hostTol = (Tolerance*)malloc(sizeof(Tolerance)*HORIZON+1);

    /* ２次超平面フィッティングの結果を反映する行列及びベクトルの宣言　（<---最適値計算にも使用）*/
    float *Hessian, *invHessian, *lowerHessian, *HessianElements;
    float *Gradient;
    CHECK( cudaMalloc(&Hessian, sizeof(float) * dimHessian) );
    CHECK( cudaMalloc(&invHessian, sizeof(float) * dimHessian) );
    CHECK( cudaMalloc(&lowerHessian, sizeof(float) * dimHessian) );
    CHECK( cudaMalloc(&HessianElements, sizeof(float) * numUnknownParamQHP) );

    CHECK( cudaMalloc(&Gradient, sizeof(float) * HORIZON) );

    /* 最小２乗法で２次超曲面を求める際に使用する配列の宣言 */
    float *Gmatrix, *invGmatrix, *CVector, *ansCVector;
    CHECK( cudaMalloc(&CVector, sizeof(float) * numUnknownParamQHP) );
    CHECK( cudaMalloc(&ansCVector, sizeof(float) * numUnknownParamQHP) );
    CHECK( cudaMalloc(&Gmatrix, sizeof(float) * numUnknownParamQHP * numUnknownParamQHP) );
    CHECK( cudaMalloc(&invGmatrix, sizeof(float) * numUnknownParamQHP * numUnknownParamQHP) );


    QHP *deviceQHP;
    CHECK( cudaMalloc(&deviceQHP, sizeof(QHP) * paramsSizeQuadHyperPlane) );

    unsigned int qhpBlocks;
    qhpBlocks = countBlocks(paramsSizeQuadHyperPlane, THREAD_PER_BLOCKS);
    printf("#qhpBlocks = %d\n", qhpBlocks);

    // 行列演算ライブラリ用の変数の宣言及び定義
    const int m_Rmatrix = numUnknownParamQHP;

    int work_size, w_si_hessian;
    float *work_space, *w_sp_hessian;
    int *devInfo;
    CHECK( cudaMalloc((void**)&devInfo, sizeof(int) ) );

    /* thrust使用のためのホスト/デバイス用ベクトルの宣言 */ 
    thrust::host_vector<int> indices_host_vec( NUM_OF_SAMPLES );
    thrust::device_vector<int> indices_device_vec = indices_host_vec;
    thrust::host_vector<float> sort_key_host_vec( NUM_OF_SAMPLES );
    thrust::device_vector<float> sort_key_device_vec = sort_key_host_vec; 

    /* 推定入力のプロット・データ転送用　*/
    float *hostData, *deviceData, *hostTempData, *deviceTempData;
    hostData = (float *)malloc(sizeof(float) * HORIZON);
    hostTempData = (float *)malloc(sizeof(float) * HORIZON);
    CHECK(cudaMalloc(&deviceData, sizeof(float) * HORIZON));
    cudaMalloc(&deviceTempData, sizeof(float) * HORIZON);
    for(int i = 0; i < HORIZON; i++){
        hostData[i] = 0.0f;
    }
    CHECK( cudaMemcpy(deviceData, hostData, sizeof(float) * HORIZON, cudaMemcpyHostToDevice));

    /* 制御ループの開始 */
    float F_input = 0.0f;
    float MCMPC_F, Proposed_F;
    // float costFromMCMPC, costFromProposed, toleranceFromMCMPC, toleranceFromProposed;
    float cost_now;
    float optimumConditions[2] = { };
    float optimumCondition_p[2] = { };
    float var;

    float process_gpu_time, procedure_all_time;
    clock_t start_t, stop_t;
    cudaEvent_t start, stop;
    
    dim3 inverseGmatrix(numUnknownParamQHP, numUnknownParamQHP);
    dim3 grid_inverse(HORIZON, HORIZON);
    dim3 threads((HORIZON + grid_inverse.x -1) / grid_inverse.x, (HORIZON + grid_inverse.y -1) / grid_inverse.y);

#ifdef USING_QR_DECOMPOSITION
    // float *QR_work_space = NULL;
    float *ws_QR_operation = NULL;
    int geqrf_work_size = 0;
    int ormqr_work_size = 0;
    int QR_work_size = 0;
    const int nrhs = 1;
    float *QR_tau = NULL;
    cublasSideMode_t side = CUBLAS_SIDE_LEFT;
    cublasOperation_t trans = CUBLAS_OP_T;
    cublasOperation_t trans_N = CUBLAS_OP_N;
    cublasFillMode_t uplo_QR = CUBLAS_FILL_MODE_UPPER;
    cublasDiagType_t cub_diag = CUBLAS_DIAG_NON_UNIT;
    CHECK(cudaMalloc((void**)&QR_tau, sizeof(float) * numUnknownParamQHP));
#endif


    for(int t = 0; t < SIM_TIME; t++)
    {
        shift_Input_vec( hostData );
        CHECK( cudaMemcpy(deviceData, hostData, sizeof(float) * HORIZON, cudaMemcpyHostToDevice) );
        start_t = clock();

        if(t == 0)
        {
            start_t = clock();
            for(int iter = 0; iter < ITERATIONS_MAX; iter++)
            {
                var = variance / sqrt(iter + 1);
                // var = variance / 2;
                MCMPC_Cart_and_SinglePole<<<numBlocks, THREAD_PER_BLOCKS>>>( deviceSCV, var, deviceRandomSeed, deviceData, deviceSampleInfo, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
                cudaDeviceSynchronize();
                thrust::sequence(indices_device_vec.begin(), indices_device_vec.end());
                thrust::sort_by_key(sort_key_device_vec.begin(), sort_key_device_vec.end(), indices_device_vec.begin());

                
                getEliteSampleInfo<<<NUM_OF_ELITES, 1>>>(deviceEliteSampleInfo, deviceSampleInfo, thrust::raw_pointer_cast( indices_device_vec.data() ));
                CHECK( cudaMemcpy(hostEliteSampleInfo, deviceEliteSampleInfo, sizeof(SampleInfo) * NUM_OF_ELITES, cudaMemcpyDeviceToHost) );
                // weighted_mean(hostData, NUM_OF_ELITES, hostSampleInfo);
                weighted_mean(hostData, NUM_OF_ELITES, hostEliteSampleInfo);
                MCMPC_F = hostData[0];
                /*if(iter == 0)
                {
                    sprintf(name[2].inputfile, "initSolution.txt");
                    name[2].dimSize = HORIZON;
                    resd_InitSolution_Input(hostData, &name[2]);
                }*/
                CHECK( cudaMemcpy(deviceData, hostData, sizeof(float) * HORIZON, cudaMemcpyHostToDevice) );
                calc_OC_for_Cart_and_SinglePole_hostF(optimumConditions, hostData, hostSCV, hostTol);
                printf("cost :: %f   KKT_Error :: %f\n", optimumConditions[0], optimumConditions[1]);
            }
            name[1].dimSize = HORIZON;
            sprintf(name[1].name,"InitInputData.txt");
            write_Vector_Information(hostData, &name[1]);
            stop_t = clock();
            procedure_all_time = stop_t - start_t;
            printf("Geometrical cooling MCMPC computation time :: %f\n", procedure_all_time / CLOCKS_PER_SEC);
        }else{
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
            start_t = clock();
            for(int iter = 0; iter < ITERATIONS; iter++)
            {
                var = variance / 2.0f;
                MCMPC_Cart_and_SinglePole<<<numBlocks, THREAD_PER_BLOCKS>>>( deviceSCV, var, deviceRandomSeed, deviceData, deviceSampleInfo, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
                cudaDeviceSynchronize();
                thrust::sequence(indices_device_vec.begin(), indices_device_vec.end());
                thrust::sort_by_key(sort_key_device_vec.begin(), sort_key_device_vec.end(), indices_device_vec.begin());

                // CHECK( cudaMemcpy(hostSampleInfo, deviceSampleInfo, sizeof(SampleInfo) * NUM_OF_SAMPLES, cudaMemcpyDeviceToHost) );
                getEliteSampleInfo<<<NUM_OF_ELITES, 1>>>(deviceEliteSampleInfo, deviceSampleInfo, thrust::raw_pointer_cast( indices_device_vec.data() ));
                CHECK( cudaMemcpy(hostEliteSampleInfo, deviceEliteSampleInfo, sizeof(SampleInfo) * NUM_OF_ELITES, cudaMemcpyDeviceToHost) );
                weighted_mean(hostData, NUM_OF_ELITES, hostEliteSampleInfo);
                MCMPC_F = hostData[0];

                CHECK( cudaMemcpy(deviceData, hostData, sizeof(float) * HORIZON, cudaMemcpyHostToDevice) );
                var = neighborVar;
                MCMPC_Cart_and_SinglePole<<<numBlocks, THREAD_PER_BLOCKS>>>( deviceSCV, var, deviceRandomSeed, deviceData, deviceSampleInfo, thrust::raw_pointer_cast( sort_key_device_vec.data() ));
                cudaDeviceSynchronize();
                thrust::sequence(indices_device_vec.begin(), indices_device_vec.end());
                thrust::sort_by_key(sort_key_device_vec.begin(), sort_key_device_vec.end(), indices_device_vec.begin());

                NewtonLikeMethodGetTensorVector<<< qhpBlocks, THREAD_PER_BLOCKS>>>(deviceQHP, deviceSampleInfo, thrust::raw_pointer_cast( indices_device_vec.data() ));
                cudaDeviceSynchronize();

                // 1024以下の"NUM_OF_PARABOLOID_COEFFICIENT"の最大約数を(thread数 / block)として計算させる方針で実行
                // 以下は正規方程式における行列の各要素を取得する関数
                // NewtonLikeMethodGenNormalizationMatrix<<<grid, block>>>(Gmatrix, deviceQHP, paramsSizeQuadHyperPlane, NUM_OF_PARABOLOID_COEFFICIENT);

                /*-----------------Error detect 2021.07.20----------------------------*/
                // Following Function has any Error (ThreadId or BlockId) --> it is required to modify original mode.
                // NewtonLikeMethodGenNormalEquation<<<grid, block>>>(Gmatrix, CVector, deviceQHP, paramsSizeQuadHyperPlane, NUM_OF_PARABOLOID_COEFFICIENT);
                NewtonLikeMethodGetRegularMatrix<<<NUM_OF_PARABOLOID_COEFFICIENT, NUM_OF_PARABOLOID_COEFFICIENT>>>(Gmatrix, deviceQHP, paramsSizeQuadHyperPlane);
                NewtonLikeMethodGetRegularVector<<<NUM_OF_PARABOLOID_COEFFICIENT, 1>>>(CVector, deviceQHP, paramsSizeQuadHyperPlane);
                cudaDeviceSynchronize();
#ifdef WRITE_MATRIX_INFORMATION
                if(t<300){
                    if(t % 50 == 0){
                        get_timeParam(timerParam, timeObject->tm_mon+1, timeObject->tm_mday, timeObject->tm_hour, timeObject->tm_min, t);
                        sprintf(name[0].name, "RegularMatrix");
                        name[0].dimSize = NUM_OF_PARABOLOID_COEFFICIENT;
                        CHECK(cudaMemcpy(WriteRegular, Gmatrix, sizeof(float) * NUM_OF_PARABOLOID_COEFFICIENT * NUM_OF_PARABOLOID_COEFFICIENT, cudaMemcpyDeviceToHost));
                        write_Matrix_Information(WriteRegular, &name[0], timerParam);
                    }
                }else{
                    if(t % 250 == 0){
                        get_timeParam(timerParam, timeObject->tm_mon+1, timeObject->tm_mday, timeObject->tm_hour, timeObject->tm_min, t);
                        sprintf(name[0].name, "RegularMatrix");
                        name[0].dimSize = NUM_OF_PARABOLOID_COEFFICIENT;
                        CHECK(cudaMemcpy(WriteRegular, Gmatrix, sizeof(float) * NUM_OF_PARABOLOID_COEFFICIENT * NUM_OF_PARABOLOID_COEFFICIENT, cudaMemcpyDeviceToHost));
                        write_Matrix_Information(WriteRegular, &name[0], timerParam);
                    }

                }
#endif

#ifndef USING_QR_DECOMPOSITION
                //以下は、正規方程式（最小二乗法で使用）のベクトル(正規方程式：Gx = v の v)の各要素を計算する関数
                // NewtonLikeMethodGenNormalizationVector<<<NUM_OF_PARABOLOID_COEFFICIENT, 1>>>(CVector, deviceQHP, paramsSizeQuadHyperPlane);
                // cudaDeviceSynchronize();

                CHECK_CUSOLVER( cusolverDnSpotrf_bufferSize(cusolverH, uplo, m_Rmatrix, Gmatrix, m_Rmatrix, &work_size), "Failed to get bufferSize");
                CHECK(cudaMalloc((void**)&work_space, sizeof(float) * work_size));

                CHECK_CUSOLVER( cusolverDnSpotrf(cusolverH, uplo, m_Rmatrix, Gmatrix, m_Rmatrix, work_space, work_size, devInfo), "Failed to inverse operation for G");
                MatrixSetUpLargeIdentityMatrix<<<grid, block>>>(invGmatrix, NUM_OF_PARABOLOID_COEFFICIENT);
                cudaDeviceSynchronize();

                CHECK_CUSOLVER( cusolverDnSpotrs(cusolverH, uplo, m_Rmatrix, m_Rmatrix, Gmatrix, m_Rmatrix, invGmatrix, m_Rmatrix, devInfo), "Failed to get inverse Matrix G");

                // 正規方程式をcuBlasで解く
                CHECK_CUBLAS( cublasSgemv(handle_cublas, CUBLAS_OP_N, m_Rmatrix, m_Rmatrix, &alpha, invGmatrix, m_Rmatrix, CVector, 1, &beta, ansCVector, 1),"Failed to get Estimate Input Sequences");
#else
                if(t==1){
                    CHECK_CUSOLVER( cusolverDnSgeqrf_bufferSize(cusolverH, m_Rmatrix, m_Rmatrix, Gmatrix, m_Rmatrix, &geqrf_work_size), "Failed to get buffersize for QR decom [1]" );
                    CHECK_CUSOLVER( cusolverDnSormqr_bufferSize(cusolverH, side, trans, m_Rmatrix, nrhs, m_Rmatrix, Gmatrix, m_Rmatrix, QR_tau, CVector, m_Rmatrix, &ormqr_work_size), "Failed to get buffersize for QR decom [2]" );
                    
                    QR_work_size = (geqrf_work_size > ormqr_work_size)? geqrf_work_size : ormqr_work_size;
                }
                CHECK( cudaMalloc((void**)&ws_QR_operation, sizeof(float) * QR_work_size) );
                /* compute QR factorization */ 
                CHECK_CUSOLVER( cusolverDnSgeqrf(cusolverH, m_Rmatrix, m_Rmatrix, Gmatrix, m_Rmatrix, QR_tau, ws_QR_operation, QR_work_size, devInfo),"Failed to compute QR factorization" );

                CHECK_CUSOLVER( cusolverDnSormqr(cusolverH, side, trans, m_Rmatrix, nrhs, m_Rmatrix, Gmatrix, m_Rmatrix, QR_tau, CVector, m_Rmatrix, ws_QR_operation, QR_work_size, devInfo), "Failed to compute Q^T*B" );
                CHECK(cudaDeviceSynchronize());

                CHECK_CUBLAS( cublasStrsm(handle_cublas, side, uplo_QR, trans_N, cub_diag, m_Rmatrix, nrhs, &alpha, Gmatrix, m_Rmatrix, CVector, m_Rmatrix), "Failed to compute X = R^-1Q^T*B" );
                CHECK(cudaDeviceSynchronize());

                NewtonLikeMethodCopyVector<<<numUnknownParamQHP, 1>>>(ansCVector, CVector);
#endif

                NewtonLikeMethodGetHessianElements<<<numUnknownParamHessian, 1>>>(HessianElements, ansCVector);
                CHECK(cudaDeviceSynchronize());
                // ヘシアンの上三角行列分の要素を取得
                NewtonLikeMethodGetHessianOriginal<<<HORIZON, HORIZON>>>(Hessian, HessianElements);
                CHECK(cudaDeviceSynchronize());

                NewtonLikeMethodGetLowerTriangle<<<HORIZON, HORIZON>>>(lowerHessian, Hessian);
                CHECK(cudaDeviceSynchronize());
                // NewtonLikeMethodGetFullHessianLtoU<<<HORIZON, HORIZON>>>(Hessian, lowerHessian);
                NewtonLikeMethodGetFullHessianUtoL<<<HORIZON, HORIZON>>>(lowerHessian, Hessian);
                NewtonLikeMethodGetGradient<<<HORIZON, 1>>>(Gradient, ansCVector, numUnknownParamHessian);
                MatrixMultiplyOperation<<<HORIZON,HORIZON>>>(Hessian, 2.0f, lowerHessian);

                CHECK_CUSOLVER( cusolverDnSpotrf_bufferSize(cusolverH, uplo, HORIZON, Hessian, HORIZON, &w_si_hessian), "Failed to get bufferSize of computing the inverse of Hessian");
                CHECK( cudaMalloc((void**)&w_sp_hessian, sizeof(float) * w_si_hessian) );
                CHECK_CUSOLVER( cusolverDnSpotrf(cusolverH, uplo, HORIZON, Hessian, HORIZON, w_sp_hessian, w_si_hessian, devInfo), "Failed to inverse operation");

                MatrixSetUpSmallIdentityMatrix<<<HORIZON, HORIZON>>>(invHessian);

                CHECK_CUSOLVER( cusolverDnSpotrs(cusolverH, uplo, HORIZON, HORIZON, Hessian, HORIZON, invHessian, HORIZON, devInfo), "Failed to get inverse of Hessian");

                // 逆行列を-1倍する操作
                MatrixMultiplyOperation<<<HORIZON, HORIZON>>>(Hessian, -1.0f, invHessian);
                CHECK_CUBLAS(cublasSgemv(handle_cublas, CUBLAS_OP_N, HORIZON, HORIZON, &alpha, Hessian, HORIZON, Gradient, 1, &beta, deviceTempData, 1), "Failed to get result by proposed method");

                CHECK( cudaMemcpy(hostTempData, deviceTempData, sizeof(float) * HORIZON, cudaMemcpyDeviceToHost) );
                NewtonLikeMethodInputSaturation(hostTempData, hostSCV->constraints[1], hostSCV->constraints[0]);
                Proposed_F = hostTempData[0]; //提案手法による推定入力値のうち最初の時刻のものをコピーする
                // 提案法の最適性条件を計算->比較(vs MC解)->物理シミュレーション(by RungeKutta4.5)->結果の保存
                calc_OC_for_Cart_and_SinglePole_hostF(optimumConditions, hostData, hostSCV, hostTol); //MC解に対する最適性条件を計算
                calc_OC_for_Cart_and_SinglePole_hostF(optimumCondition_p, hostTempData, hostSCV, hostTol); //提案手法に対する最適性条件を計算
                
            }
            cudaEventRecord(stop,0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&process_gpu_time, start, stop);
            stop_t = clock();
            procedure_all_time = stop_t - start_t;
        }
        printf("TIME stpe :: %f", t * interval);
        printf("MCMPC optimum condition := %f  Proposed optimum condition := %f\n", optimumConditions[1], optimumCondition_p[1]);
        printf("MCMPC cost value := %f  Proposed cost value := %f\n", optimumConditions[0], optimumCondition_p[0]);
        // 評価値比較に基づく投入する入力の決定
        if(optimumCondition_p[0] < optimumConditions[0] /*&& optimumCondition_p[1] < optimumConditions[1]*/)
        {
            F_input = Proposed_F;
            cost_now = optimumCondition_p[0];
            CHECK( cudaMemcpy(deviceData, hostTempData, sizeof(float) * HORIZON, cudaMemcpyHostToDevice) );
        }else{
            F_input = MCMPC_F;
            cost_now = optimumConditions[0];
            CHECK( cudaMemcpy(deviceData, hostData, sizeof(float) * HORIZON, cudaMemcpyHostToDevice) );
        }

        Runge_Kutta45_for_SecondaryOderSystem( hostSCV, F_input, interval);
        CHECK( cudaMemcpy(deviceSCV, hostSCV, sizeof(SystemControlVariable), cudaMemcpyHostToDevice) );
        
        fprintf(fp, "%f %f %f %f %f %f %f %f\n", t * interval, F_input, MCMPC_F, Proposed_F, hostSCV->state[0], hostSCV->state[1], hostSCV->state[2], hostSCV->state[3]);
        fprintf(opco, "%f %f %f %f %f %f %f %f\n", t * interval, cost_now, optimumConditions[0], optimumCondition_p[0], optimumConditions[1], optimumCondition_p[1], process_gpu_time/10e3, procedure_all_time/CLOCKS_PER_SEC);
    }
    if(cusolverH) cusolverDnDestroy(cusolverH);
    if(handle_cublas) cublasDestroy(handle_cublas);
    fclose(fp);
    fclose(opco);
    cudaDeviceReset( );
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}