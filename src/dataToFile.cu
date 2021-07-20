/*
ある行列の固有値、固有ベクトルを.txtファイルに書き込む関数
*/
#include "../include/dataToFile.cuh"

void get_timeParam(int *tparam,int month, int day, int hour, int min, int step)
{
    tparam[0] = month;
    tparam[1] = day;
    tparam[2] = hour;
    tparam[3] = min;
    tparam[4] = step;
}

void write_Matrix_Information(float *data, dataName *d_name, int *timeparam)
{
    FILE *fp;
    char filename_Temp[35];
    sprintf(filename_Temp,"%s_%d%d_%d%d_%dstep.txt", d_name->name, timeparam[0], timeparam[1], timeparam[2], timeparam[3], timeparam[4]);
    fp = fopen(filename_Temp, "w");
    int nameSize = d_name->dimSize;
    for(int row = 0; row < nameSize; row++){
        for(int col = 0; col < nameSize; col++){
            if(col == nameSize -1)
            {
                fprintf(fp,"%f\n", data[row + col * nameSize]);
            }else{
                fprintf(fp,"%f ", data[row + col * nameSize]);
            }
        }
    }
    fclose(fp);
}