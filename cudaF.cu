#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <time.h>
#include <cuda/atomic>
#include <helper_functions.h>
#include <nvjpeg.h>

#define CUDA_CALL(x) { cudaError_t err = x; if((err) != cudaSuccess) { \
    printf("Error: %s at %s:%d\n",cudaGetErrorString (err),__FILE__,__LINE__); \
    }}

bool getNum(std::ifstream& file, int& num) {
    num = 0;
    char a;
    file.read(&a, 1);
    while (isdigit(a)) {
        num = num * 10 + a - '0';
        file.read(&a, 1);
    }
    if (isalpha(a)) {
        printf("separator error ");
        return false;
    }
    return true;
}

bool readFile(std::string name, float** pics,int& width,int& heigh) {
    
    std::ifstream file(name, std::ios::binary | std::ios::in);
    if (!file.good()) {
        printf("file error for %s\n", name.c_str());
        return false;
    }

    int maxV=0;
    char a;
    file.read(&a,1);
    if (a != 'P') {
        printf("file type error for %s\n", name.c_str());
        return false;
    }
    file.read(&a, 1);
    if (a != '6') {
        printf("file format error for %s\n", name.c_str());
        return false;
    }
    file.read(&a, 1);
    if (isalnum(a)) {
        printf("separation error for %s\n", name.c_str());
        return false;
    }
    if (!getNum(file, width) || !getNum(file, heigh) || !getNum(file, maxV)) {
        printf("for %s\n", name.c_str());
        return false;
    }
    unsigned char* P = new unsigned char[width * heigh * 3];
    file.read((char*)P, width * heigh * 3);

    file.close();

    float* N = new float[width * heigh * 3];
    for (int i = 0; i < width * heigh * 3; i++)
        N[i] = P[i] / (float)maxV;

    delete[] P;

    CUDA_CALL(cudaMalloc((void**)pics, width * heigh * 3 * sizeof(float)));
    CUDA_CALL(cudaMemcpy(*pics, N, width * heigh * 3 * sizeof(float), cudaMemcpyHostToDevice));

    delete[] N;

    std::cout << "loaded file: " << name << std::endl;

    return true;
}

void saveFile(std::string name, float* pics, int width, int heigh, int maxV) {
    std::ofstream file(name, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!file.good()) {
        printf("file error for %s\n", name);
        return;
    }
    char header[] = "P6\n";
    file.write(header, 3);
    std::string s = std::to_string(width)+"\n";
    file.write(s.c_str(), s.size());
    s = std::to_string(heigh) + "\n";
    file.write(s.c_str(), s.size());
    s = std::to_string(maxV) + "\n";
    file.write(s.c_str(), s.size());

    unsigned char* data = new unsigned char[width * heigh * 3];
    for (int i = 0; i < width * heigh * 3; i++) {
        data[i] = pics[i] * maxV;
    }

    file.write((char*)data, width * heigh * 3);

    file.close();
    delete[] data;

    std::cout << "saved file: " << name << std::endl;
}