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
#include <thread>
#include <mutex>
#include <condition_variable>

#include <helper_gl.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#include "cudaF.cuh"

#define SEED 12
#define PRIMITIVAE 1000
#define PRIMITIVAE_DATA_SIZE 10*PRIMITIVAE

#define CUDA_CALL(x) { cudaError_t err = x; if((err) != cudaSuccess) { \
    printf("Error: %s at %s:%d\n",cudaGetErrorString (err),__FILE__,__LINE__); \
    }}

#define POW(x) ((x)*(x))

__device__ curandStateXORWOW_t randState;
__device__ cudaTextureObject_t imageB;
__device__ float* pImg;

bool cont = true;
bool savePop = false;

struct calData;

calData* data;

GLuint pbo;
struct cudaGraphicsResource* cuda_pbo_resource;
unsigned int* d_output = nullptr;
std::atomic<bool> ready = false;
std::atomic<bool> end = false;
int widthG, heighG;
std::mutex m;
std::condition_variable cv;

std::atomic<bool> waiter = true;

struct calData {
    int populationSize;
    int eliminationSize;
    curandStateXORWOW_t state;
    float** population;
    float* precision;
    int best;
    int* eliminated;
    int* copied;
    float* newRes;
    float* randNum;
    unsigned long long n;
    int width;
    int heigh;
};

__device__ bool isRight(float ax, float ay, float bx, float by, float cx, float cy) {
    return (bx - ax) * (cy - ay) <= (by - ay) * (cx - ax);
}

__device__ void swapT(float& x, float& y) {
    float xt = x;
    x = y;
    y = xt;
}

__global__ void correctRes(float* res) {
    int i = threadIdx.x * 10;
    if (!isRight(res[i],res[i+1],res[i+2],res[i+3],res[i+4],res[i+5])) {
        swapT(res[i + 2], res[i + 4]);
        swapT(res[i + 3], res[i + 5]);
    }
}

__global__ void compPart(float* res, cudaTextureObject_t data,float* sum,int width,int heigh,float* img=nullptr,
    unsigned int* d_output = nullptr) {

    long j = blockIdx.x * 1024 + threadIdx.x;
    if (j >= width * heigh)
        return;
    float r=0,g=0,b=0,a=0;
    float x = ((j % width) + 0.5) / width;
    float y = ((j / width) + 0.5) / heigh;
    for (int i = 0; i < PRIMITIVAE_DATA_SIZE; i += 10) {
        if (isRight(res[i], res[i + 1], res[i + 2], res[i + 3], x, y)  && 
            isRight(res[i + 2], res[i + 3], res[i + 4], res[i + 5], x, y) && 
            isRight(res[i + 4], res[i + 5], res[i], res[i + 1], x, y)) {
            float alph = res[i + 9];
            r += res[i + 6] * alph;
            g += res[i + 7] * alph;
            b += res[i + 8] * alph;
            a += alph;
        }
    }
    if (a == 0)
        a = 1;
    r /= a;
    g /= a;
    b /= a;
    float diff = POW(r - tex1Dfetch<float>(data,j * 3)) + POW(g - tex1Dfetch<float>(data, j * 3 + 1)) + POW(b - tex1Dfetch<float>(data, j * 3 + 2));
    atomicAdd(sum, diff);
    if (img != nullptr) {
        img[j * 3] = r;
        img[j * 3 + 1] = g;
        img[j * 3 + 2] = b;
    }
    if (d_output != nullptr) {
        j = j%width + (heigh - j/width - 1) * width;
        d_output[j] = (int)(255*r) | ((int)(255*g) << 8) | ((int)(255 * b) << 16);
    }
}

__device__ float comp(float* res, cudaTextureObject_t data, int width, int heigh, float* img=nullptr,
    unsigned int* d_output=nullptr) {

    float *sum;
    CUDA_CALL(cudaMalloc((void**)&sum, sizeof(float)));
    *sum = 0;
    long hSplit = (long)width*heigh / 1024;
    if (hSplit * 1024 < (long)width * heigh)
        hSplit++;
    compPart << <hSplit, 1024 >> > (res,data,sum, width, heigh,img, d_output);
    CUDA_CALL(cudaDeviceSynchronize());
    float sumCopy = *sum;
    cudaFree(sum);
    return sqrtf(sumCopy);
}

__global__ void makeImg(cudaTextureObject_t picsObj, float* res, int width, int heigh, float* img) {
     comp(res, picsObj, width, heigh, img);
}

__host__ __device__ cudaTextureObject_t makeTexObj(float* data,int imgSize) {
    cudaTextureObject_t picsObj;
    cudaResourceDesc res;
    res.resType = cudaResourceType::cudaResourceTypeLinear;
    res.res.linear.devPtr = data;
    res.res.linear.desc = cudaCreateChannelDesc(sizeof(float) * 8, 0, 0, 0,
        cudaChannelFormatKind::cudaChannelFormatKindFloat);
    res.res.linear.sizeInBytes = imgSize * 3 * sizeof(float);
    cudaTextureDesc des;
    des.addressMode[0] = cudaTextureAddressMode::cudaAddressModeBorder;
    des.addressMode[1] = cudaTextureAddressMode::cudaAddressModeBorder;
    des.addressMode[2] = cudaTextureAddressMode::cudaAddressModeBorder;
    des.filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
    des.readMode = cudaTextureReadMode::cudaReadModeElementType;
    des.sRGB = false;
    des.borderColor[0] = 0;
    des.borderColor[1] = 0;
    des.borderColor[2] = 0;
    des.borderColor[3] = 0;
    des.normalizedCoords = false;
    des.maxAnisotropy = 1;
    des.mipmapFilterMode = cudaTextureFilterMode::cudaFilterModeLinear;
    des.mipmapLevelBias = 0;
    des.minMipmapLevelClamp = 0;
    des.maxMipmapLevelClamp = 0;
    des.disableTrilinearOptimization = false;

    CUDA_CALL(cudaCreateTextureObject(&picsObj, &res, &des, nullptr));
    return picsObj;
}

__device__ void clearMem() {
    cudaFree(pImg);
}

__global__ void prepPopulation(float** population) {
    curandStateXORWOW_t state;
    curand_init(SEED, 100 + threadIdx.x,0, &state);
    CUDA_CALL(cudaMalloc((void**)&(population[threadIdx.x]), PRIMITIVAE_DATA_SIZE*sizeof(float)));
    for (int i = 0; i < PRIMITIVAE_DATA_SIZE; i++) {
        population[threadIdx.x][i] = curand_uniform(&state);
    }
}

__inline__ __device__ void populationDestroy(float** popualtion,int populationSize) {
    for (int i = 0; i < populationSize; i++) {
        cudaFree(popualtion[i]);
    }
    cudaFree(popualtion);
}

__device__ float analizeElement(float* el, int elNum,cudaTextureObject_t image, int width, int heigh) {
    correctRes<<<1, PRIMITIVAE>>>(el);
    cudaDeviceSynchronize();
    return comp(el, image,width,heigh);
}

__global__ void firstAssesment(float* prec,float** population, cudaTextureObject_t image, int width, int heigh) {
    prec[blockIdx.x] = analizeElement(population[blockIdx.x], blockIdx.x,image, width, heigh);
}

__device__ int findMin(float* prec, int populationSize) {
    int idx = 0;
    for (int i = 1; i < populationSize; i++)
        if (prec[idx] > prec[i])
            idx = i;
    return idx;
}

__global__ void selectSort(float* table, int begin, int end) {
    int m;
    for (int j = begin; j < end; j++) {
        m = j;
        for (int i = j + 1; i <= end; i++) {
            if (table[i] < table[m])
                m = i;
        }
        swapT(table[j], table[m]);
    }
}

__inline__ __global__ void qsort(float* table, int begin, int end,int depth=0) {
    float cur = table[end];
    int i = begin;
    int j = end;
    while (i < j) {
        while (table[i] < cur)
            i++;
        while (table[j] >= cur && i<j)
            j--;
        swapT(table[i], table[j]);
    }
    swapT(table[i], table[end]);
    if (begin - i < 20 || depth>15) {
        if (i - end < 20 || depth>15) {
            selectSort<<<1,1>>>(table, i + 1, end);
        }
        else {
            qsort << <1, 1 >> > (table, i + 1, end,depth+1);
        }
        selectSort << <1, 1 >> > (table, begin, i-1);
    }
    else {
        if (i - end < 20) {
            selectSort << <1, 1 >> > (table, i + 1, end);
        }
        else {
            qsort << <1, 1 >> > (table, i + 1, end,depth + 1);
        }
        qsort << <1, 1 >> > (table, begin, i-1, depth + 1);
    }
    CUDA_CALL(cudaDeviceSynchronize());
}

__device__ int chooseToEliminate(float* prec, const int populationSize, const int eliminationSize, int* eliminated,int best, curandStateXORWOW_t* state) {
    float sum=0;
    for (int i = 1; i < populationSize; i++)
        sum += prec[i];
    sum -= prec[best];
    float* randy = new float[eliminationSize];
    for (int j = 0; j < eliminationSize; j++)
        randy[j] = curand_uniform(state);

    qsort<<<1,1>>>(randy, 0,eliminationSize-1);
    CUDA_CALL(cudaDeviceSynchronize());

    float current = 0;
    int idx = 0;
    int k = 1;
    for (int i = 0; i < populationSize; i++) {
        if (i != best) {
            current += prec[i] / sum;
            if (current >= randy[idx] || populationSize - i - k <= eliminationSize - idx) {
                sum -= prec[i];
                prec[i] = -1;
                eliminated[idx++] = i;
                if (idx == eliminationSize)
                    break;
            }
        }
        else if(i>best)
            k = 0;
    }

    delete[] randy;
}

__device__ int chooseToReproduce(float* prec, const int populationSize, const int eliminationSize, int* copied, int best, curandStateXORWOW_t* state) {
    float sum = 0;
    for (int i = 1; i < populationSize; i++)
        if(prec[i]!=-1)
            sum += 1/(prec[i]);

    float* randy = new float[eliminationSize];
    for (int j = 0; j < eliminationSize; j++)
        randy[j] = curand_uniform(state);

    qsort << <1, 1 >> > (randy, 0, eliminationSize - 1);
    CUDA_CALL(cudaDeviceSynchronize());

    float current = 0;
    int idx = 0;
    for (int i = 0; i < populationSize; i++) {
        if (prec[i] != -1) {
            current += (1 / prec[i]) / sum;
            while (current >= randy[idx]) {
                copied[idx++] = i;
                if (idx == eliminationSize)
                    break;
            }
        }
    }

    delete[] randy;
}

__global__ void reproduce(float** population,int* eliminated,int* copied,float* randNum) {
    int i = blockIdx.x;
    int from = copied[i];
    int to = eliminated[i];
    memcpy(population[to], population[from], PRIMITIVAE_DATA_SIZE * sizeof(float));
    int pos = int(PRIMITIVAE_DATA_SIZE*(1- randNum[2*i]));
    population[to][pos] += (-0.5 + randNum[2*i+1]) / 2;
    if (population[to][pos] < 0)
        population[to][pos] = 0;
    else if (population[to][pos] > 1)
        population[to][pos] = 1;
}

__global__ void analizeNew(float** population, int* eliminated, cudaTextureObject_t image,float* newRes, int width, int heigh) {
    int i = blockIdx.x;
    newRes[i] = analizeElement(population[eliminated[i]], eliminated[i], image, width, heigh);
}

__global__ void startCal(calData& data, const int populationSize, const int eliminationSize, int width, int heigh, cudaTextureObject_t image) {
    data.populationSize = populationSize;
    data.eliminationSize = eliminationSize;
    data.width = width;
    data.heigh = heigh;

    curand_init(SEED, 17, 0, &(data.state));

    CUDA_CALL(cudaMalloc((void**)&(data.population), data.populationSize * sizeof(float*)));
    prepPopulation << <1, data.populationSize >> > (data.population);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMalloc((void**)&(data.precision), data.populationSize * sizeof(float)));
    firstAssesment << <data.populationSize, 1 >> > (data.precision, data.population, image, data.width, data.heigh);
    CUDA_CALL(cudaDeviceSynchronize());
    data.best = findMin(data.precision, data.populationSize);

    CUDA_CALL(cudaMalloc((void**)&(data.eliminated), data.eliminationSize * sizeof(int)));

    CUDA_CALL(cudaMalloc((void**)&(data.copied), data.eliminationSize * sizeof(int)));

    CUDA_CALL(cudaMalloc((void**)&(data.newRes), data.eliminationSize * sizeof(float)));

    CUDA_CALL(cudaMalloc((void**)&(data.randNum), 2 * data.eliminationSize * sizeof(float)));

    data.n = 0;
}

__global__ void startLoad(calData& data) {
    CUDA_CALL(cudaMalloc((void**)&(data.population), data.populationSize * sizeof(float*)));

    for (int i = 0; i < data.populationSize; i++) {
        CUDA_CALL(cudaMalloc((void**)&(data.population[i]), PRIMITIVAE_DATA_SIZE * sizeof(float)));
    }

    CUDA_CALL(cudaMalloc((void**)&(data.precision), data.populationSize * sizeof(float)));

    CUDA_CALL(cudaMalloc((void**)&(data.eliminated), data.eliminationSize * sizeof(int)));

    CUDA_CALL(cudaMalloc((void**)&(data.copied), data.eliminationSize * sizeof(int)));

    CUDA_CALL(cudaMalloc((void**)&(data.newRes), data.eliminationSize * sizeof(float)));

    CUDA_CALL(cudaMalloc((void**)&(data.randNum), 2 * data.eliminationSize * sizeof(float)));
}

__global__ void calMid(calData& data, cudaTextureObject_t image) {
    int k = 100;
    while (k-- && data.precision[data.best] != 0) {

        chooseToEliminate(data.precision, data.populationSize, data.eliminationSize, data.eliminated, data.best, &(data.state));
        chooseToReproduce(data.precision, data.populationSize, data.eliminationSize, data.copied, data.best, &(data.state));

        for (int i = 0; i < 2 * data.eliminationSize; i++)
            data.randNum[i] = curand_uniform(&(data.state));

        reproduce << <data.eliminationSize, 1 >> > (data.population, data.eliminated, data.copied, data.randNum);
        CUDA_CALL(cudaDeviceSynchronize());

        analizeNew << <data.eliminationSize, 1 >> > (data.population, data.eliminated, image, data.newRes, data.width, data.heigh);
        CUDA_CALL(cudaDeviceSynchronize());

        for (int i = 0; i < data.eliminationSize; i++) {
            data.precision[data.eliminated[i]] = data.newRes[i];
            if (data.newRes[i] < data.precision[data.best]) {
                data.best = data.eliminated[i];
            }
        }
        data.n++;
    }
    printf("best:%f,%d,%llu\n", data.precision[data.best], data.best,data.n);
}

__global__ void calFinal(calData& data, float* dominating) {
    cudaFree(data.randNum);
    cudaFree(data.newRes);
    cudaFree(data.copied);
    cudaFree(data.eliminated);

    memcpy(dominating, data.population[data.best], PRIMITIVAE_DATA_SIZE * sizeof(float));

    printf("final best:%f,%d\n", data.precision[data.best], data.best);

    cudaFree(data.precision);
    populationDestroy(data.population, data.populationSize);
}

void stop() {
    std::string s;
    while (s != "exit" && s!="save") {
        std::cin >> s;
    }
    cont = false;
    if (s == "save")
        savePop = true;
}

void initGLBuffers(){
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, widthG * heighG * sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW_ARB);

    CUDA_CALL(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));
}

__global__ void setBuffer(calData* data,cudaTextureObject_t image,int width,int heigh, unsigned int* d_output) {
    comp(data->population[data->best],image,width,heigh,nullptr,d_output);
}

void display() {
    if (ready) {
        std::unique_lock<std::mutex> lk(m);
        glClear(GL_COLOR_BUFFER_BIT);

        CUDA_CALL(cudaGraphicsMapResources(1, &cuda_pbo_resource));
        size_t num_bytes;
        CUDA_CALL(cudaGraphicsResourceGetMappedPointer((void**)&d_output, &num_bytes, cuda_pbo_resource));
        setBuffer<<<1,1>>>(data,imageB, widthG, heighG,d_output);

        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

        glDisable(GL_DEPTH_TEST);
        glRasterPos2i(0, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
        glDrawPixels(widthG, heighG, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        glFlush();
        glutReportErrors();
        ready = false;
        lk.unlock();
        cv.notify_one();
    }
}

void idle() {
    if (ready)
        glutPostRedisplay();
    if(end)
        glutLeaveMainLoop();
    waiter = true;
}

void cleanUp() {
    CUDA_CALL(cudaGraphicsUnregisterResource(cuda_pbo_resource));
    glDeleteBuffers(1, &pbo);
}

void reshape(int x, int y) {
    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void initGL() {
    int argc = 1;
    char* blah = new char[10];
    blah[0] = 'p';
    blah[1] = 0;
    glutInit(&argc, &blah);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowSize(widthG, heighG);
    glutCreateWindow("CUDA EVOLUTION IMG");
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    if (!isGLVersionSupported(2, 0) || !areGLExtensionsSupported("GL_ARB_pixel_buffer_object"))
    {
        fprintf(stderr, "Required OpenGL extensions are missing.");
        exit(EXIT_FAILURE);
    }
}

void makePhoto() {
    initGL();
    initGLBuffers();
    glutMainLoop();
}

void saveCurandState(std::ofstream& out, curandStateXORWOW_t* state) {
    out.write((char*)&(state->d), sizeof(unsigned int));
    out.write((char*)&(state->v), 5*sizeof(unsigned int));
    out.write((char*)&(state->boxmuller_flag), sizeof(int));
    out.write((char*)&(state->boxmuller_flag_double), sizeof(int));
    out.write((char*)&(state->boxmuller_extra), sizeof(float));
    out.write((char*)&(state->boxmuller_extra_double), sizeof(double));
}

__global__ void copyDataToSave(float** populationPtr, calData& dataO) {
    for (int i = 0; i < dataO.populationSize; i++) {
        memcpy(populationPtr[i], dataO.population[i], PRIMITIVAE_DATA_SIZE * sizeof(float));
    }
}

void saveState(calData& dataO,std::string name) {
    calData* data = new calData;
    CUDA_CALL(cudaMemcpy(data, &dataO, sizeof(calData), cudaMemcpyDeviceToHost));
    std::ofstream out(name, std::ios::binary | std::ios::out | std::ios::trunc);

    out.write((char*)&(data->populationSize), sizeof(int));
    out.write((char*)&(data->eliminationSize), sizeof(int));
    out.write((char*)&(data->best), sizeof(int));
    out.write((char*)&(data->n), sizeof(unsigned long long));
    out.write((char*)&(data->width), sizeof(int));
    out.write((char*)&(data->heigh), sizeof(int));

    saveCurandState(out, &(data->state));

    float** populations;
    CUDA_CALL(cudaMallocManaged((void**)&populations, data->populationSize * sizeof(float*)));
    for (int i = 0; i < data->populationSize; i++) {
        CUDA_CALL(cudaMallocManaged((void**)&(populations[i]), PRIMITIVAE_DATA_SIZE * sizeof(float)));
    }
    copyDataToSave << <1, 1 >> > (populations, dataO);
    CUDA_CALL(cudaDeviceSynchronize());

    for (int i = 0; i < data->populationSize; i++) {
        out.write((char*)(populations[i]), sizeof(float) * PRIMITIVAE_DATA_SIZE);
    }
    for (int i = 0; i < data->populationSize; i++)
        cudaFree(populations[i]);
    cudaFree(populations);

    out.close();

    delete data;
}

__global__ void copyDataToLoad(float** populationPtr, calData& dataO) {
    for (int i = 0; i < dataO.populationSize; i++) {
        memcpy(dataO.population[i], populationPtr[i], PRIMITIVAE_DATA_SIZE * sizeof(float));
    }
}

void loadCurandState(std::ifstream& in, curandStateXORWOW_t* state) {
    in.read((char*)&(state->d), sizeof(unsigned int));
    in.read((char*)&(state->v), 5 * sizeof(unsigned int));
    in.read((char*)&(state->boxmuller_flag), sizeof(int));
    in.read((char*)&(state->boxmuller_flag_double), sizeof(int));
    in.read((char*)&(state->boxmuller_extra), sizeof(float));
    in.read((char*)&(state->boxmuller_extra_double), sizeof(double));
}

__global__ void calPrep(calData& data, cudaTextureObject_t image){
    firstAssesment << <data.populationSize, 1 >> > (data.precision, data.population, image, data.width, data.heigh);
    CUDA_CALL(cudaDeviceSynchronize());
    data.best = findMin(data.precision, data.populationSize);
}

void loadState(calData& dataO, std::string name, cudaTextureObject_t image) {
    calData* data = new calData;
    
    std::ifstream in(name, std::ios::binary | std::ios::in);

    in.read((char*)&(data->populationSize), sizeof(int));
    in.read((char*)&(data->eliminationSize), sizeof(int));
    in.read((char*)&(data->best), sizeof(int));
    in.read((char*)&(data->n), sizeof(unsigned long long));
    in.read((char*)&(data->width), sizeof(int));
    in.read((char*)&(data->heigh), sizeof(int));

    loadCurandState(in, &(data->state));

    CUDA_CALL(cudaMemcpy(&dataO, data, sizeof(calData), cudaMemcpyHostToDevice));

    startLoad << <1, 1 >> > (dataO);
    CUDA_CALL(cudaDeviceSynchronize());

    float** populations;
    CUDA_CALL(cudaMallocManaged((void**)&populations, data->populationSize * sizeof(float*)));
    for (int i = 0; i < data->populationSize; i++) {
        CUDA_CALL(cudaMallocManaged((void**)&(populations[i]), PRIMITIVAE_DATA_SIZE * sizeof(float)));
    }

    for (int i = 0; i < data->populationSize; i++) {
        in.read((char*)(populations[i]), sizeof(float) * PRIMITIVAE_DATA_SIZE);
    }

    copyDataToLoad << <1, 1 >> > (populations,dataO);
    CUDA_CALL(cudaDeviceSynchronize());


    for (int i = 0; i < data->populationSize; i++)
        cudaFree(populations[i]);
    cudaFree(populations);

    calPrep<<<1,1>>>(dataO,image);
    CUDA_CALL(cudaDeviceSynchronize());
    in.close();

    delete data;
}

int main() {
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    printf("WRITE FILE NAME\n");
    char fileRead[1024];
    scanf("%[^\n]", fileRead);

    int width, heigh;
    if (!readFile(fileRead, &pImg, width,heigh)){
        return 1;
    }
    widthG = width;
    heighG = heigh;
    imageB = makeTexObj(pImg, width * heigh);

    printf("LOAD PREVIOUS STATE? [Y/N]\n");
    char answer;

    scanf("%c", &answer);//consume \n
    scanf("%c", &answer);

    std::thread t(makePhoto);

    CUDA_CALL(cudaMalloc((void**)&data, sizeof(calData)));

    if (answer == 'Y') {
        printf("WRITE PREV STATE FILE NAME\n");
        char prevFileRead[1024];
        scanf("%s", prevFileRead);
        scanf("%c", &answer);//consume \n
        loadState(*data, prevFileRead, imageB);
    }
    else {
        startCal << <1, 1 >> > (*data, 10, 3, width, heigh, imageB);
        CUDA_CALL(cudaDeviceSynchronize());
    }

    if(waiter){
        std::unique_lock<std::mutex> lk(m);
        ready = true;
        auto now = std::chrono::system_clock::now();
        std::cv_status st = cv.wait_until(lk, now + std::chrono::milliseconds(10));
        if (st == std::cv_status::timeout) {
            waiter = false;
            ready = false;
        }
    }

    std::thread t1(stop);

    while (cont) {
        calMid << <1, 1 >> > (*data, imageB);
        CUDA_CALL(cudaDeviceSynchronize());

        if (waiter) {
            std::unique_lock<std::mutex> lk(m);
            ready = true;
            auto now = std::chrono::system_clock::now();
            std::cv_status st = cv.wait_until(lk,now + std::chrono::milliseconds(10));
            if (st == std::cv_status::timeout) {
                waiter = false;
                ready = false;
            }
        }
    }

    if (savePop) {
        printf("WRITE NEXT STATE FILE NAME\n");
        char nextFileSave[1024];
        scanf("%s", nextFileSave);
        scanf("%c", &answer);//consume \n
        saveState(*data, nextFileSave);
    }
    float* res;
    CUDA_CALL(cudaMallocManaged((void**)&res, PRIMITIVAE_DATA_SIZE * sizeof(float)));

    calFinal << <1, 1 >> > (*data, res);
    CUDA_CALL(cudaDeviceSynchronize());

    cudaFree(data);
    if (!savePop) {
        float* imageDone;
        CUDA_CALL(cudaMallocManaged((void**)&imageDone, width * heigh * 3 * sizeof(float)));

        makeImg << <1, 1 >> > (imageB, res, width, heigh, imageDone);
        float* R = new float[width * heigh * 3];
        CUDA_CALL(cudaMemcpy(R, imageDone, width * heigh * 3 * sizeof(float), cudaMemcpyDeviceToHost));

        printf("WRITE RESULT FILE NAME\n");
        char resultFileSave[1024];
        scanf("%s", resultFileSave);
        scanf("%c", &answer);//consume \n

        saveFile(resultFileSave, R, width, heigh, 255);
        delete[] R;
        cudaFree(imageDone);
    }
    cudaFree(res);
    cudaDestroyTextureObject(imageB);
    cudaFree(pImg);
    t1.join();

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    end = true;
    t.join();
    return 0;
}
