
#include "utils.h"

#define BLOCK_SIZE 128

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

#define CONFLICT_FREE_INDEX(index) (index + CONFLICT_FREE_OFFSET(index))

template<typename T>
void printDevMem(const T * addr, size_t len)
{
    T *h_data = (T*)malloc(len * sizeof(T));
    
    checkCudaErrors(cudaMemcpy(h_data, addr, len * sizeof(T), cudaMemcpyDeviceToHost));
    
    for (size_t i = 0; i < len; i++)
    {
        printf("h_data[%lu]=%lu\n", (unsigned long)i, (unsigned long)h_data[i]);
    }
    
    free(h_data);
    
}

template<typename T>
__global__ void uniformAdd(T *g_odata, T *g_scanSumPerBlock, size_t total_len, int block_base)
{
    __shared__ T prevBlockSum;
    
    unsigned int blkId = (block_base + blockIdx.x + blockIdx.y * gridDim.x);
    unsigned int baseIdx = ((blkId + 1) * blockDim.x + threadIdx.x) << 1;
    
    
    if (threadIdx.x == 0)
    {
        prevBlockSum = g_scanSumPerBlock[blkId + 1];
    }
    
    __syncthreads();
    
    if (baseIdx < total_len)
    {
        g_odata[baseIdx] += prevBlockSum;
    }
    if (baseIdx + 1 < total_len)
    {
        g_odata[baseIdx + 1] += prevBlockSum;
    }
}

template<typename T>
__global__ void prescan(T *g_odata,
    const T *g_idata,
    unsigned int data_len,
    T *g_scanSumPerBlock,
    int block_base)
{
    extern __shared__ unsigned char s_data1[];
    T *s_data = (T*)s_data1;
    
    unsigned int baseIdx = (block_base + blockIdx.x + blockIdx.y * gridDim.x) * data_len;
    
    int thid = threadIdx.x;
    int offset = 1;
    
    int ai = 2 * thid;
    int bi = 2 * thid + 1;
    
    s_data[CONFLICT_FREE_INDEX(ai)] = g_idata[baseIdx + 2 * thid]; // load input into shared memory
    s_data[CONFLICT_FREE_INDEX(bi)] = g_idata[baseIdx + 2 * thid + 1];
    
    for (int d = data_len >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (thid < d)
        {
            ai = offset*(2 * thid + 1) - 1;
            bi = offset*(2 * thid + 2) - 1;
            
            s_data[CONFLICT_FREE_INDEX(bi)] += s_data[CONFLICT_FREE_INDEX(ai)];
        }
        offset <<= 1;
    }
    
    if (thid == 0) { s_data[CONFLICT_FREE_INDEX(data_len - 1)] = 0; }
    
    for (unsigned int d = 1; d < data_len; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            ai = offset*(2 * thid + 1) - 1;
            bi = offset*(2 * thid + 2) - 1;
            
            T t = s_data[CONFLICT_FREE_INDEX(ai)];
            s_data[CONFLICT_FREE_INDEX(ai)] = s_data[CONFLICT_FREE_INDEX(bi)];
            s_data[CONFLICT_FREE_INDEX(bi)] += t;
        }
    }
    __syncthreads();
    
    g_odata[baseIdx + 2 * thid] = s_data[CONFLICT_FREE_INDEX(2 * thid)];
    g_odata[baseIdx + 2 * thid + 1] = s_data[CONFLICT_FREE_INDEX(2 * thid + 1)];
    
    if (thid == 0 && g_scanSumPerBlock != 0)
    {
        int i = block_base + blockIdx.x + blockIdx.y * gridDim.x;
        T t = s_data[CONFLICT_FREE_INDEX(data_len - 1)] +
        g_idata[baseIdx + data_len - 1];
        
        g_scanSumPerBlock[i] = t;
    }
}

template<typename T>
__global__ void addLastSum(T *d_lastSumVal, T *outArray, const T *inArray, size_t numElements)
{
    d_lastSumVal[0] = outArray[numElements - 1] + inArray[numElements - 1];
}

template<typename T>
void prescanArray(T *outArray, const T *inArray, const size_t numElements, T *d_lastSumVal)
{
    unsigned int numBlocks = max(1, (int)ceil((float)numElements / (2.f * BLOCK_SIZE)));
    
    if (numBlocks > 1)
    {
        unsigned int numThreads = min((size_t)BLOCK_SIZE, (size_t)(numElements / 2));
        
        unsigned int numEltsPerBlock = numThreads * 2;
        //unsigned int numFullBlock = numBlocks;
        
        size_t numLastBlockElts = numElements - (numBlocks - 1) * numEltsPerBlock;
        if (numLastBlockElts == numEltsPerBlock)
        {
            numLastBlockElts = 0;
        }
        
        T *g_scanSumPerBlock;
        
        size_t x_numsBlockSum = numBlocks;//numFullBlock;
        size_t y_numsBlockSum = 1;
        while (x_numsBlockSum > 65535) {
            y_numsBlockSum <<= 1;
            x_numsBlockSum >>= 1;
        }
        
        dim3 dim3Block(x_numsBlockSum, y_numsBlockSum, 1);
        
        checkCudaErrors(cudaMalloc((void**)&g_scanSumPerBlock, numBlocks * sizeof(T)));
        
        // padding space is used to avoid shared memory bank conflicts
        unsigned int extraSpace = 2 * BLOCK_SIZE / NUM_BANKS;
        unsigned int sharedMemSize = sizeof(T) * (2 * BLOCK_SIZE + extraSpace);
        
        prescan<T> << < dim3Block, numThreads, sharedMemSize >> >
        (outArray, inArray, numEltsPerBlock, g_scanSumPerBlock, 0);
        
        
        if (y_numsBlockSum * x_numsBlockSum != numBlocks)
        {
            size_t finishedBlock = y_numsBlockSum * x_numsBlockSum;
            
            size_t leftBlock = (numBlocks - finishedBlock);
            dim3 dim3Blockleft(leftBlock, 1, 1);
            prescan<T> << < dim3Blockleft, numThreads, sharedMemSize >>>
            (outArray, inArray, numEltsPerBlock, g_scanSumPerBlock, (int)finishedBlock);
        }
        
        T *g_outscanBlockSum;
        checkCudaErrors(cudaMalloc((void**)&g_outscanBlockSum, numBlocks * sizeof(T)));
        
        prescanArray<T>(g_outscanBlockSum, g_scanSumPerBlock, numBlocks, 0);
        
        if (numLastBlockElts != 0)
        {
            unsigned int startLastBlock = numElements - numLastBlockElts;
            prescanArray<T>(&outArray[startLastBlock], &inArray[startLastBlock], numLastBlockElts, 0);
        }
        
        int x_numBlocks = numBlocks - 1;
        int y_numBlocks = 1;
        while (x_numBlocks > 65535) {
            y_numBlocks <<= 1;
            x_numBlocks >>= 1;
        }
        
        dim3 dim3UniformBlock(x_numBlocks, y_numBlocks, 1);
        uniformAdd <T> << < dim3UniformBlock, BLOCK_SIZE >> > (outArray, g_outscanBlockSum, numElements, 0);
        
        //printDevMem1(outArray, inArray, numElements);
        
        if (y_numBlocks * x_numBlocks != numBlocks - 1)
        {
            unsigned long long finishedBlock = y_numBlocks * x_numBlocks;
            
            unsigned int leftBlock = (unsigned int)((long long)numBlocks - 1 - finishedBlock);
            dim3 dim3Blockleft(leftBlock, 1, 1);
            uniformAdd<T> << < dim3Blockleft, numThreads >> >
            (outArray, g_outscanBlockSum, numElements, (int)finishedBlock);
        }
        
        if (d_lastSumVal != 0)
        {
            addLastSum << <1, 1 >> > (d_lastSumVal, outArray, inArray, numElements);
        }
        
        checkCudaErrors(cudaFree(g_outscanBlockSum));
        checkCudaErrors(cudaFree(g_scanSumPerBlock));
        
    }
    else
    {
        T *d_tmp, *d_out;
        checkCudaErrors(cudaMalloc(&d_tmp, sizeof(T)*BLOCK_SIZE * 2));
        checkCudaErrors(cudaMalloc(&d_out, sizeof(T)*BLOCK_SIZE * 2));
        
        
        checkCudaErrors(cudaMemset(d_tmp, 0, sizeof(T)*BLOCK_SIZE * 2));
        checkCudaErrors(cudaMemcpy((void*)d_tmp, (void*)inArray, numElements * sizeof(T), cudaMemcpyDeviceToDevice));
        
        unsigned int extraSpace = 2 * BLOCK_SIZE / NUM_BANKS;
        unsigned int sharedMemSize = sizeof(T) * (2 * BLOCK_SIZE + extraSpace);
        
        prescan<T> << < 1, BLOCK_SIZE, sharedMemSize >> > (d_out, d_tmp, (size_t)(BLOCK_SIZE * 2), 0, 0);
        
        checkCudaErrors(cudaMemcpy((void*)outArray, (void*)d_out, numElements * sizeof(T), cudaMemcpyDeviceToDevice));
        
        if (d_lastSumVal != 0)
        {
            //printDevMem(d_lastSumVal, 1);
            addLastSum << <1, 1 >> > (d_lastSumVal, outArray, inArray, numElements);
        }
        
        
        checkCudaErrors(cudaFree(d_tmp));
        checkCudaErrors(cudaFree(d_out));
    }
}

__global__ void convert_exclusive_to_inclusice(size_t *outArray, const size_t *inArray,
size_t *d_lastSumVal, size_t numElements)
{
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    if (thid >= numElements)
    {
        return;
    }
    int q = 0;
    if (thid == numElements - 1)
    {
        q = d_lastSumVal[0];
    }
    else {
        q = inArray[thid + 1];
    }
    outArray[thid] = q;
}

void exclusive_scanArray(size_t *outArray, const size_t *inArray, size_t numElements, size_t *d_lastSumVal)
{
    prescanArray<size_t>(outArray, inArray, numElements, d_lastSumVal);
}

void exclusive_scan(size_t *outArray, const size_t *inArray, size_t numElements, size_t *d_lastSumVal)
{
    prescanArray<size_t>(outArray, inArray, numElements, d_lastSumVal);
}

void exclusive_scan(int *outArray, const int *inArray, size_t numElements, int *d_lastSumVal)
{
    prescanArray<int>(outArray, inArray, numElements, d_lastSumVal);
}

//thrust::inclusive_scan(thrust::device, d_interiorMask, d_interiorMask + srcSize, d_interiorMaskPrefixSum);
void inclusive_scan(size_t *outArray, const size_t *inArray, size_t numElements)
{
    size_t *d_lastSumVal, *d_exclusivesum;
    
    checkCudaErrors(cudaMalloc(&d_exclusivesum, numElements * sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&d_lastSumVal, sizeof(size_t)));
    
    exclusive_scanArray(d_exclusivesum, inArray, numElements, d_lastSumVal);
    convert_exclusive_to_inclusice << <ceil(numElements/1024.0), 1024 >> > (
    outArray, d_exclusivesum, d_lastSumVal, numElements);
    
    //cudaDeviceSynchronize();
    checkCudaErrors(cudaFree(d_exclusivesum));
    checkCudaErrors(cudaFree(d_lastSumVal));
}
