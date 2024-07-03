#include <stdio.h>
#include <malloc.h>
#include <cuda.h>

#define N 10
#define c 0.002
#define delta_t 0.05
#define delta_s 0.04
#define epoch 125

__global__ void updateGPU(float *u_old, float *u_new)
{
    // [][N + 2]
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    //unsigned int idx = iy * N + ix;
    int i, j;
    i = iy + 1;
    j = ix + 1;
    /*printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) "
           "global index %2d i: %d j: %d \n",
           threadIdx.x, threadIdx.y, blockIdx.x,
           blockIdx.y, ix, iy, idx, i, j);
     */
    int x = N + 2;
    float *uij = u_old + (i * x + j);
    float *uip1j = u_old + ((i + 1) * x + j);
    float *uis1j = u_old + ((i - 1) * x + j);
    float *uijp1 = u_old + (i * x + j + 1);
    float *uijs1 = u_old + (i * x + j - 1);

    float *un = u_new + (i * x + j);

    *un = (*uij) + c * delta_t /
                       (delta_s * delta_s) *
                       ((*uip1j) + (*uis1j) - 4 * (*uij) + (*uijp1) + (*uijs1));
}

__global__ void copyGPU(float *u_old, float *u_new)
{
    // [][N + 2]
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    // unsigned int idx = iy * N + ix;
    int i, j;
    i = iy + 1;
    j = ix + 1;
    /*printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) "
           "global index %2d i: %d j: %d \n",
           threadIdx.x, threadIdx.y, blockIdx.x,
           blockIdx.y, ix, iy, idx, i, j);*/

    int x = N + 2;
    float *uij = u_old + (i * x + j);
    float *un = u_new + (i * x + j);
    *uij = *un;
}

void init1DArray(float *u_old)
{
    // [][N + 2]
    int x = N + 2;

    size_t n = N;
    float *p;
    for (size_t i = 0; i <= n + 1; i++)
    {
        for (size_t j = 0; j <= n + 1; j++)
        {
            p = u_old + (i * x + j);
            *p = 25;
        }
    }
    for (size_t j = 0; j <= n + 1; j++)
    {
        p = u_old + (j * x + N + 1);
        *p = 100;
        // u_old[j][6] = 100;
    }
}

void print1DArray(float *u_old)
{
    int n = N + 2;
    float *cell;
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            cell = u_old + (i * n + j);
            printf("%8.1f", *cell);
        }
        printf("\n");
    }
}

void parallelCode()
{
    int nx = N;
    int ny = N;
    dim3 block(N, 1);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    int size = (nx + 2) * (ny + 2) * sizeof(float);

    float *h_old = (float *)malloc(size);

    float *d_old;
    cudaMalloc((void **)&d_old, size);
    float *d_new;
    cudaMalloc((void **)&d_new, size);

    init1DArray(h_old);
    cudaMemcpy(d_old, h_old, size, cudaMemcpyHostToDevice);

    for (size_t i = 0; i < epoch; i++)
    {
        updateGPU<<<grid, block>>>(d_old, d_new);
        cudaDeviceSynchronize();
        
        copyGPU<<<grid, block>>>(d_old, d_new);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_old, d_old, size, cudaMemcpyDeviceToHost);

    print1DArray(h_old);

    free(h_old);
    cudaFree(d_old);
    cudaFree(d_new);
    // reset device
    cudaDeviceReset();
}

void serialCode()
{
    const int n = N;
    double u_old[n + 2][n + 2];
    double u_new[n + 2][n + 2];
   

    for (size_t i = 0; i <= n + 1; i++)
    {
        for (size_t j = 0; j <= n + 1; j++)
        {
            u_old[i][j] = 25;
        }
    }
    for (size_t j = 0; j <= n + 1; j++)
    {
        u_old[j][N+1] = 100;
    }

    int time = 0;
    while (time < epoch)
    {
        for (size_t i = 1; i <= n; i++)
        {
            for (size_t j = 1; j <= n; j++)
            {
                u_new[i][j] = u_old[i][j] + c * delta_t /
                                                (delta_s * delta_s) *
                                                (u_old[i + 1][j] + u_old[i - 1][j] - 4 * u_old[i][j] +
                                                 u_old[i][j + 1] + u_old[i][j - 1]);
            }
        }
        for (size_t i = 1; i < n + 1; i++)
        {
            for (size_t j = 1; j < n + 1; j++)
            {
                u_old[i][j] = u_new[i][j];
            }
        }
        time++;
    }

    for (size_t i = 0; i <= n + 1; i++)
    {
        for (size_t j = 0; j <= n + 1; j++)
        {
            printf("%8.1f", u_old[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    printf("serial code\n");
    serialCode();
    printf("\n");
    printf("parallel code\n");
    parallelCode();
    return 0;
}