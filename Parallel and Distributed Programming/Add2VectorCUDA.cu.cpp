#include <stdio.h>
#include <malloc.h>
#include <cuda.h>
#define N 16
// Kernel code:
__global__ void Add2Array(int *a, int *b, int *c)
{	int index;
	index = blockIdx.x * blockDim.x + threadIdx.x;
	*(c + index) = *(a + index) + *(b + index);
}

int main(){
// Host code
//1a. Delare and Allocate Mem on CPU
	int *Acpu, *Bcpu, *Ccpu,i;
	Acpu = (int *) malloc (N*sizeof(int));
	Bcpu = (int *) malloc (N*sizeof(int));
	Ccpu = (int *) malloc (N*sizeof(int));
//1b. Delare and Allocate Mem on GPU
	int *Agpu, *Bgpu, *Cgpu;
	cudaMalloc((void**)&Agpu,N*sizeof(int));
	cudaMalloc((void**)&Bgpu,N*sizeof(int));
	cudaMalloc((void**)&Cgpu,N*sizeof(int));
//1c. Input data on CPU
	for (i=0;i<N;i++){ *(Acpu+i) = 3*i; *(Bcpu+i) = 2*i;}
//2. Copy Input from CPU to GPU
	cudaMemcpy(Agpu,Acpu,N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(Bgpu,Bcpu,N*sizeof(int),cudaMemcpyHostToDevice);
//3. Define Block and Thread Structure
	dim3 dimGrid(1);
	dim3 dimBlock(16);
//4. Invoke Kernel
	Add2Array<<<dimGrid,dimBlock>>>(Agpu,Bgpu,Cgpu);
//5. Copy Output from GPU to CPU
	cudaMemcpy(Ccpu,Cgpu,N*sizeof(int),cudaMemcpyDeviceToHost);
	printf("C:");
	for (i=0;i<N;i++) printf("%d  ",*(Ccpu+i));
//6. Free Mem on CPU and GPU
	free(Acpu);free(Bcpu);free(Ccpu);
	cudaFree(Agpu);cudaFree(Bgpu);cudaFree(Cgpu);

return 0;
}
