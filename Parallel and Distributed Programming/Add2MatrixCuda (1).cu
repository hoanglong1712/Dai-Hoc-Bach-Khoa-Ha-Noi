#include <stdio.h>
#include <malloc.h>
#include <cuda.h>
#define M 12
#define N 12
#define GridSizeX 2
#define GridSizeY 3
#define BlockSizeX 3
#define BlockSizeY 2
#define ThreadSizeX M/(GridSizeX*BlockSizeX)
#define ThreadSizeY N/(GridSizeY*BlockSizeY)
// Kernel code:
__global__ void Add2Array(int *a, int *b, int *c)
{	
	int indexX,indexY,i,j,startX,startY,stopX,stopY;
	indexX = blockIdx.x * blockDim.x + threadIdx.x;
	indexY = blockIdx.y * blockDim.y + threadIdx.y;
  startX = indexX*ThreadSizeX;
	stopX  = startX + ThreadSizeX;
	startY = indexY*ThreadSizeY;
	stopY  = startY + ThreadSizeY;
	for (i=startX; i<stopX;i++)
	  for (j=startY; j<stopY;j++)
	    *(c+i*N+j) = *(a+i*N+j) + *(b+i*N+j);
}

int main(){
// Host code
//1a. Delare and Allocate Mem on CPU
int *Acpu, *Bcpu, *Ccpu,i,j;
Acpu = (int *) malloc (M*N*sizeof(int));
Bcpu = (int *) malloc (M*N*sizeof(int));
Ccpu = (int *) malloc (M*N*sizeof(int));
//1b. Delare and Allocate Mem on GPU
int *Agpu, *Bgpu, *Cgpu;
cudaMalloc((void**)&Agpu,M*N*sizeof(int));
cudaMalloc((void**)&Bgpu,M*N*sizeof(int));
cudaMalloc((void**)&Cgpu,M*N*sizeof(int));
//1c. Input data on CPU
for (i=0;i<M;i++) 
	for (j=0;j<N;j++) { 
		*(Acpu+i*N+j) = 3*(i*N+j); 
		*(Bcpu+i*N+j) = 2*(i*N+j);
	}
//2. Copy Input from CPU to GPU
cudaMemcpy(Agpu,Acpu,M*N*sizeof(int),cudaMemcpyHostToDevice);
cudaMemcpy(Bgpu,Bcpu,M*N*sizeof(int),cudaMemcpyHostToDevice);
//3. Define Block and Thread Structure
dim3 dimGrid(GridSizeX,GridSizeY);
dim3 dimBlock(BlockSizeX,BlockSizeY);
//4. Invoke Kernel
Add2Array<<<dimGrid,dimBlock>>>(Agpu,Bgpu,Cgpu);
//5. Copy Output from GPU to CPU
cudaMemcpy(Ccpu,Cgpu,M*N*sizeof(int),cudaMemcpyDeviceToHost);
printf("C:");
for (i=0;i<M;i++) {
	for (j=0;j<N;j++)
	   printf("%d  ",*(Ccpu+i*N+j));
		 printf("\n");
}
//6. Free Mem on CPU and GPU
free(Acpu);free(Bcpu);free(Ccpu);
cudaFree(Agpu);cudaFree(Bgpu);cudaFree(Cgpu);

return 0;
}
