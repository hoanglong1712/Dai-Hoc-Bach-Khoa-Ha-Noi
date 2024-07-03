#include <stdio.h>
#include <mpi.h>
#include <malloc.h>
int main(int argc, char *argv[]){
  int N = 12,NP,Rank;
  MPI_Init (&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD,&NP);
  MPI_Comm_rank(MPI_COMM_WORLD,&Rank);
  MPI_Status Warning;
  int *A, i; 
  A 	= (int *) malloc (N*sizeof(int));
  // Innitialize Input Data
  if (Rank==0) {
    for (i=0;i<N;i++) *(A+i) = i; 
    printf("Input array:");
    for (i=0;i<N;i++) printf("%d ",*(A+i));
    printf("\n");
  }
  // Domain Decomposition
  int Ns = N/NP;
  int *A_s,*Sum_s;
  A_s 	= (int *) malloc (Ns*sizeof(int));
  Sum_s = (int *) malloc (Ns*sizeof(int));
  // Distribute Input Data
  MPI_Scatter(A,Ns,MPI_INT,A_s,Ns,MPI_INT,0,MPI_COMM_WORLD);
  // Do reduction
  MPI_Reduce(A_s,Sum_s,Ns,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
  // Print result
  if (Rank==0){
     printf("Reduction at Rank 0:");
     for (i=0;i<Ns;i++) printf("%d ",*(Sum_s+i));
     printf("\n");
     int Global_Sum = 0;
     for (i=0;i<Ns;i++) Global_Sum += *(Sum_s+i);
     printf("Global Sum of A: %d\n", Global_Sum);
  }
  MPI_Finalize();
  return 0;
}

