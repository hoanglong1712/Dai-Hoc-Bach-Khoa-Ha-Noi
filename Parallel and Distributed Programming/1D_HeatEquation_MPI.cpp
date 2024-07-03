#include <stdio.h>
#include <malloc.h>
#include <mpi.h>
#define  M       20
#define  Ntime   100
#define  dt      0.01
#define  dx      0.1
#define  D       0.1

//=========================
void InputData(float *T)
{
  int i;
  for (  i = 0 ; i < M ; i++ )
     *(T+i) = 25.0;
}
//=========================
void Derivative2(float *Ts, float Tl, float Tr, float *dTs, int ms) {
	int i;
	float c, l, r;
	for (i = 0 ; i < ms ; i++ ) {
		c = *(Ts + i);
		l = (i == 0)       ? Tl  : *(Ts + i - 1);
		r = (i == ms - 1)  ? Tr  : *(Ts + i + 1);
		*(dTs + i) = D * (l - 2 * c + r) / (dx * dx);
	}
}
//=========================
int main(int argc, char *argv[])
{
 	int NP,Rank;
 	MPI_Init(&argc, &argv);
 	MPI_Status Stat;
 	MPI_Comm_size(MPI_COMM_WORLD,&NP);
 	MPI_Comm_rank(MPI_COMM_WORLD,&Rank);
//
	int i,t;
	float *T,*dT;
	T  = (float *) malloc ((M)*sizeof(float));
	dT = (float *) malloc ((M)*sizeof(float));
//  Step 1: Input Data at Rank 0
	if (Rank==0) InputData(T);
//  Step 2: Domain Decomposition
	int Mc;
	float *Tc,*dTc;
	Mc = M/NP;
	Tc  = (float *) malloc ((Mc)*sizeof(float));
	dTc = (float *) malloc ((Mc)*sizeof(float));
//  Step 3: Distribution Input
	MPI_Scatter(T,Mc,MPI_FLOAT,Tc,Mc,MPI_FLOAT,0,MPI_COMM_WORLD);
//	Step 4
	float Tl,Tr;
 for (t=0;t<Ntime;t++) {
//  4.1.a: Communication Tl
	if (Rank == 0){
		Tl = 100;
		MPI_Send (Tc + Mc - 1, 1, MPI_FLOAT, Rank + 1, Rank, MPI_COMM_WORLD);
	} else if (Rank == NP - 1) {
		MPI_Recv (&Tl, 1, MPI_FLOAT, Rank - 1, Rank - 1, MPI_COMM_WORLD, &Stat);
	} else {
		MPI_Send (Tc + Mc - 1, 1, MPI_FLOAT, Rank + 1, Rank, MPI_COMM_WORLD);
		MPI_Recv (&Tl, 1, MPI_FLOAT, Rank - 1, Rank - 1, MPI_COMM_WORLD, &Stat);
	}
//	4.1.b: Communication Tr
	if (Rank == NP - 1){
		Tr = 25;
		MPI_Send (Tc, 1, MPI_FLOAT, Rank - 1, Rank, MPI_COMM_WORLD);
	} else if (Rank == 0) {
		MPI_Recv (&Tr, 1, MPI_FLOAT, Rank + 1, Rank + 1, MPI_COMM_WORLD, &Stat);
	} else {
		MPI_Send (Tc, 1, MPI_FLOAT, Rank - 1, Rank, MPI_COMM_WORLD);
		MPI_Recv (&Tr, 1, MPI_FLOAT, Rank + 1, Rank + 1, MPI_COMM_WORLD, &Stat);
	}
//	4.2 : Computation
    Derivative2(Tc, Tl, Tr, dTc, Mc);
    // Update Temperature at step t+1
    for (  i = 0 ; i < Mc ; i++ )
        *(Tc+i) = *(Tc+i) + dt*(*(dTc+i));
 }
//	Step 5: Gathering Output
	MPI_Gather(Tc,Mc,MPI_FLOAT,T,Mc,MPI_FLOAT,0,MPI_COMM_WORLD);
//  Print Result
	if (Rank==0) for (  i = 0 ; i < M ; i++ ) printf("%f \n",*(T+i));
return 0;
}


