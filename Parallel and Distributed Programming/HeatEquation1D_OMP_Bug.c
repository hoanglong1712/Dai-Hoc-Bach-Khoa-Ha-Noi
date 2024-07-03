#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define  M       20
#define  Time    1
#define  dt      0.01
#define  dx      0.1
#define  D       0.1
//=========================
void DisplayArray(float *T, int size)
{
  int i;
  for(i=0;i<size;i++)
    printf("  %.2f",*(T+i));
  printf("\n");  
}
//=========================
void InputData(float *T)
{
  int i,j;
  for (  i = 0 ; i < M ; i++ )
      *(T+i) = 25.0;
}
//=========================
void Derivative(float *T, float *dT, int start, int stop)
{
  int i;
  float c,l,r;
  for (  i = start ; i < stop ; i++ )
    {
      c = *(T+i);
      l = (i==0)   ? 100.0 : *(T+(i-1));
      r = (i==M-1) ? 25.0  : *(T+(i+1));
      *(dT+i) = (r-2*c+l)/(dx*dx);
    }
}
//=========================
int main()
{
  int i,t, Ntime;
  float *T,*dT;
  T  = (float *) malloc ((M)*sizeof(float));
  dT = (float *) malloc ((M)*sizeof(float));
  InputData(T);
  printf("Input Data:\n");
  DisplayArray(T,M);
//
  int NT, ID, Mc, start, stop;
  omp_set_num_threads(4);  
  #pragma omp parallel private(ID,start,stop,i,t)
  {
    ID = omp_get_thread_num();
    NT = omp_get_num_threads();
    Mc = M/NT;
    start = ID*Mc;
    stop  = start + Mc;
    Ntime = Time/dt;
    for (t=0;t<Ntime;t++)
    {
       Derivative(T, dT, start, stop);
      #pragma omp barrier
      //#pragma omp for
       for (i = start;i < stop;i++ )
          *(T+i) = *(T+i) + D*dt*(*(dT+i));
      #pragma omp barrier
    }
  }
//
  printf("Result of OMP:\n");
  DisplayArray(T, M);
return 0;
}

