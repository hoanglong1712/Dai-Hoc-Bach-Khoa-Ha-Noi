#include <stdio.h>
#include <malloc.h>
// #include <conio.h>

#define M 20
#define Time 10
#define dt 0.01
#define dx 0.1
#define D 0.1
//=========================
void DisplayArray(float *T, int size)
{
  int i;
  for (i = 0; i < size; i++)
    printf("  %.2f", *(T + i));
  printf("\n");
}
//=========================
void Write2File(float *T, int size)
{
  FILE *result = fopen("result.txt", "a");
  int i;

  for (i = 0; i < size; i++)
  {
    fprintf(result, "%lf", *(T + i));
    fprintf(result, "\n");
  }

  fclose(result);
}
//=========================
void InputData(float *T)
{
  int i;
  for (i = 0; i < M; i++)
    *(T + i) = 25.0;
}
//=========================
void Derivative(float *T, float *dT, int Dau, int Cuoi)
{
  int i;
  float c, l, r;
  for (i = Dau; i < Cuoi; i++)
  {
    c = *(T + i);
    l = (i == 0) ? 100.0 : *(T + (i - 1));
    r = (i == M - 1) ? 25.0 : *(T + (i + 1));
    *(dT + i) = D * (r - 2 * c + l) / (dx * dx);
  }
}
//=========================
int main()
{
  int i, t, Ntime;
  float *T, *dT;
  T = (float *)malloc((M) * sizeof(float));
  dT = (float *)malloc((M) * sizeof(float));
  InputData(T);
  printf("Input Data:\n");
  DisplayArray(T, M);
  Write2File(T, M);
  Ntime = Time / dt;
  for (t = 0; t < Ntime; t++)
  {
    Derivative(T, dT, 0, M);
    for (i = 0; i < M; i++)
      *(T + i) = *(T + i) + dt * (*(dT + i));
    // Write2File(T,M);
  }
  Write2File(T, M);
  printf("Result of C:\n");
  DisplayArray(T, M);
  // getch();
  return 0;
}
