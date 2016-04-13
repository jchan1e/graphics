/*
 *  CUDA square matrix multiplier
 *
 *  The size of the matrix is width*blocks
 *
 *  Parameters:
 *  -v      Verbose - show hardware detila
 *  width   Block width (width squared <= max threads/block)
 *  blocks  Number of blocks
 */

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <cuda.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

/*
 *  Return elapsed wall time since last call (seconds)
 */
static double t0=0;
double Elapsed(void)
{
#ifdef _WIN32
   //  Windows version of wall time
   LARGE_INTEGER tv,freq;
   QueryPerformanceCounter((LARGE_INTEGER*)&tv);
   QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
   double t = tv.QuadPart/(double)freq.QuadPart;
#else
   //  Unix/Linux/OSX version of wall time
   struct timeval tv;
   gettimeofday(&tv,NULL);
   double t = tv.tv_sec+1e-6*tv.tv_usec;
#endif
   double s = t-t0;
   t0 = t;
   return s;
}

/*
 *  Print message to stderr and exit
 */
void Fatal(const char* format , ...)
{
   va_list args;
   va_start(args,format);
   vfprintf(stderr,format,args);
   va_end(args);
   exit(1);
}

/*
 *  Initialize matrix with random values
 */
void RandomInit(float x[],const unsigned int n)
{
   for (unsigned int i=0;i<n*n;i++)
      x[i] = rand() / (float)RAND_MAX;
}

/*
 *  Initialize fastest GPU device
 */
int InitGPU(int verbose)
{
   //  Get number of CUDA devices
   int num;
   if (cudaGetDeviceCount(&num)) Fatal("Cannot get number of CUDA devices\n");
   if (num<1) Fatal("No CUDA devices found\n");

   //  Get fastest device
   cudaDeviceProp prop;
   int   MaxDevice = -1;
   int   MaxGflops = -1;
   for (int dev=0;dev<num;dev++)
   {
      if (cudaGetDeviceProperties(&prop,dev)) Fatal("Error getting device %d properties\n",dev);
      int Gflops = prop.multiProcessorCount * prop.clockRate;
      if (verbose) printf("CUDA Device %d: %s Gflops %f Processors %d Threads/Block %d\n",dev,prop.name,1e-6*Gflops,prop.multiProcessorCount,prop.maxThreadsPerBlock);
      if(Gflops > MaxGflops)
      {
         MaxGflops = Gflops;
         MaxDevice = dev;
      }
   }

   //  Print and set device
   if (cudaGetDeviceProperties(&prop,MaxDevice)) Fatal("Error getting device %d properties\n",MaxDevice);
   printf("Fastest CUDA Device %d: %s\n",MaxDevice,prop.name);
   cudaSetDevice(MaxDevice);

   //  Return max thread count
   return prop.maxThreadsPerBlock;
}

/*
 * C = A * B -- host
 */
void AxBh(float C[], const float A[], const float B[], unsigned int n)
{
   for (unsigned int i=0;i<n;i++)
      for (unsigned int j=0;j<n;j++)
      {
         double sum=0;
         for (unsigned int k=0;k<n;k++)
            sum += (double)A[i*n+k] * (double)B[k*n+j];
         C[i*n+j] = (float)sum;
      }
}

/*
 * Compute one element of A * B
 */
__global__ void AxB(float C[],const float A[],const float B[],const unsigned int n)
{
   unsigned int j = blockIdx.x*blockDim.x+threadIdx.x;
   unsigned int i = blockIdx.y*blockDim.y+threadIdx.y;
   float sum =0;
   for (int k=0;k<n;k++)
      sum += A[i*n+k] * B[k*n+j];
   C[i*n+j] = sum;
}

/*
 * C = A * B -- device
 */
void AxBd(float Ch[],const float Ah[],const float Bh[],const unsigned int Bw,const unsigned int Bn)
{
   //  Calculate matrix dimensions
   int n = Bw*Bn;
   int N = n*n*sizeof(float);

   // Allocate device memory
   float* Ad;
   float* Bd;
   float* Cd;
   if (cudaMalloc((void**)&Ad,N)) Fatal("Cannot allocate device memory Ad\n");
   if (cudaMalloc((void**)&Bd,N)) Fatal("Cannot allocate device memory Bd\n");
   if (cudaMalloc((void**)&Cd,N)) Fatal("Cannot allocate device memory Cd\n");

   // Copy A and B from host to device
   if (cudaMemcpy(Ad,Ah,N,cudaMemcpyHostToDevice)) Fatal("Cannot copy A from host to device\n");
   if (cudaMemcpy(Bd,Bh,N,cudaMemcpyHostToDevice)) Fatal("Cannot copy B from host to device\n");

   // Set size of block to Bw x Bw, and Bn x Bn blocks
   dim3 threads(Bw,Bw);
   dim3 grid(Bn,Bn);
   // Execute the kernel
   AxB<<<grid,threads>>>(Cd,Ad,Bd,n);
   if (cudaGetLastError()) Fatal("AxB failed\n");

   // Copy C from device to host
   if (cudaMemcpy(Ch,Cd,N,cudaMemcpyDeviceToHost)) Fatal("Cannot copy C from device to host\n");

   //  Free device memory
   cudaFree(Ad);
   cudaFree(Bd);
   cudaFree(Cd);
}

/*
 *  main
 */
int main(int argc, char* argv[])
{
   //  Process options
   int opt;
   int verbose=0;
   while ((opt=getopt(argc,argv,"v"))!=-1)
   {
      if (opt=='v')
         verbose++;
      else
         Fatal("Usage: [-v] <block width> <number of blocks>\n");
   }
   argc -= optind;
   argv += optind;
 
   //  Get width and number of blocks
   if (argc!=2) Fatal("Usage: [-v] <block width> <number of blocks>\n");
   int Bw = atoi(argv[0]);
   if (Bw<1) Fatal("Block width out of range %d\n",Bw);
   int Bn = atoi(argv[1]);
   if (Bn<1) Fatal("Number of blocks out of range %d\n",Bn);
   //  Total width is block times number of blocks
   int n = Bw*Bn;
   int N = n*n*sizeof(float);
   printf("Bw=%d Bn=%d n=%d\n",Bw,Bn,n);

   //  Initialize GPU
   int Mw = InitGPU(verbose);
   if (Mw<Bw*Bw) Fatal("Thread count %d exceeds threads per block of %d\n",Bw*Bw,Mw);

   // Allocate host matrices A/B/C/R
   float* Ah = (float*)malloc(N);
   float* Bh = (float*)malloc(N);
   float* Ch = (float*)malloc(N);
   float* Rh = (float*)malloc(N);
   if (!Ah || !Bh || !Ch || !Rh) Fatal("Cannot allocate host memory\n");

   // Initialize A & B
   srand(9999);
   RandomInit(Ah,n);
   RandomInit(Bh,n);

   //  Compute R = AB on host
   Elapsed();
   AxBh(Rh,Ah,Bh,n);
   double Th = Elapsed();

   //  Compute C = AB on device
   Elapsed();
   AxBd(Ch,Ah,Bh,Bw,Bn);
   double Td = Elapsed();

   //  Compute difference between R and C
   double r2=0;
   for (int i=0;i<n*n;i++)
      r2 += fabs(Ch[i]-Rh[i]);
   r2 /= n*n;

   //  Free host memory
   free(Ah);
   free(Bh);
   free(Ch);
   free(Rh);

   //  Print results
   printf("Host   Time = %6.3f s\n",Th);
   printf("Device Time = %6.3f s\n",Td);
   printf("Speedup = %.1f\n",Th/Td);
   printf("Difference = %.2e\n",r2);

   //  Done
   return 0;
}
