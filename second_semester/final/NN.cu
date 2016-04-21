#include "reader_.c"
#include <sys/time.h>

double t0 = 0;
double Elapsed()
{
   struct timeval tv;
   gettimeofday(&tv, NULL);
   double t = tv.tv_sec+1e-6*tv.tv_usec;
   double s = t - t0;
   t0 = t;
   return s;
}

void err(cudaError_t returnVal)
{
   if (returnVal != cudaSuccess)
   {
      fprintf(stderr, "CUDA Failure: %s\n", cudaGetErrorString(returnVal));
      exit(EXIT_FAILURE);
   }
}

__device__ float sigmoid(float x)
{
   return 1.0/(1.0+expf(-x));
}

__device__ float dsigmoid(float x)
{
   return x*(1.0-x);
}

//__device__ float tanh_(float x)
//{
//   // e**2x - 1
//   // ---------
//   // e**2x + 1
//   float exp2x =    expf(2.0*x);
//   return (exp2x - 1.0)/(exp2x + 1.0);
//}

__device__ float dtanh(float x)
{
   return 1.0 - x*x;
}

__global__ void Tanh(float* layer1)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x; //Data.count * 256
   layer1[i] = tanh(layer1[i]);
}
__global__ void Sigmoid(float* layer1)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x; //Data.count * 256
   layer1[i] = sigmoid(layer1[i]);
}
__global__ void Fprop1(const float* in, const float* syn1, float* layer1)
{
   int i = threadIdx.x;                         //256
   //int j = blockDim.y*blockIdx.y + threadIdx.y; //28*28
   int k = blockIdx.x;                          //Data.count
   float x = 0.0;
   for (int j=0; j < 28*28; ++j)
      x += in[k*28*28 + j] * syn1[j*256 + i];
   layer1[k*256 + i] = x;
}
__global__ void Fprop2(const float* layer1, const float* syn2, float* out)
{
   int i = blockDim.y*blockIdx.y + threadIdx.y; //10
   int j = blockIdx.x;  //Data.count
   //int k = threadIdx.x; //256
   float x = 0.0;
   for (int k=0; k < 256; ++k)
      x += layer1[j*256 + k] * syn2[k*10 + i];
   out[j*10 + i] = x;
}
__global__ void Ecalc2(float* out, const float* label)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x; //10 * Data.count
   out[i] = label[i] - out[i];
}
__global__ void Dcalc2(float* out, const float* label)
{
   int i = blockDim.x*blockIdx.x + threadIdx.x; //10 * Data.count
   float x = label[i] - out[i];
   out[i] = x * dsigmoid(x);
}
__global__ void Bprop2(const float* layer1, float* dsyn2, const float* out, const float alpha)
{
   int i = threadIdx.x; //256
   int j = blockDim.y*blockIdx.y + threadIdx.y; //10
   int k = blockIdx.x;  //Data.count

   atomicAdd(&dsyn2[i*10 + j], out[k*10 + j] * layer1[256*k + i] * alpha);
}
__global__ void Dcalc1(float* dlayer1, const float* syn2, const float* out)
{
   int i = threadIdx.x; //256
   int j = blockIdx.x;  //Data.count

   float x = 0.0;
#pragma unroll
   for (int k=0; k < 10; ++k)
      x += out[j*10 + k] * syn2[i*10 + k];
   dlayer1[j*256 + i] = x * dtanh(x);
}
__global__ void Bprop1(const float* in, float* dsyn1, const float* dlayer1, const float alpha)
{
   int i = blockDim.y*blockIdx.y + threadIdx.y; //28*28
   int j = threadIdx.x;                         //256
   int k = blockIdx.x;                          //Data.count

   atomicAdd(&dsyn1[i*256 + j], dlayer1[k*256 + j] * in[k*28*28 + i] * alpha);
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
   if (argc != 7)
   {
      printf("usage: run trainingImages trainingLabels testImages testLabels iterations alpha\n");
      return 2;
   }
   //printf("%d\n", argc);

   struct data Data = read_this(argv[1], argv[2]);
   struct data Test = read_this(argv[3], argv[4]);

   float weights1[28*28*256];    //input to middle layer weights
   float weights2[256*10];     //middle to output layer weights
   float alpha = atof(argv[argc-1]);

   float* d_in;     err(cudaMalloc((void**)&d_in,     28*28*Data.count*sizeof(float)));
   float* d_layer1; err(cudaMalloc((void**)&d_layer1, 256*Data.count*sizeof(float)));
   float* d_out;    err(cudaMalloc((void**)&d_out,    10*Data.count*sizeof(float)));
   float* d_label;  err(cudaMalloc((void**)&d_label,  10*Data.count*sizeof(float)));
   float* d_syn1;   err(cudaMalloc((void**)&d_syn1,   28*28*256*sizeof(float)));
   float* d_dsyn1;  err(cudaMalloc((void**)&d_dsyn1,  28*28*256*sizeof(float)));
   float* d_syn2;   err(cudaMalloc((void**)&d_syn2,   256*10*sizeof(float)));
   float* d_dsyn2;  err(cudaMalloc((void**)&d_dsyn2,  256*10*sizeof(float)));

   //Initialize weights to random values
   //printf("randomizing initial weights\n");
   srand(112992); //make the random values the same each time
   for (int j=0; j < 28*28; ++j)
   {
      for (int k=0; k < 256; ++k)
      {
         weights1 [j*256 + k] = (float)rand()/(RAND_MAX/2.0) - 1.0;
      }
   }
   for (int i=0; i < 256; ++i)
   {
      for (int j=0; j < 10; ++j)
      {
         weights2[i*10 + j] = (float)rand()/(RAND_MAX/2.0) - 1.0;
      }
   }


   err(cudaMemcpy(d_in,    Data.Image, 28*28*Data.count*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_label, Data.Label, 10*Data.count*sizeof(float),    cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_syn1,  weights1,   28*28*256*sizeof(float),        cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_dsyn1, weights1,   28*28*256*sizeof(float),        cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_syn2,  weights2,   10*256*sizeof(float),           cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_dsyn2, weights2,   10*256*sizeof(float),           cudaMemcpyHostToDevice));

   //err(cudaMemset(d_layer1, 0.0,       256*Data.count*sizeof(float)));
   //err(cudaMemset(d_out,    0.0,       10*Data.count*sizeof(float)));

   //cudaStream_t s[10];
   //cudaStreamCreate(&s[0]);
   //cudaStreamCreate(&s[1]);
   //cudaStreamCreate(&s[2]);
   //cudaStreamCreate(&s[3]);

   //train
   int iterations = atoi(argv[argc-2]);
   printf("training %d iterations\n", iterations);
   //clock_t start_time = clock();
   Elapsed();
   for (int iter=0; iter<iterations; ++iter)
   {
      err(cudaMemset(d_layer1, 0.0, Data.count*256*sizeof(float)));
      err(cudaMemset(d_out,    0.0, Data.count* 10*sizeof(float)));
      Fprop1  <<<dim3(Data.count, 1), dim3(256, 1)>>> (d_in, d_syn1, d_layer1);
      Tanh    <<<Data.count, 256>>>                       (d_layer1);
      Fprop2  <<<dim3(Data.count, 1),    dim3(1, 10)>>> (d_layer1, d_syn2, d_out);
      Sigmoid <<<Data.count, 10>>>                        (d_out);
      Dcalc2  <<<Data.count, 10>>>                        (d_out, d_label);
      Bprop2  <<<dim3(Data.count, 10),    dim3(256, 1)>>> (d_layer1, d_dsyn2, d_out, alpha/Data.count/10);
      Dcalc1  <<<Data.count, 256>>>                       (d_layer1, d_syn2, d_out);
      Bprop1  <<<dim3(Data.count, 28*28), dim3(256, 1)>>> (d_in, d_dsyn1, d_layer1, alpha/Data.count/10);
      err(cudaMemcpy(d_syn2, d_dsyn2,       10*256*sizeof(float), cudaMemcpyDeviceToDevice));
      err(cudaMemcpy(d_syn1, d_dsyn1,    28*28*256*sizeof(float), cudaMemcpyDeviceToDevice));
      //err(cudaMemcpy(d_syn2,  weights2,       10*256*sizeof(float), cudaMemcpyHostToDevice));
      //err(cudaMemcpy(d_syn1,  weights1,    28*28*256*sizeof(float), cudaMemcpyHostToDevice));
   }

   cudaDeviceSynchronize();
   //clock_t end_time = clock();
   //double training_time = (double)(end_time - start_time)/CLOCKS_PER_SEC;
   double training_time = Elapsed();
   printf("training time: %f\n", training_time);

   err(cudaFree(d_in));
   err(cudaFree(d_label));
   err(cudaFree(d_layer1));
   err(cudaFree(d_out));

   ////////////////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////////////////

   //test
   printf("testing\n");
   double error = 0.0;

   err(cudaMalloc((void**)&d_in,     28*28*Test.count*sizeof(float)));
   err(cudaMalloc((void**)&d_layer1, 256*Test.count*sizeof(float)));
   err(cudaMalloc((void**)&d_out,    10*Test.count*sizeof(float)));
   err(cudaMalloc((void**)&d_label,  10*Test.count*sizeof(float)));

   err(cudaMemcpy(d_in,    Test.Image, 28*28*Test.count*sizeof(float), cudaMemcpyHostToDevice));
   err(cudaMemcpy(d_label, Test.Label, 10*Test.count*sizeof(float),    cudaMemcpyHostToDevice));
   
   err(cudaMemset(d_layer1, 0.0, Test.count*256*sizeof(float)));
   err(cudaMemset(d_out,    0.0, Test.count* 10*sizeof(float)));

   Fprop1  <<<dim3(Test.count, 1), dim3(256, 1)>>> (d_in, d_syn1, d_layer1);
   Tanh    <<<Test.count, 256>>>                       (d_layer1);
   Fprop2  <<<dim3(Test.count, 1),    dim3(1, 10)>>> (d_layer1, d_syn2, d_out);
   Sigmoid <<<Test.count, 10>>>                        (d_out);
   Ecalc2  <<<Test.count, 10>>>                        (d_out, d_label);

   
   float* out = (float*)malloc(Test.count*10*sizeof(float));
   err(cudaMemcpy(out, d_out, Test.count*10*sizeof(float), cudaMemcpyDeviceToHost));
   cudaDeviceSynchronize();

   for (int i=0; i < Test.count*10; ++i)
   {
      error += fabs(out[i]);
      //printf("%f ", out[i]);
      //if (i%10 == 9)
      //   printf("\n");
   }
   error /= Test.count*10;

   printf("Error: %f %%\n", error*100.0);

   free(out);

   free(Data.Image);
   free(Data.Label);
   free(Test.Image);
   free(Test.Label);

   err(cudaFree(d_in));
   err(cudaFree(d_layer1));
   err(cudaFree(d_out));
   err(cudaFree(d_label));
   err(cudaFree(d_syn1));
   err(cudaFree(d_syn2));
   err(cudaFree(d_dsyn1));
   err(cudaFree(d_dsyn2));

   return EXIT_SUCCESS;
}
