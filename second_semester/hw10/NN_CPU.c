#include "reader_.c"
#include <math.h>
#include <sys/time.h>
#include <string.h>

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

//void err(cudaError_t returnVal)
//{
//   if (returnVal != cudaSuccess)
//   {
//      fprintf(stderr, "CUDA Failure: %s\n", cudaGetErrorString(returnVal));
//      exit(EXIT_FAILURE); //   }
//}

float sigmoid(float x)
{
   return 1.0/(1.0+expf(-x));
}

float dsigmoid(float x)
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

float dtanh(float x)
{
   return 1.0 - x*x;
}

void Tanh(int I, float* layer1)
{
   //int i = blockDim.x*blockIdx.x + threadIdx.x; //Data.count * 256

   for (int i=0; i < I; ++i)
      layer1[i] = tanh(layer1[i]);
}
void Sigmoid(int I, float* layer1)
{
   //int i = blockDim.x*blockIdx.x + threadIdx.x; //Data.count * 256

   for (int i=0; i < I; ++i)
      layer1[i] = sigmoid(layer1[i]);
}
void Fprop1(int K, int J, int I, const float* in, const float* syn1, float* layer1)
{
   //int i = threadIdx.x;                         //256
   //int j = blockDim.y*blockIdx.y + threadIdx.y; //28*28
   //int k = blockIdx.x;                          //Data.count

   for (int k=0; k < K; ++k)
   {
      for (int i=0; i < I; ++i)
      {
         for (int j=0; j < J; ++j)
            layer1[k*256 + i] += in[k*28*28 + j] * syn1[j*256 + i];
      }
   }
}
void Fprop2(int J, int I, int K, const float* layer1, const float* syn2, float* out)
{
   //int i = blockDim.y*blockIdx.y + threadIdx.y; //10
   //int j = blockIdx.x;  //Data.count
   //int k = threadIdx.x; //256

   for (int j=0; j < J; ++j)
   {
      for (int i=0; i < I; ++i)
      {
         for (int k=0; k < K; ++k)
            out[j*10 + i] += layer1[j*256 + k] * syn2[k*10 + i];
      }
   }
}
void Ecalc2(int I, float* out, const float* label)
{
   //int i = blockDim.x*blockIdx.x + threadIdx.x; //10 * Data.count

   for (int i=0; i < I; ++i)
      out[i] = label[i] - out[i];
}
void Dcalc2(int I, float* out, const float* label)
{
   //int i = blockDim.x*blockIdx.x + threadIdx.x; //10 * Data.count

   for (int i=0; i < I; ++i)
   {
      float x = label[i] - out[i];
      out[i] = x * dsigmoid(x);
   }
}
void Bprop2(int K, int J, int I, const float* layer1, float* dsyn2, const float* out, const float alpha)
{
   //int i = threadIdx.x; //256
   //int j = blockDim.y*blockIdx.y + threadIdx.y; //10
   //int k = blockIdx.x;  //Data.count


   for (int k=0; k < K; ++k)
   {
      for (int i=0; i < I; ++i)
      {
         for (int j=0; j < J; ++j)
            dsyn2[i*10 + j] += out[k*10 + j] * layer1[256*k + i] * alpha;
      }
   }
}
void Dcalc1(int J, int I, float* dlayer1, const float* syn2, const float* out)
{
   //int i = threadIdx.x; //256
   //int j = blockIdx.x;  //Data.count


   for (int j=0; j < J; ++j)
   {
      for (int i=0; i < I; ++i)
      {
         float x = 0.0;
         for (int k=0; k < 10; ++k)
            x += out[j*10 + k] * syn2[i*10 + k];
         dlayer1[j*256 + i] = x * dtanh(x);
      }
   }
}
void Bprop1(int K, int I, int J, const float* in, float* dsyn1, const float* dlayer1, const float alpha)
{
   //int i = blockDim.y*blockIdx.y + threadIdx.y; //28*28
   //int j = threadIdx.x;                         //256
   //int k = blockIdx.x;                          //Data.count


   for (int k=0; k < K; ++k)
   {
      for (int i=0; i < I; ++i)
      {
         for (int j=0; j < J; ++j)
            dsyn1[i*256 + j] += dlayer1[k*256 + j] * in[k*28*28 + i] * alpha;
      }
   }
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

   float* d_in     = malloc(28*28*Data.count*sizeof(float));
   float* d_layer1 = malloc(256*Data.count*sizeof(float));
   float* d_out    = malloc(10*Data.count*sizeof(float));
   float* d_label  = malloc(10*Data.count*sizeof(float));
   float* d_syn1   = malloc(28*28*256*sizeof(float));
   float* d_dsyn1  = malloc(28*28*256*sizeof(float));
   float* d_syn2   = malloc(256*10*sizeof(float));
   float* d_dsyn2  = malloc(256*10*sizeof(float));

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


   memcpy(d_in,    Data.Image, 28*28*Data.count*sizeof(float) );
   memcpy(d_label, Data.Label, 10*Data.count*sizeof(float)    );
   memcpy(d_syn1,  weights1,   28*28*256*sizeof(float)        );
   memcpy(d_dsyn1, weights1,   28*28*256*sizeof(float)        );
   memcpy(d_syn2,  weights2,   10*256*sizeof(float)           );
   memcpy(d_dsyn2, weights2,   10*256*sizeof(float)           );

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
      memset(d_layer1, 0.0, Data.count*256*sizeof(float));
      memset(d_out,    0.0, Data.count* 10*sizeof(float));
      Fprop1  (Data.count, 28*28, 256, d_in, d_syn1, d_layer1);
      Tanh    (Data.count * 256,        d_layer1);
      Fprop2  (Data.count, 10,    256, d_layer1, d_syn2, d_out);
      Sigmoid (Data.count * 10,         d_out);
      Dcalc2  (Data.count * 10,         d_out, d_label);
      Bprop2  (Data.count, 10,    256, d_layer1, d_dsyn2, d_out, alpha/Data.count/10);
      Dcalc1  (Data.count, 256,        d_layer1, d_syn2, d_out);
      Bprop1  (Data.count, 28*28, 256, d_in, d_dsyn1, d_layer1, alpha/Data.count/10);
      memcpy(weights2, d_dsyn2,       10*256*sizeof(float));
      memcpy(weights1, d_dsyn1,    28*28*256*sizeof(float));
      memcpy(d_syn2,  weights2,       10*256*sizeof(float));
      memcpy(d_syn1,  weights1,    28*28*256*sizeof(float));
   }

   //clock_t end_time = clock();
   //double training_time = (double)(end_time - start_time)/CLOCKS_PER_SEC;
   double training_time = Elapsed();
   printf("training time: %f\n", training_time);

   free(d_in);
   free(d_label);
   free(d_layer1);
   free(d_out);

   ////////////////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////////////////

   //test
   printf("testing\n");
   double error = 0.0;

   d_in     = malloc(28*28*Test.count*sizeof(float));
   d_layer1 = malloc(256*Test.count*sizeof(float));
   d_out    = malloc(10*Test.count*sizeof(float));
   d_label  = malloc(10*Test.count*sizeof(float));

   memcpy(d_in,    Test.Image, 28*28*Test.count*sizeof(float));
   memcpy(d_label, Test.Label, 10*Test.count*sizeof(float));

   memset(d_layer1, 0.0, Test.count*256*sizeof(float));
   memset(d_out,    0.0, Test.count* 10*sizeof(float));

      Fprop1  (Test.count, 28*28, 256, d_in, d_syn1, d_layer1);
      Tanh    (Test.count * 256,        d_layer1);
      Fprop2  (Test.count, 10,    256, d_layer1, d_syn2, d_out);
      Sigmoid (Test.count * 10,         d_out);
      Ecalc2  (Test.count * 10,         d_out, d_label);

   
   float* out = (float*)malloc(Test.count*10*sizeof(float));
   memcpy(out, d_out, Test.count*10*sizeof(float));

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

   free(d_in);
   free(d_layer1);
   free(d_out);
   free(d_label);
   free(d_syn1);
   free(d_syn2);
   free(d_dsyn1);
   free(d_dsyn2);

   return EXIT_SUCCESS;
}
