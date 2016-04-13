#include "CSCIx239.h"
#include "InitCL.h"

/*
 *
 */
void ErrCheckCL(cl_int err)
{
   switch (err)
   {
      case CL_SUCCESS:                            break;
      case CL_DEVICE_NOT_FOUND:                   Fatal("Device not found.\n");
      case CL_DEVICE_NOT_AVAILABLE:               Fatal("Device not available\n");
      case CL_COMPILER_NOT_AVAILABLE:             Fatal("Compiler not available\n");
      case CL_MEM_OBJECT_ALLOCATION_FAILURE:      Fatal("Memory object allocation failure\n");
      case CL_OUT_OF_RESOURCES:                   Fatal("Out of resources\n");
      case CL_OUT_OF_HOST_MEMORY:                 Fatal("Out of host memory\n");
      case CL_PROFILING_INFO_NOT_AVAILABLE:       Fatal("Profiling information not available\n");
      case CL_MEM_COPY_OVERLAP:                   Fatal("Memory copy overlap\n");
      case CL_IMAGE_FORMAT_MISMATCH:              Fatal("Image format mismatch\n");
      case CL_IMAGE_FORMAT_NOT_SUPPORTED:         Fatal("Image format not supported\n");
      case CL_BUILD_PROGRAM_FAILURE:              Fatal("Program build failure\n");
      case CL_MAP_FAILURE:                        Fatal("Map failure\n");
      case CL_INVALID_VALUE:                      Fatal("Invalid value\n");
      case CL_INVALID_DEVICE_TYPE:                Fatal("Invalid device type\n");
      case CL_INVALID_PLATFORM:                   Fatal("Invalid platform\n");
      case CL_INVALID_DEVICE:                     Fatal("Invalid device\n");
      case CL_INVALID_CONTEXT:                    Fatal("Invalid context\n");
      case CL_INVALID_QUEUE_PROPERTIES:           Fatal("Invalid queue properties\n");
      case CL_INVALID_COMMAND_QUEUE:              Fatal("Invalid command queue\n");
      case CL_INVALID_HOST_PTR:                   Fatal("Invalid host pointer\n");
      case CL_INVALID_MEM_OBJECT:                 Fatal("Invalid memory object\n");
      case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    Fatal("Invalid image format descriptor\n");
      case CL_INVALID_IMAGE_SIZE:                 Fatal("Invalid image size\n");
      case CL_INVALID_SAMPLER:                    Fatal("Invalid sampler\n");
      case CL_INVALID_BINARY:                     Fatal("Invalid binary\n");
      case CL_INVALID_BUILD_OPTIONS:              Fatal("Invalid build options\n");
      case CL_INVALID_PROGRAM:                    Fatal("Invalid program\n");
      case CL_INVALID_PROGRAM_EXECUTABLE:         Fatal("Invalid program executable\n");
      case CL_INVALID_KERNEL_NAME:                Fatal("Invalid kernel name\n");
      case CL_INVALID_KERNEL_DEFINITION:          Fatal("Invalid kernel definition\n");
      case CL_INVALID_KERNEL:                     Fatal("Invalid kernel\n");
      case CL_INVALID_ARG_INDEX:                  Fatal("Invalid argument index\n");
      case CL_INVALID_ARG_VALUE:                  Fatal("Invalid argument value\n");
      case CL_INVALID_ARG_SIZE:                   Fatal("Invalid argument size\n");
      case CL_INVALID_KERNEL_ARGS:                Fatal("Invalid kernel arguments\n");
      case CL_INVALID_WORK_DIMENSION:             Fatal("Invalid work dimension\n");
      case CL_INVALID_WORK_GROUP_SIZE:            Fatal("Invalid work group size\n");
      case CL_INVALID_WORK_ITEM_SIZE:             Fatal("Invalid work item size\n");
      case CL_INVALID_GLOBAL_OFFSET:              Fatal("Invalid global offset\n");
      case CL_INVALID_EVENT_WAIT_LIST:            Fatal("Invalid event wait list\n");
      case CL_INVALID_EVENT:                      Fatal("Invalid event\n");
      case CL_INVALID_OPERATION:                  Fatal("Invalid operation\n");
      case CL_INVALID_GL_OBJECT:                  Fatal("Invalid OpenGL object\n");
      case CL_INVALID_BUFFER_SIZE:                Fatal("Invalid buffer size\n");
      case CL_INVALID_MIP_LEVEL:                  Fatal("Invalid mip-map level\n");
      default: Fatal("Unknown OpenCL Error\n");
   }
}

/*
 *  OpenCL notify callback (echo to stderr)
 */
static void Notify(const char* errinfo,const void* private_info,size_t cb,void* user_data)
{
   fprintf(stderr,"%s\n",errinfo);
}

/*
 *  Initialize fastest OpenCL device
 */
int InitGPU(int verbose,cl_device_id& devid,cl_context& context,cl_command_queue& queue)
{
   cl_uint Nplat;
   cl_int  err;
   char name[1024];
   int  MaxGflops = -1;

   //  Get platforms
   cl_platform_id platforms[1024];
   int ios = clGetPlatformIDs(1024,platforms,&Nplat);
   if (ios)
      Fatal("Cannot get number of OpenCL platforms %d\n",ios);
   else if (Nplat<1)
      Fatal("No OpenCL platforms found\n");
   //  Loop over platforms
   for (unsigned int platform=0;platform<Nplat;platform++)
   {
      if (clGetPlatformInfo(platforms[platform],CL_PLATFORM_NAME,sizeof(name),name,NULL)) Fatal("Cannot get OpenCL platform name\n");
      if (verbose) printf("OpenCL Platform %d: %s\n",platform,name);

      //  Get GPU device IDs
      cl_uint Ndev;
      cl_device_id id[1024];
      if (clGetDeviceIDs(platforms[platform],CL_DEVICE_TYPE_GPU,1024,id,&Ndev))
         Fatal("Cannot get number of OpenCL devices\n");
      else if (Ndev<1)
         Fatal("No OpenCL devices found\n");

      //  Find the fastest device
      for (unsigned int dev=0;dev<Ndev;dev++)
      {
         cl_uint proc,freq;
         if (clGetDeviceInfo(id[dev],CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(proc),&proc,NULL)) Fatal("Cannot get OpenCL device units\n");
         if (clGetDeviceInfo(id[dev],CL_DEVICE_MAX_CLOCK_FREQUENCY,sizeof(freq),&freq,NULL)) Fatal("Cannot get OpenCL device frequency\n");
         if (clGetDeviceInfo(id[dev],CL_DEVICE_NAME,sizeof(name),name, NULL)) Fatal("Cannot get OpenCL device name\n");
         int Gflops = proc*freq;
         if (verbose) printf("OpenCL Device %d: %s Gflops %f\n",dev,name,1e-3*Gflops);
         if(Gflops > MaxGflops)
         {
            devid = id[dev];
            MaxGflops = Gflops;
         }
      }
   }

   //  Print fastest device info
   if (clGetDeviceInfo(devid,CL_DEVICE_NAME,sizeof(name),name,NULL)) Fatal("Cannot get OpenCL device name\n");
   printf("Fastest OpenCL Device: %s\n",name);

   //  Check thread count
   size_t mwgs;
   if (clGetDeviceInfo(devid,CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(mwgs),&mwgs,NULL)) Fatal("Cannot get OpenCL max work group size\n");

   //  Create OpenCL context for fastest device
   context = clCreateContext(0,1,&devid,Notify,NULL,&err);
   if(!context || err) Fatal("Cannot create OpenCL context\n");

   //  Create OpenCL command queue for fastest device
   queue = clCreateCommandQueue(context,devid,0,&err);
   if(!queue || err) Fatal("Cannot create OpenCL command cue\n");

   return mwgs;
} 
