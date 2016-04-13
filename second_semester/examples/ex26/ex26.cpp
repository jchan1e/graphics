/*
 *  nBody Simulator
 *  This program requires OpenGL 3.2 or above
 *
 *  Demonstrates using OpenMP and OpenCL to accelerate computations and
 *  displaying the result in OpenGL
 *
 *  Key bindings:
 *  m/M        Cycle through modes
 *  a          Toggle axes
 *  arrows     Change view angle
 *  PgDn/PgUp  Zoom in and out
 *  0          Reset view angle
 *  ESC        Exit
 */
#include "CSCIx239.h"
#include "InitCL.h"
int    N=8192;   //  Number of bodies
size_t nloc=1;   //  Number of local work threads
int    src=0;    //  Offset of star positions
int    axes=0;   //  Display axes
int    th=0;     //  Azimuth of view angle
int    ph=0;     //  Elevation of view angle
double dim=10;   //  Size of universe
double spd=1;    //  Relative speed
double mass=2;   //  Relative mass
int    mode=0;   //  Solver mode
int    shader=0; //  Shader
const char* text[] = {"Sequential","OpenMP","OpenCL"};

//  RGB Color class
class Color
{
   public:
      float r,g,b;
      Color()
      {
         r = g = b = 0;
      }
      Color(float R,float G,float B)
      {
         r = R;
         g = G;
         b = B;
      }
};
//  float3 class
//  Killer fact:  OpenCL doesn't have a real float3, it is an alias for float4
//  So the C++ float3 MUST pad out with a dummy w value
class float3
{
   public:
      float x,y,z,w;
      float3(void)
      {
         x = y = z = w = 0;
      }
      float3(float X,float Y,float Z)
      {
         x = X;
         y = Y;
         z = Z;
         w = 0;
      }
   inline float3& operator+=(const float3& v) {x+=v.x;y+=v.y;z+=v.z; return *this;}
};
//  float3 operators
inline float3  operator+(const float3& v1  , const float3& v2) {return float3(v1.x+v2.x , v1.y+v2.y , v1.z+v2.z);}  //  v1+v2
inline float3  operator-(const float3& v1  , const float3& v2) {return float3(v1.x-v2.x , v1.y-v2.y , v1.z-v2.z);}  //  v1-v2
inline float   operator*(const float3& v1  , const float3& v2) {return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;}          //  v1*v2
inline float3  operator*(float f           , const float3& v)  {return float3(f*v.x   , f*v.y   , f*v.z);}          //   f*v
//  Position, velocity and color
float3* pos[]={NULL,NULL};
float3* vel=NULL;
float*  M=NULL;
Color*  col=NULL;
//  OpenCL constants
cl_device_id     devid;
cl_context       context;
cl_command_queue queue;
cl_program       prog;
cl_kernel        kernel;
cl_mem           Dpos[2],Dvel,Dm;


/*
 *  Advance time one time step for star k
 */
void Move(float3 pos0[],float3 pos1[],int k)
{
   //  Position of this star
   float3 X = pos0[k];
   float dt = 1e-3;
   //  Calculate force components
   float3 F;
   for (int i=0;i<N;i++)
   {
      float3 D = pos0[i] - X;
      float d = sqrt(D*D)+1;  //  Add 1 to d to dampen movement
      F += M[i]/(d*d*d)*D;    // Normalize and scale to 1/r^2
   }
   //  Update velocity
   vel[k] += dt*F;
   //  Set new position
   pos1[k] = X + dt*vel[k];
}

/*
 *  Advance time one time step for star k
 */
const char* source =
"__kernel void Move(__global const float3 pos0[],__global float3 pos1[],__global float3 vel[],__global const float M[],const int N) \n"
"{                                 \n"
"   int k = get_global_id(0);      \n"
"   float3 X = pos0[k];            \n"
"   float dt = 1e-3;               \n"
"   float3 F = (float3)(0,0,0);    \n"
"   for (int i=0;i<N;i++)          \n"
"   {                              \n"
"      float3 D = pos0[i] - X;     \n"
"      float d = sqrt(dot(D,D))+1; \n"
"      F += M[i]/(d*d*d)*D;        \n"
"   }                              \n"
"   vel[k] += dt*F;                \n"
"   pos1[k] = X + dt*vel[k];       \n"
"}                                 \n"
;

/*
 *  Advance time one time step
 */
void Step()
{
   int k;
   //  Destination is 1-src;
   int dst = 1-src;

   //  Sequential
   if (mode==0)
   {
      for (k=0;k<N;k++)
         Move(pos[src],pos[dst],k);
   }
   //  OpenMP
   else if (mode==1)
   {
      #pragma omp parallel for
      for (k=0;k<N;k++)
         Move(pos[src],pos[dst],k);
   }
   //  OpenCL
   else
   {
      //  Set parameters for kernel
      if (clSetKernelArg(kernel,0,sizeof(Dvel),&Dpos[src])) Fatal("Cannot set kernel parameter Dpos0\n");
      if (clSetKernelArg(kernel,1,sizeof(Dvel),&Dpos[dst])) Fatal("Cannot set kernel parameter Dpos1\n");
      if (clSetKernelArg(kernel,2,sizeof(Dvel),&Dvel)) Fatal("Cannot set kernel parameter Dvel\n");
      if (clSetKernelArg(kernel,3,sizeof(Dm)  ,&Dm))   Fatal("Cannot set kernel parameter Dm\n");
      if (clSetKernelArg(kernel,4,sizeof(int) ,&N))    Fatal("Cannot set kernel parameter N\n");

      //  Queue kernel
      size_t Global[1] = {(size_t)N};
      size_t Local[1]  = {nloc};
      if (clEnqueueNDRangeKernel(queue,kernel,1,NULL,Global,Local,0,NULL,NULL)) Fatal("Cannot run kernel\n");
      //  Wait for kernel to finish
      if (clFinish(queue)) Fatal("Wait for kernel failed\n");

      // Copy pos from device to host (block until done)
      if (clEnqueueReadBuffer(queue,Dpos[dst],CL_TRUE,0,N*sizeof(float3),pos[dst],0,NULL,NULL)) Fatal("Cannot copy pos from device to host\n");
   }
   //  Set new source
   src = dst;
}

/*
 *  Scaled random value
 */
float rand1(float S)
{
   return S*exp(rand()/RAND_MAX);
}
float3 rand3(float S)
{
   float d = 2;
   float3 v(0,0,0);
   while (d>1)
   {
      v.x = rand()/(0.5*RAND_MAX)-1;
      v.y = rand()/(0.5*RAND_MAX)-1;
      v.z = rand()/(0.5*RAND_MAX)-1;
      d = v.x*v.x+v.y*v.y+v.z*v.z;
   }
   return S*v;
}

/*
 *  Initialize Nbody problem
 */
void InitLoc()
{
   int k;
   Color color[3];
   color[0] = Color(1.0,1.0,1.0);
   color[1] = Color(1.0,0.9,0.5);
   color[2] = Color(0.5,0.9,1.0);
   //  Allocate room for twice as many bodies to facilitate ping-pong
   pos[0] = (float3*)malloc(N*sizeof(float3));
   if (!pos[0]) Fatal("Error allocating memory for %d stars\n",N);
   pos[1] = (float3*)malloc(N*sizeof(float3));
   if (!pos[1]) Fatal("Error allocating memory for %d stars\n",N);
   vel = (float3*)malloc(N*sizeof(float3));
   if (!vel) Fatal("Error allocating memory for %d stars\n",N);
   M = (float*)malloc(N*sizeof(float));
   if (!M) Fatal("Error allocating memory for %d stars\n",N);
   col = (Color*)malloc(N*sizeof(Color));
   if (!col) Fatal("Error allocating memory for %d stars\n",N);
   //  Assign random locations
   for (k=0;k<N;k++)
   {
      pos[0][k] = rand3(dim/2);
      vel[k] = rand3(spd); 
      col[k] = color[k%3];
      M[k]   = rand1(mass);
   }
   //  Initialize src
   src = 0;
}

/*
 *  OpenGL (GLUT) calls this routine to display the scene
 */
void display()
{
   const double len=2.5;  //  Length of axes
   double Ex = -2*dim*Sin(th)*Cos(ph);
   double Ey = +2*dim        *Sin(ph);
   double Ez = +2*dim*Cos(th)*Cos(ph);

   //  Erase the window and the depth buffer
   glClear(GL_COLOR_BUFFER_BIT);
   //  Undo previous transformations
   glLoadIdentity();
   //  Perspective - set eye position
   gluLookAt(Ex,Ey,Ez , 0,0,0 , 0,Cos(ph),0);

   //  Integrate
   Step();

   //  Set shader
   glUseProgram(shader);
   int id = glGetUniformLocation(shader,"star");
   if (id>=0) glUniform1i(id,0);
   id = glGetUniformLocation(shader,"size");
   if (id>=0) glUniform1f(id,0.1);
   glBlendFunc(GL_ONE,GL_ONE);
   glEnable(GL_BLEND);

   //  Draw stars using vertex arrays
   glEnableClientState(GL_VERTEX_ARRAY);
   glEnableClientState(GL_COLOR_ARRAY);
   glVertexPointer(3,GL_FLOAT,sizeof(float3),pos[src]);
   glColorPointer(3,GL_FLOAT,sizeof(Color),col);
   //  Draw all stars
   glDrawArrays(GL_POINTS,0,N);
   //  Disable vertex arrays
   glDisableClientState(GL_VERTEX_ARRAY);
   glDisableClientState(GL_COLOR_ARRAY);

   //  Unset shader
   glUseProgram(0);
   glDisable(GL_BLEND);

   //  Draw axes
   glDisable(GL_LIGHTING);
   glColor3f(1,1,1);
   if (axes)
   {
      glBegin(GL_LINES);
      glVertex3d(0.0,0.0,0.0);
      glVertex3d(len,0.0,0.0);
      glVertex3d(0.0,0.0,0.0);
      glVertex3d(0.0,len,0.0);
      glVertex3d(0.0,0.0,0.0);
      glVertex3d(0.0,0.0,len);
      glEnd();
      //  Label axes
      glRasterPos3d(len,0.0,0.0);
      Print("X");
      glRasterPos3d(0.0,len,0.0);
      Print("Y");
      glRasterPos3d(0.0,0.0,len);
      Print("Z");
   }
   //  Display parameters
   glWindowPos2i(5,5);
   Print("FPS=%d Angle=%d,%d Mode=%s",
      FramesPerSecond(),th,ph,text[mode]);
   //  Render the scene and make it visible
   ErrCheck("display");
   glFlush();
   glutSwapBuffers();
}

/*
 *  GLUT calls this routine when an arrow key is pressed
 */
void special(int key,int x,int y)
{
   //  Right arrow key - increase angle by 5 degrees
   if (key == GLUT_KEY_RIGHT)
      th += 5;
   //  Left arrow key - decrease angle by 5 degrees
   else if (key == GLUT_KEY_LEFT)
      th -= 5;
   //  Up arrow key - increase elevation by 5 degrees
   else if (key == GLUT_KEY_UP)
      ph += 5;
   //  Down arrow key - decrease elevation by 5 degrees
   else if (key == GLUT_KEY_DOWN)
      ph -= 5;
   //  Keep angles to +/-360 degrees
   th %= 360;
   ph %= 360;
   //  Tell GLUT it is necessary to redisplay the scene
   glutPostRedisplay();
}

/*
 *  GLUT calls this routine when a key is pressed
 */
void key(unsigned char ch,int x,int y)
{
   int mode0 = mode;
   //  Exit on ESC
   if (ch == 27)
      exit(0);
   //  Cycle modes
   else if (ch == 'm')
      mode = (mode+1)%3;
   else if (ch == 'M')
      mode = (mode+2)%3;
   //  Reset view angle
   else if (ch == '0')
      th = ph = 0;
   //  Toggle axes
   else if (ch == 'a' || ch == 'A')
      axes = 1-axes;

   // Initialize pos and vel on device (block until done)
   if (mode==2 && mode0!=2)
   {
      if (clEnqueueWriteBuffer(queue,Dpos[src],CL_TRUE,0,N*sizeof(float3),pos[src],0,NULL,NULL)) Fatal("Cannot copy pos from host to device\n");
      if (clEnqueueWriteBuffer(queue,Dvel,CL_TRUE,0,N*sizeof(float3),vel,0,NULL,NULL)) Fatal("Cannot copy vel from host to device\n");
      if (clEnqueueWriteBuffer(queue,Dm,CL_TRUE,0,N*sizeof(float),M,0,NULL,NULL)) Fatal("Cannot copy M from host to device\n");
   }
   // Initiaize vel on host (block until done)
   else if (mode!=2 && mode0==2)
   {
      if (clEnqueueReadBuffer(queue,Dvel,CL_TRUE,0,N*sizeof(float3),vel,0,NULL,NULL)) Fatal("Cannot copy vel from device to host\n");
   }

   //  Tell GLUT it is necessary to redisplay the scene
   glutPostRedisplay();
}

/*
 *  GLUT calls this routine when the window is resized
 */
void reshape(int width,int height)
{
   int fov=55;       //  Field of view (for perspective)
   //  Ratio of the width to the height of the window
   double asp = (height>0) ? (double)width/height : 1;
   //  Set the viewport to the entire window
   glViewport(0,0, width,height);
   //  Set projection
   Project(fov,asp,dim);
}

/*
 *  GLUT calls this routine when the window is resized
 */
void idle()
{
   //  Tell GLUT it is necessary to redisplay the scene
   glutPostRedisplay();
}

//
//  Create Shader Program including Geometry Shader
//
int CreateShaderProgGeom()
{
   //  Create program
   int prog = glCreateProgram();
   //  Compile and add shaders
   CreateShader(prog,GL_VERTEX_SHADER  ,"nbody.vert");
#ifdef __APPLE__
   //  OpenGL 3.1 for OSX
   CreateShader(prog,GL_GEOMETRY_SHADER_EXT,"nbody.geom_ext");
   glProgramParameteriEXT(prog,GL_GEOMETRY_INPUT_TYPE_EXT  ,GL_POINTS);
   glProgramParameteriEXT(prog,GL_GEOMETRY_OUTPUT_TYPE_EXT ,GL_TRIANGLE_STRIP);
   glProgramParameteriEXT(prog,GL_GEOMETRY_VERTICES_OUT_EXT,4);
#else
   //  OpenGL 3.2 adds layout ()
   CreateShader(prog,GL_GEOMETRY_SHADER,"nbody.geom");
#endif
   CreateShader(prog,GL_FRAGMENT_SHADER,"nbody.frag");
   //  Link program
   glLinkProgram(prog);
   //  Check for errors
   PrintProgramLog(prog);
   //  Return name
   return prog;
}

/*
 *  Initialize OpenCL
 */
void InitCL()
{
   cl_int err;
   //  Initialize OpenCL
   nloc = InitGPU(1,devid,context,queue);
   //  Allocate memory for array on device
   Dpos[0] = clCreateBuffer(context,CL_MEM_READ_WRITE,N*sizeof(float3),NULL,&err);
   Dpos[1] = clCreateBuffer(context,CL_MEM_READ_WRITE,N*sizeof(float3),NULL,&err);
   Dvel    = clCreateBuffer(context,CL_MEM_READ_WRITE,N*sizeof(float3),NULL,&err);
   Dm      = clCreateBuffer(context,CL_MEM_WRITE_ONLY,N*sizeof(float) ,NULL,&err);
   if (err) Fatal("Cannot create array on device\n");
   //  Compile kernel
   prog = clCreateProgramWithSource(context,1,&source,0,&err);
   if (err) Fatal("Cannot create program\n");
   if (clBuildProgram(prog,0,NULL,NULL,NULL,NULL))
   {
      char log[1048576];
      if (clGetProgramBuildInfo(prog,devid,CL_PROGRAM_BUILD_LOG,sizeof(log),log,NULL))
         Fatal("Cannot get build log\n");
      else
         Fatal("Cannot build program\n%s\n",log);
   }
   kernel = clCreateKernel(prog,"Move",&err);
   if (err) Fatal("Cannot create kernel\n");
}

/*
 *  Start up GLUT and tell it what to do
 */
int main(int argc,char* argv[])
{
   //  Initialize GLUT
   glutInit(&argc,argv);
   //  Request double buffered, true color window with Z buffering at 600x600
   glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
   glutInitWindowSize(600,600);
   glutCreateWindow("Nbody Simulator");
#ifdef USEGLEW
   //  Initialize GLEW
   if (glewInit()!=GLEW_OK) Fatal("Error initializing GLEW\n");
   if (!GLEW_VERSION_2_0) Fatal("OpenGL 2.0 not supported\n");
#endif
   //  Set callbacks
   glutDisplayFunc(display);
   glutReshapeFunc(reshape);
   glutSpecialFunc(special);
   glutKeyboardFunc(key);
   glutIdleFunc(idle);
   //  Initialize stars
   InitLoc();
   //  Initialize OpenCL
   InitCL();
   //  Shader program
   shader = CreateShaderProgGeom();
   ErrCheck("init");
   //  Star texture
   LoadTexBMP("star.bmp");
   //  Pass control to GLUT so it can interact with the user
   glutMainLoop();
   return 0;
}
