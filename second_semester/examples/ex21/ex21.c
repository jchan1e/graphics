/*
 *  nBody Simulator
 *  This program requires OpenGL 3.2 or above
 *
 *  Demonstartes a geometry shader.
 *  OpenMP is used to accelerate computations.
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
int N=1024;       //  Number of bodies
int src=0;        //  Offset of first star in source
int dst=0;        //  Offset of first star in destination
int axes=0;       //  Display axes
int th=0;         //  Azimuth of view angle
int ph=0;         //  Elevation of view angle
double dim=10;    //  Size of universe
double vel=0.1;   //  Relative speed
int mode=0;       //  Solver mode
int shader=0;     //  Shader
char* text[] = {"Sequential","OpenMP","OpenMP+Geometry Shader"};

//  Star
typedef struct
{
   float x,y,z;  //  Position
   float u,v,w;  //  Velocity
   float r,g,b;  //  Color
}  Star;
Star* stars=NULL;

/*
 *  Advance time one time step for star k
 */
void Move(int k)
{
   int k0 = k+src;
   int k1 = k+dst;
   float dt = 1e-3;
   int i;
   //  Calculate force components
   double a=0;
   double b=0;
   double c=0;
   for (i=src;i<src+N;i++)
   {
      double dx = stars[i].x-stars[k0].x;
      double dy = stars[i].y-stars[k0].y;
      double dz = stars[i].z-stars[k0].z;
      double d = sqrt(dx*dx+dy*dy+dz*dz)+1;  // Add 1 to d to dampen movement
      double f = 1/(d*d*d);                  // Normalize and scale to 1/r^2
      a += f*dx;
      b += f*dy;
      c += f*dz;
   }
   //  Update velocity
   stars[k1].u = stars[k0].u + dt*a;
   stars[k1].v = stars[k0].v + dt*b;
   stars[k1].w = stars[k0].w + dt*c;
   //  Update position
   stars[k1].x = stars[k0].x + dt*stars[k1].u;
   stars[k1].y = stars[k0].y + dt*stars[k1].v;
   stars[k1].z = stars[k0].z + dt*stars[k1].w;
}

/*
 *  Advance time one time step
 */
void Step()
{
   int k;
   //  Switch source and destination
   src = src?0:N;
   dst = dst?0:N;
   //  OpenMP
   if (mode)
      #pragma omp parallel for
      for (k=0;k<N;k++)
         Move(k);
   //  Sequential
   else
      for (k=0;k<N;k++)
         Move(k);
}

/*
 *  Scaled random value
 */
void rand3(float Sx,float Sy,float Sz,float* X,float* Y,float* Z)
{
   float x = 0;
   float y = 0;
   float z = 0;
   float d = 2;
   while (d>1)
   {
      x = rand()/(0.5*RAND_MAX)-1;
      y = rand()/(0.5*RAND_MAX)-1;
      z = rand()/(0.5*RAND_MAX)-1;
      d = x*x+y*y+z*z;
   }
   *X = Sx*x;
   *Y = Sy*y;
   *Z = Sz*z;
}

/*
 *  Initialize nBody problem
 */
void InitLoc()
{
   int k;
   //  Allocate room for twice as many bodies to facilitate ping-pong
   if (!stars) stars = malloc(2*N*sizeof(Star));
   if (!stars) Fatal("Error allocating memory for %d stars\n",N);
   src = N;
   dst = 0;
   //  Assign random locations
   for (k=0;k<N;k++)
   {
      rand3(dim/2,dim/2,dim/3,&stars[k].x,&stars[k].y,&stars[k].z);
      rand3(vel,vel,vel,&stars[k].u,&stars[k].v,&stars[k].w);
      switch (k%3)
      {
         case 0:
           stars[k].r = 1.0;
           stars[k].g = 1.0;
           stars[k].b = 1.0;
           break;
         case 1:
           stars[k].r = 1.0;
           stars[k].g = 0.9;
           stars[k].b = 0.5;
           break;
         case 2:
           stars[k].r = 0.5;
           stars[k].g = 0.9;
           stars[k].b = 1.0;
           break;
      }
      stars[k+N].r = stars[k].r;
      stars[k+N].g = stars[k].g;
      stars[k+N].b = stars[k].b;
   }
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
   if (mode==2)
   {
      glUseProgram(shader);
      int id = glGetUniformLocation(shader,"star");
      if (id>=0) glUniform1i(id,0);
      id = glGetUniformLocation(shader,"size");
      if (id>=0) glUniform1f(id,0.3);
      glBlendFunc(GL_ONE,GL_ONE);
      glEnable(GL_BLEND);
   }

   //  Draw stars using vertex arrays
   glEnableClientState(GL_VERTEX_ARRAY);
   glEnableClientState(GL_COLOR_ARRAY);
   glVertexPointer(3,GL_FLOAT,sizeof(Star),&stars[0].x);
   glColorPointer(3,GL_FLOAT,sizeof(Star),&stars[0].r);
   //  Draw all stars from dst count N
   glDrawArrays(GL_POINTS,dst,N);
   //  Disable vertex arrays
   glDisableClientState(GL_VERTEX_ARRAY);
   glDisableClientState(GL_COLOR_ARRAY);

   //  Unset shader
   if (mode==2)
   {
      glUseProgram(0);
      glDisable(GL_BLEND);
   }

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
   //  Exit on ESC
   if (ch == 27)
      exit(0);
   //  Cycle modes
   else if (ch == 'm')
      mode = (mode+1)%3;
   //  Reset view angle
   else if (ch == '0')
      th = ph = 0;
   //  Reset simulation
   else if (ch == 'r')
      InitLoc();
   //  Toggle axes
   else if (ch == 'a' || ch == 'A')
      axes = 1-axes;
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
 *  Start up GLUT and tell it what to do
 */
int main(int argc,char* argv[])
{
   //  Initialize GLUT
   glutInit(&argc,argv);
   //  Request double buffered, true color window with Z buffering at 600x600
   glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
   glutInitWindowSize(600,600);
   glutCreateWindow("nBody Simulator");
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
   //  Shader program
   shader = CreateShaderProgGeom();
   ErrCheck("init");
   //  Star texture
   LoadTexBMP("star.bmp");
   //  Pass control to GLUT so it can interact with the user
   glutMainLoop();
   return 0;
}
