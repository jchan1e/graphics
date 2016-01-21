/*
 *  Anti-aliased Lorenz Attractor
 *
 *  Draw the Lorenz Attractor with anti-aliasing.
 *
 *  Key bindings:
 *  m          Switch mode between Jagged/Anti-Aliased
 *  d/D        Increase/decrease length of line segments (step)
 *  0          Color shows time
 *  1          Single (red) trace
 *  2          Two (red & green) traces (perturbed origin)
 *  3          Three (red, green & blue) traces (perturbed origins)
 *  +/-        Change value of r (rho) parameter
 *  x/X        View down the X-axis
 *  y/Y        View down the Y-axis
 *  z/Z        View down the Z-axis
 *  a          Toggle axes
 *  arrows     Change view angle
 *  PgDn/PgUp  Zoom in and out
 *  0          Reset view angle
 *  ESC        Exit
 */

#include "CSCIx229.h"

/*  Projection Parameters  */
int    mode = 0;
int    step = 0;
double Len  = 50;
double asp  = 1;
static double dim  = 60;
int ph =  20;
int th = -30;
/*  Lorenz Parameters  */
int    n  = 0;
double s  = 10;
double b  = 2.6666;
double r  = 28;
/*  Location  */
double X=1;
double Y=1;
double Z=1;
/*  Last Direction and Time  */
double t0=0;
double dx0=1;
double dy0=1;
double dz0=1;

/*
 *  Display the Lorenz Contractor
 */
static void lorenz(double x,double y,double z,int color)
{
   /*  Time step  */
   int i;
   double dt = 0.001;
   /*  Colors */
   double R[] = {1.0,1.0,1.0,0.0,0.0,0.0};
   double G[] = {0.0,0.5,1.0,1.0,1.0,0.0};
   double B[] = {0.0,0.0,0.0,0.0,1.0,1.0};
   /*  Start trace  */
   glBegin(GL_LINE_STRIP);
   if (color) glColor3f(R[0],G[0],B[0]);
   glVertex3d(x,y,z);
   /*  Integrate  */
   double d=0;
   for (i=0;i<50000;i++)
   {
      double dx = s*(y-x);
      double dy = x*(r-z)-y;
      double dz = x*y - b*z;
      x += dt*dx;
      y += dt*dy;
      z += dt*dz;
      d += dt*sqrt(dx*dx+dy*dy+dz*dz);
      if (color) glColor3f(R[i/10000],G[i/10000],B[i/10000]);
      if (d>=0.1*step)
      {
         d = 0;
         glVertex3d(x,y,z);
      }
   }
   glEnd();
}

/*
 *  Display the Lorenz Contractor
 */
static void display(void)
{

   /*  Erase the last picture  */
   glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
   glEnable(GL_DEPTH_TEST);
   /*  Set anti-aliasing  */
   if (mode)
   {
      glEnable(GL_LINE_SMOOTH);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
      glHint(GL_LINE_SMOOTH_HINT,GL_NICEST);
      glLineWidth(1.5);
   }
   /*  Jagged single line  */
   else
   {
      glDisable(GL_LINE_SMOOTH);
      glDisable(GL_BLEND);
      glLineWidth(1.0);
   }
   /*  Set transofrmation   */
   glPushMatrix();
   glRotatef(ph, 1, 0, 0);
   glRotatef(th, 0, 1, 0);
   /*  Trace  */
   switch (n)
   {
   // Blue trace
   case 3:
      glColor3f(0,0,1);
      lorenz(1.02,1.02,1.02,0);
      /*  Fall through  */
   // Green trace
   case 2:
      glColor3f(0,1,0);
      lorenz(1.01,1.01,1.01,0);
      /*  Fall through  */
   // Red trace
   case 1:
      glColor3f(1,0,0);
      lorenz(1.00,1.00,1.00,0);
      break;
   // Multi-color trace
   default:
      lorenz(1,1,1,1);
      break;
   }
   /*  Axes  */
   glBegin(GL_LINES);
   glColor3f(1,1,1);
   glVertex3d(0.0,0.0,0.0);
   glVertex3d(Len,0.0,0.0);
   glVertex3d(0.0,0.0,0.0);
   glVertex3d(0.0,Len,0.0);
   glVertex3d(0.0,0.0,0.0);
   glVertex3d(0.0,0.0,Len);
   glEnd();
   /*  Text  */
   glWindowPos2i(5,5);
   Print("Rx=%d,Ry=%d,rho=%.1f Step=%.1f Mode=%s",ph,th,r,0.1*step,mode?"Anti-aliased":"Jagged");
   glRasterPos3d(Len,0.0,0.0);
   Print("X");
   glRasterPos3d(0.0,Len,0.0);
   Print("Y");
   glRasterPos3d(0.0,0.0,Len);
   Print("Z");
   /*  Pop and flush  */
   glPopMatrix();
   ErrCheck("display");
   glFlush();
   glutSwapBuffers();
}

/*
 *  Exit on ESC
 */
static void Keyboard(unsigned char ch, int x, int y)
{
   if (ch=='x' || ch=='X')
   {
      ph =  0;
      th = 90;
   }
   else if (ch=='y' || ch=='Y')
   {
      ph = -90;
      th =   0;
   }
   else if (ch=='z' || ch=='Z')
      ph = th = 0;
   else if ('0'<=ch && ch<='3')
      n = ch-'0';
   else if (ch=='d' && step<20)
      step++;
   else if (ch=='D' && step>0)
      step--;
   else if (ch=='+' && r<50)
      r += 0.1;
   else if (ch=='-' && r>1)
      r -= 0.1;
   else if (ch=='m')
      mode = 1-mode;
   else if (ch==27)
      exit(0);
   glutPostRedisplay();
}

/*
 *  Special user actions
 */
static void Special(int ch, int x, int y)
{
   if (ch==GLUT_KEY_UP)
      ph += 5;
   else if (ch==GLUT_KEY_DOWN)
      ph -= 5;
   else if (ch==GLUT_KEY_LEFT)
      th += 5;
   else if (ch==GLUT_KEY_RIGHT)
      th -= 5;
   else if (ch==GLUT_KEY_PAGE_DOWN && dim<100)
      dim += 5;
   else if (ch==GLUT_KEY_PAGE_UP && dim>10)
      dim -= 5;
   ph = ph%360;
   th = th%360;
   Project(0,asp,dim);
   glutPostRedisplay();
}

/*
 *  Special user actions
 */
static void Idle(void)
{
   if (n<0) glutPostRedisplay();
}

/*
 *  New window size or Exposure
 */
static void Reshape(GLint width,GLint height)
{
   asp = (double)width/height;
   /*  Set View  */
   glViewport(0,0, width,height);
   Project(0,asp,dim);
}

/*
 *  Main
 */
int main(int argc, char *argv[])
{
   /*  Create Window  */
   glutInit(&argc,argv);
   glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_ALPHA);
   glutInitWindowSize(600,600);
   glutCreateWindow("Anti-Aliasing");
   /*  Set Callbacks  */
   glutDisplayFunc(display);
   glutReshapeFunc(Reshape);
   glutKeyboardFunc(Keyboard);
   glutSpecialFunc(Special);
   glutIdleFunc(Idle);
   /*  Pass Control  */
   ErrCheck("init");
   glutMainLoop();
   return 0;
}
