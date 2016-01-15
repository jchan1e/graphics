/*
 *  Piecewise Polynomials
 *
 *  Demonstarte different methods of interpolation
 *    Continuous Bezier
 *    Smooth Bezier
 *    Interpolated Bezier
 *    B Spline
 *    Natural Cubic Spline
 *
 *  Key bindings:
 *  +/-        Cycle through interpolation methods
 *  LeftClick  Add a point
 *  RightClick Move a point
 *  x          Delete the last point
 *  0          Delete all points
 *  m/M        Increase/decrease number of evaluations
 *  ESC        Exit
 */
#include "CSCIx229.h"
int width  = 1;
int height = 1;
double asp = 1;
//  Data points
#define NMAX 25 //  Maximum number of points
#define MODE 5  //  Number of methods
int mode = 0;   //  Interpolation method
int n=0;        //  Number of points
int move=-1;    //  Point to move
int m=25;       //  Points on line
typedef struct {double x,y,z;} Point;
Point P[NMAX];  //  Data points
char* text[] = {"Continuous Bezier","Smooth Bezier","Interpolated","B Spline","Natural Cubic Spline"};

//  Interpolation matrix for Bezier
static double Mi[4][4] =
{
   { 1.00000000, 0.00000000, 0.00000000, 0.00000000},
   {-0.83333333, 3.00000000,-1.50000000, 0.33333333},
   { 0.33333333,-1.50000000, 3.00000000,-0.83333333},
   { 0.00000000, 0.00000000, 0.00000000, 1.00000000},
};
//  Interpolation matrix for B-spline
static double Ms[4][4] =
{
   {0.166666667, 0.666666667, 0.166666667, 0.000000000},
   {0.000000000, 0.666666667, 0.333333333, 0.000000000},
   {0.000000000, 0.333333333, 0.666666667, 0.000000000},
   {0.000000000, 0.166666667, 0.666666667, 0.166666667},
};

// This routine multiplies a 4 x 4 matrix with a vector of 4 points.
void Pmult(double M[4][4],Point v[4],Point r[4])
{
   int i,k;
   for (i=0;i<4;i++)
   {
      r[i].x = r[i].y = r[i].z = 0;
      for (k=0;k<4;k++)
      {
         r[i].x += M[i][k]*v[k].x;
         r[i].y += M[i][k]*v[k].y;
         r[i].z += M[i][k]*v[k].z;
      }
   }
}

/*
 *  Compute natural cubic spline
 *  X is the array of points
 *  S is the second derivative (calculated here)
 *  k is the component
 *  n is the number of points
 */
void CubicSpline(double X[NMAX][3],double S[NMAX][3],int k,int n)
{
   int i;
   double W[NMAX];
   S[0][k] = W[0] = 0;
   for (i=1;i<n-1;i++)
   {
      double f = 0.5*S[i-1][k] + 2;
      S[i][k] = -0.5/f;
      W[i] = (3*(X[i+1][k]-2*X[i][k]+X[i-1][k]) - 0.5*W[i-1])/f;
   }
   S[n-1][k] = 0;
   for (i=n-2;i>0;i--)
      S[i][k] = S[i][k]*S[i+1][k] + W[i];
}

/*
 *  OpenGL (GLUT) calls this routine to display the scene
 */
void display()
{
   int k,i;
   Point S[NMAX];

   //  Erase the window and the depth buffer
   glClear(GL_COLOR_BUFFER_BIT);
   //  Enable vertex generation
   glEnable(GL_MAP1_VERTEX_3);

   //  Draw and label points
   glColor3f(1,1,1);
   glPointSize(5);
   glBegin(GL_POINTS);
   for (k=0;k<n;k++)
      glVertex2f(P[k].x,P[k].y);
   glEnd();
   for (k=0;k<n;k++)
   {
      glRasterPos2d(P[k].x,P[k].y);
      Print("P%d",k);
   }
   //  Draw curve when we have enough points
   switch (mode)
   {
      //  Continuous Bezier
      case 0:
         for (k=0;k<n-3;k+=3)
         {
            glColor3f(1,1,0);
            glMap1d(GL_MAP1_VERTEX_3,0.0,1.0,3,4,(void*)(P+k));
            glMapGrid1f(m,0.0,1.0);
            glEvalMesh1(GL_LINE,0,m);
         }
         break;
      //  Smooth Bezier
      case 1:
         //  First four
         if (n>=4)
         {
            glColor3f(1,1,0);
            glMap1d(GL_MAP1_VERTEX_3,0.0,1.0,3,4,(void*)P);
            glMapGrid1f(m,0.0,1.0);
            glEvalMesh1(GL_LINE,0,m);
         }
         //  Subsequent triples
         for (k=3;k<n-2;k+=2)
         {
            Point r[4];
            //  P0, P2 and P3 are specified points
            r[0] = P[k];
            r[2] = P[k+1];
            r[3] = P[k+2];
            //  P1 is computed by reflecting P2 from previous segment
            r[1].x = 2*P[k].x-P[k-1].x;
            r[1].y = 2*P[k].y-P[k-1].y;
            r[1].z = 2*P[k].z-P[k-1].z;
            //  Draw curve
            glColor3f(1,1,0);
            glMap1d(GL_MAP1_VERTEX_3,0.0,1.0,3,4,(void*)r);
            glMapGrid1f(m,0.0,1.0);
            glEvalMesh1(GL_LINE,0,m);
            //  Draw reflected point in red
            glColor3f(1,0,0);
            glBegin(GL_POINTS);
            glVertex2f(r[1].x,r[1].y);
            glEnd();
            glRasterPos2d(r[1].x,r[1].y);Print("R%d",k-1);
         }
         break;
      //  Interpolate 4 at a time
      case 2:
         for (k=0;k<n-3;k+=3)
         {
            Point r[4];
            //  Transform 4 data points to 4 control points
            Pmult(Mi,P+k,r);
            //  Draw curve
            glColor3f(1,1,0);
            glMap1d(GL_MAP1_VERTEX_3,0.0,1.0,3,4,(void*)r);
            glMapGrid1f(m,0.0,1.0);
            glEvalMesh1(GL_LINE,0,m);
            //  Draw control points
            glColor3f(1,0,0);
            glBegin(GL_POINTS);
            glVertex2f(r[1].x,r[1].y);
            glVertex2f(r[2].x,r[2].y);
            glEnd();
            glRasterPos2d(r[1].x,r[1].y);Print("R%d",k+1);
            glRasterPos2d(r[2].x,r[2].y);Print("R%d",k+2);
         }
         break;
      //  B Spline 4 at a time
      case 3:
         for (k=0;k<n-3;k++)
         {
            int j;
            Point r[4];
            //  Transform 4 data points to 4 control points (note increment is 1)
            Pmult(Ms,P+k,r);
            //  Draw curve
            glColor3f(1,1,0);
            glMap1d(GL_MAP1_VERTEX_3,0.0,1.0,3,4,(void*)r);
            glMapGrid1f(m,0.0,1.0);
            glEvalMesh1(GL_LINE,0,m);
            //  Draw control points
            glColor3f(1,0,0);
            glBegin(GL_POINTS);
            for (j=0;j<4;j++)
               glVertex2f(r[j].x,r[j].y);
            glEnd();
         }
         break;
      //  Cubic Natural Spline
      case 4:
         //  Calculate (x,y,z) splines
         CubicSpline((void*)P,(void*)S,0,n);
         CubicSpline((void*)P,(void*)S,1,n);
         CubicSpline((void*)P,(void*)S,2,n);
         //  Draw entire curve
         glColor3f(1,1,0);
         glBegin(GL_LINE_STRIP);
         for (k=0;k<n-1;k++)
         {
            //  Cardinal point
            glVertex2d(P[k].x,P[k].y);
            //  Increment between k and k+1
            for (i=1;i<m;i++)
            {
               double f  = (double)i/m;
               double g  = 1-f;
               double f2 = (f*f*f-f)/6;
               double g2 = (g*g*g-g)/6;
               double x  = f*P[k+1].x + g*P[k].x + f2*S[k+1].x + g2*S[k].x;
               double y  = f*P[k+1].y + g*P[k].y + f2*S[k+1].y + g2*S[k].y;
               double z  = f*P[k+1].z + g*P[k].z + f2*S[k+1].z + g2*S[k].z;
               glVertex3d(x,y,z);
            }
         }
         //  Last cardinal point
         glVertex2d(P[n-1].x,P[n-1].y);
         glEnd();
         break;
      default:
         break;
   }
   
   //  Display parameters
   glColor3f(1,1,1);
   glWindowPos2i(5,5);
   Print(text[mode]);
   if (n<NMAX)
      Print(" Click to add point\n");
   else
      Print(" Click to start new curve\n");

   //  Render the scene and make it visible
   ErrCheck("display");
   glFlush();
   glutSwapBuffers();
}

/*
 *  GLUT calls this routine when a key is pressed
 */
void key(unsigned char ch,int x,int y)
{
   //  Exit on ESC
   if (ch == 27)
      exit(0);
   //  Reset
   else if (ch == '0')
      n = 0;
   //  Delete one point
   else if (ch == 'x' && n>0)
      n--;
   //  Cycle modes
   else if (ch == '-')
      mode = (mode+MODE-1)%MODE;
   else if (ch == '+')
      mode = (mode+1)%MODE;
   //  Tell GLUT it is necessary to redisplay the scene
   glutPostRedisplay();
}

/*
 *  GLUT calls this routine when the window is resized
 */
void reshape(int w,int h)
{
   //  Remeber dimensions
   width  = w;
   height = h;
   //  Ratio of the width to the height of the window
   asp = (h>0) ? (double)width/height : 1;
   //  Set the viewport to the entire window
   glViewport(0,0, width,height);
   //  Project
   Project(0,asp,1);
}

/*
 *  Translate mouse (x,y) to world coordinates
 */
Point Mouse2World(int x,int y)
{
   Point p;
   p.x = 2*asp*x       /(float)(width -1) - asp;
   p.y = 2*(height-1-y)/(float)(height-1) - 1;
   p.z = 0;
   return p;
}

/*
 *  Distance to point
 */
double Distance(Point p,int k)
{
   double dx = p.x - P[k].x;
   double dy = p.y - P[k].y;
   double dz = p.z - P[k].z;
   return sqrt(dx*dx+dy*dy+dz*dz);
}

/*
 *  GLUT calls this routine when a mouse is moved
 */
void motion(int x,int y)
{
   if (move<0) return;
   //  Update point
   P[move] = Mouse2World(x,y);
   //  Redisplay
   glutPostRedisplay();
}

/*
 *  GLUT calls this routine when a mouse button is pressed or released
 */
void mouse(int button, int state, int x, int y)
{
   Point p = Mouse2World(x,y);

   //  Add a point
   if (button==GLUT_LEFT_BUTTON && state==GLUT_DOWN)
   {
      int k = n%NMAX;
      n = k+1;
      P[k] = p;
      move = -1;
   }
   //  Stop move
   else if (button==GLUT_RIGHT_BUTTON && state==GLUT_UP && move>=0)
   {
      P[move] = p;
      move = -1;
   }
   //  Start move
   else if (button==GLUT_RIGHT_BUTTON && state==GLUT_DOWN && n>0)
   {
      ///  Find nearest point
      int k;
      double dmin = Distance(p,0);
      move = 0;
      for (k=1;k<n;k++)
      {
         double d = Distance(p,k);
         if (d<dmin)
         {
            move = k;
            dmin = d;
         }
      }
      P[move] = p;
   }
   //  Redisplay
   glutPostRedisplay();
}

/*
 *  Start up GLUT and tell it what to do
 */
int main(int argc,char* argv[])
{
   //  Initialize GLUT
   glutInit(&argc,argv);
   //  Request double buffered, true color window with Z buffering at 600x600
   glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
   glutInitWindowSize(600,600);
   glutCreateWindow("Piecewise Polynomials");
   //  Set callbacks
   glutDisplayFunc(display);
   glutReshapeFunc(reshape);
   glutKeyboardFunc(key);
   glutMouseFunc(mouse);
   glutMotionFunc(motion);
   //  Pass control to GLUT so it can interact with the user
   ErrCheck("init");
   glutMainLoop();
   return 0;
}
