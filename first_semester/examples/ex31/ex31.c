/*
 * 1D Evaluators
 *
 *  Demonstrate the use of Bezier polygons.
 *
 *  Key bindings:
 *  +/-        Increase/decrease order of polygon
 *  LeftClick  Add a point
 *  RightClick Move a point
 *  x          Delete the last point
 *  0          Delete all points
 *  m/M        Increase/decrease number of evaluations
 *  ESC        Exit
 */
#include "CSCIx229.h"
int width  = 1;  //  Window width
int height = 1;  //  Window height
double asp = 1;  //  Window aspect ratio
//  Data points
#define NMIN  3  //  Minimum order
#define NMAX  8  //  Maximum order
#define MEVAL 25 //  Maximum evaluations
int n=0;       //  Number of points
int move=-1;   //  Point being moved
int hull=0;    //  Convex hull display list
int N=NMIN;    //  Maximum order
int m=MEVAL;   //  Number of evaluations
typedef struct {double x,y,z;} Point;
Point P[NMAX]; //  Data points

/*
 *  Find convex hull of the points (2d)
 *  Uses the Gift Wrap algorithm
 */
void Hull2D()
{
   int k=0;        //  Number of points
   int i,i0,i1,i2; //  Counters
   int I[NMAX+1];  //  Ordered vertexes

   //  Need at least three points
   if (n<3)
   {
      if (hull) glDeleteLists(hull,1);
      hull = 0;
      return;
   }

   //  Start at point with minimum y value
   i0 = 0;
   for (i=1;i<n;i++)
      if (P[i].y < P[i0].y) i0 = i;
   I[k++] = i0;

   //  Loop until next point is the first point
   while (k==1 || I[0]!=I[k-1])
   {
      //  Loop over all potential next points
      for (i1=0;i1<n;i1++)
      {
         //  Skip myself
         if (i1==i0) continue;
         //  Check that all points are to the left
         for (i2=0,i=1;i2<n && i;i2++)
            i = (i2==i0) || (i2==i1) || ((P[i1].x-P[i0].x)*(P[i2].y-P[i0].y)-(P[i2].x-P[i0].x)*(P[i1].y-P[i0].y)>=0);
         //  Check passed - record and move to next line segment
         if (i)
         {
            i0 = I[k++] = i1;
            break;
         }
      }
   }
   k--;

   //  Store hull in display list
   hull = glGenLists(1);
   glNewList(hull,GL_COMPILE);
   glColor3f(1,0,0);
   glBegin(GL_LINE_LOOP);
   for (i=0;i<k;i++)
      glVertex2d(P[I[i]].x,P[I[i]].y);
   glEnd();
   glEndList();
}

/*
 *  OpenGL (GLUT) calls this routine to display the scene
 */
void display()
{
   int k;

   //  Erase the window and the depth buffer
   glClear(GL_COLOR_BUFFER_BIT);

   //  Compute and draw convex hull
   if (hull) glCallList(hull);
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
   //  Draw Tangents
   glColor3f(0,0,1);
   glBegin(GL_LINES);
   if (n>2)
   {
      glVertex2f(P[0].x,P[0].y);
      glVertex2f(P[1].x,P[1].y);
   }
   if (n>=3)
   {
      glVertex2f(P[n-2].x,P[n-2].y);
      glVertex2f(P[n-1].x,P[n-1].y);
   }
   glEnd();
   //  Draw Bezier when we have enough points
   if (n>1)
   {
      glColor3f(1,1,0);

      glEnable(GL_MAP1_VERTEX_3);
      glMap1d(GL_MAP1_VERTEX_3,0.0,1.0,3,n,(void*)P);
/*
      glBegin(GL_LINE_STRIP);
      for (k=0;k<=MEVAL;k++)
         glEvalCoord1f(k/(float)MEVAL)
      glEnd();
*/
      glMapGrid1f(MEVAL,0.0,1.0);
      glEvalMesh1(GL_LINE, 0, MEVAL);
      //  Draw coarser approximation (for demonstration purposes only)
      if (m<MEVAL)
      {
         glLineWidth(2);
         glMapGrid1f(m, 0.0, 1.0);
         glEvalMesh1(GL_POINT, 0, m);
         glEvalMesh1(GL_LINE, 0, m);
         glLineWidth(1);
      }
   }
   
   //  Display parameters
   glColor3f(1,1,1);
   glWindowPos2i(5,5);
   Print("Order %d Evaluations=%d",N,m);
   if (n<N)
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
   //  Increase order
   else if (ch == '+')
      N++;
   //  Increase order
   else if (ch == '-')
      N--;
   //  Cycle m
   else if (ch == 'm')
      m = m%25+1;
   else if (ch == 'M')
      m = (m+23)%25+1;
   //  Reset
   else if (ch == '0')
      n = 0;
   //  Delete one point
   else if (ch == 'x' && n>0)
      n--;
   //  Limit range of N
   if (N<NMIN)
     N = NMAX;
   else if (N>NMAX)
     N = NMIN;
   //  Calculate convex hull
   Hull2D();
   //  Tell GLUT it is necessary to redisplay the scene
   glutPostRedisplay();
}

/*
 *  GLUT calls this routine when the window is resized
 */
void reshape(int w,int h)
{
   //  Remember dimensions
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
   //  Calculate convex hull
   Hull2D();
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
      int k = n%N;
      n = k+1;
      P[k] = p;
      move = -1;
   }
   //  Stop move
   else if (button==GLUT_RIGHT_BUTTON && state==GLUT_UP)
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
   //  Calculate convex hull
   Hull2D();
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
   glutCreateWindow("Bezier Polynomials");
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
