/*
 * Simple program to demonstrate generating coordinates
 * using the Lorenz Attractor
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif


/*  View parameters  */
int th = 0;  //degrees
int ph = 0;  //degrees

/*  Lorenz Parameters  */
double s  = 10;
double b  = 2.6666;
double r  = 28;

// convenience function for scaling values to within 0.0 - 1.0
double scale(double dx)
{
   double vmax = 1/0.15;
   return fabs(dx*vmax)>1.0 ? 1.0 : fabs(dx*vmax);
}

void display()
{
   int i;
   /*  Coordinates  */
   double x = 1;
   double y = 1;
   double z = 1;
   /*  Time step  */
   double dt = 0.001;

   glClear(GL_COLOR_BUFFER_BIT);
   glLoadIdentity();

   glRotated(ph,1,0,0);
   glRotated(th,0,1,0);
   glScalef(0.035,0.035,0.035);
   
   // Axes
   glColor3f(1,1,1);
   glBegin(GL_LINES);
   glVertex3f(0.0, 0.0, 0.0);
   glVertex3f(30.0, 0.0, 0.0);
   glVertex3f(0.0, 0.0, 0.0);
   glVertex3f(0.0, 30.0, 0.0);
   glVertex3f(0.0, 0.0, 0.0);
   glVertex3f(0.0, 0.0, 30.0);
   glEnd();

   glBegin(GL_LINE_STRIP);
   glVertex3f(1, 1, 1);
   //printf("%5d %8.3f %8.3f %8.3f\n",0,x,y,z);
   /*
    *  Integrate 50,000 steps (50 time units with dt = 0.001)
    *  Explicit Euler integration
    */
   for (i=0;i<50000;i++)
   {
      double dx = s*(y-x);
      double dy = x*(r-z)-y;
      double dz = x*y - b*z;
      x += dt*dx;
      y += dt*dy;
      z += dt*dz;
      glColor3f(scale(dt*dx), scale(dt*dy), scale(dt*dz));
      //printf("%f,\t%f,\t%f\n",scale(dt*dx), scale(dt*dy), scale(dt*dz));
      glVertex3f(x, y, z);
      //printf("%5d %8.3f %8.3f %8.3f\n",i+1,x,y,z);
   }
   glEnd();
   
   glFlush();
   glutSwapBuffers();
}


//Reshape function from example 6 code
void reshape(int width, int height)
{
   //new aspect ratio
   double w2h = (height > 0) ? (double)width/height : 1;
   //set viewport to the whole new window
   glViewport(0,0 , width,height);
   
   //manipulate projection matrix
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   //adjust aspect ratio
   glOrtho(-2*w2h, +2*w2h, -2, +2, -2, +2);
   
   //go back to model matrix
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
}

void special(int key, int x, int y)
{
   switch (key)
   {
      // use the arrow keys to rotate the view
      case GLUT_KEY_RIGHT:
         th += 5;
         printf("th: %d\n", th);
         break;
      case GLUT_KEY_LEFT:
         th -= 5;
         printf("th: %d\n", th);
         break;
      case GLUT_KEY_UP:
         ph -= 5;
         printf("ph: %d\n", ph);
         break;
      case GLUT_KEY_DOWN:
         ph += 5;
         printf("ph: %d\n", ph);
         break;
   }

   th %= 360;
   ph %= 360;

   glutPostRedisplay();
}


int main(int argc, char *argv[])
{
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);

   glutInitWindowPosition(0,0);
   glutInitWindowSize(500,500);
   glutCreateWindow("4229 - Jordan Dick: Lorenz");

   glutDisplayFunc(display);
   glutReshapeFunc(reshape);
   glutSpecialFunc(special);

   glutMainLoop();

   return 0;
}
