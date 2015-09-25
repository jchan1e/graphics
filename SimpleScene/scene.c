// 
// manually draws various objects in a simple scene
//

#ifndef STDIncludes
#define STDIncludes
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#endif

#include "objects.c"

//GLOBAL VARIABLES//

//View Angles
int th = 30;
int ph = 20;
//Window Size
int w = 800;
int h = 800;

//sphere rotation
double r = 0;
double rate = 0.25; //rotation rate

////////////////////


void display()
{
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glEnable(GL_DEPTH_TEST);
   glEnable(GL_CULL_FACE);

   glLoadIdentity();
   //view angle
   glRotatef(ph, 1,0,0);
   glRotatef(th, 0,1,0);
   glScaled(0.4,0.4,0.4);

   sphere(0, 0, 0, 1.5*r, 0.5); //Jupiter
   glPushMatrix();
   glRotated(r/2, 0,1,0);
   cube(1, 0, 0, r, 0.25); //IO
   glPopMatrix();
   glPushMatrix();
   glRotated(r/4, 0,1,0);
   octahedron(-2, 0, 0, 4.0/3.0*r, 0.25); //Europa
   glPopMatrix();
   glPushMatrix();
   glRotated(r/8, 0,1,0);
   dodecahedron(3, 0, 0, -1.125*r, 0.25); //Ganymede
   glPopMatrix();
   glPushMatrix();
   glRotated(r/18.4, 0,1,0);
   icosahedron(4, 0, 0, 0.75*r, 0.25); //Callisto
   glPopMatrix();

   r += rate;
   r = fmod(r, 360*24*9.2);
   glFlush();
   glutSwapBuffers();
}

void reshape(int width, int height)
{
   w = width;
   h = height;
   //new aspect ratio
   double w2h = (height > 0) ? (double)width/height : 1;
   //set viewport to the new window
   glViewport(0,0 , width,height);

   //switch to projection matrix
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   //adjust aspect ratio
   glOrtho(-2*w2h, 2*w2h, -2, 2, -2, 2);
   //gluPerspective(60, w2h, 3/4, 3*4);

   //switch back to model matrix
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
}

void special(int key, int mousex, int mousey)
{
   switch(key)
   {
      case GLUT_KEY_UP:
         ph += 5;
         break;
      case GLUT_KEY_DOWN:
         ph -= 5;
         break;
      case GLUT_KEY_LEFT:
         th += 5;
         break;
      case GLUT_KEY_RIGHT:
         th -= 5;
         break;
   }
   ph %= 360;
   th %= 360;
}

void keyboard(unsigned char key, int mousex, int mousey)
{
   switch(key)
   {
      case 27: //escape
         exit(0);
         break;
      case 'q':
         exit(0);
         break;
      case '.':
         rate *= 2;
         break;
      case ',':
         rate /= 2;
         break;
   }
   glutPostRedisplay();
}

void idle()
{
   glutPostRedisplay();
}

int main(int argc, char *argv[])
{
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);

   glutInitWindowPosition(0,0);
   glutInitWindowSize(w,h);
   glutCreateWindow("Jordan Dick: HW3 - Galilean Moons");

   glutDisplayFunc(display);
   glutReshapeFunc(reshape);
   glutSpecialFunc(special);
   glutKeyboardFunc(keyboard);
   //glutPassiveMotionFunc(motion);
   glutIdleFunc(idle);

   glutMainLoop();

   return EXIT_SUCCESS;
}
