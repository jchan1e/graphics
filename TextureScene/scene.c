// hw4 - scene.c
// manually draws various objects in a simple scene
// Jordan Dick
// jordan.dick@colorado.edu

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
int th = 0;
int ph = 0;
//Window Size
int w = 800;
int h = 800;

//orbits & rotation
double r = 0;
double rate = 1/8.0;

//perspective mode
int mode = 1;  // 0 = ortho, 1 = perspective, 2 = first person

//eye position and orientation
double ex = 0;
double ey = 0;
double ez = 0;

double vx = 0;
double vy = 0;
double vz = 0;

//lighting arrays
float Ambient[4];
float Diffuse[4];
float Specular[4];
float shininess[1];
float Position[3]; 

////////////////////


void display()
{
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glEnable(GL_DEPTH_TEST);
   glEnable(GL_CULL_FACE);

   glLoadIdentity();

   //view angle
   if (mode == 0) //rotation for ortho mode
   {
      glRotatef(ph, 1,0,0);
      glRotatef(th, 0,1,0);
      glScaled(0.4,0.4,0.4);
   }
   else if (mode == 1) //rotation for perspective mode
   {
      ex = Sin(-th)*Cos(ph)*8;
      ey = Sin(ph)*8;
      ez = Cos(-th)*Cos(ph)*8;

      gluLookAt(ex,ey,ez , 0,0,0 , 0,Cos(ph),0);
      //glScaled(0.3,0.3,0.3);
   }
   else //mode == 2              // rotation and movement for FP mode
   {                             // occur in keyboard & special
      vx = ex - Sin(th)*Cos(ph); // here we simply update
      vy = ey - Sin(ph);         // location of view target
      vz = ez - Cos(th)*Cos(ph);

      gluLookAt(ex,ey,ez , vx,vy,vz , 0,Cos(ph),0);
   }

   
   //////////Lighting//////////

   // Light position and rendered marker (unlit)
   //glDisable(GL_LIGHTING);
   Position[0] = 4*Cos(r/3.0); Position[1] = 3.0; Position[2] = 4*Sin(r/3.0);

   // lighting colors/types
   Ambient[0] = 0.1; Ambient[1] = 0.12; Ambient[2] = 0.15; Ambient[3] = 1.0;
   Diffuse[0] = 0.75; Diffuse[1] = 0.75; Diffuse[2] = 0.6; Diffuse[3] = 1.0;
   Specular[0] = 0.7; Specular[1] = 0.7; Specular[2] = 1.0; Specular[3] = 1.0;
   shininess[0] = 64;

   // normally normalize normals
   glEnable(GL_NORMALIZE);

   // enable lighting
   glEnable(GL_LIGHTING);

   // set light model with ciewer location for specular lights
   glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, 1);

   glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
   glEnable(GL_COLOR_MATERIAL);

   // enable the light and position it
   glEnable(GL_LIGHT0);
   glLightfv(GL_LIGHT0, GL_AMBIENT, Ambient);
   glLightfv(GL_LIGHT0, GL_DIFFUSE, Diffuse);
   glLightfv(GL_LIGHT0, GL_SPECULAR, Specular);
   glLightfv(GL_LIGHT0, GL_POSITION, Position);


   ///////////////////////////

   float white[] = {1.0, 1.0, 1.0, 1.0};
   //float emission[] = {0.0, 0.4, 0.9, 1.0};

   glMaterialfv(GL_FRONT, GL_SHININESS, shininess);
   glMaterialfv(GL_FRONT, GL_SPECULAR, white);
   //glMaterialfv(GL_FRONT, GL_EMISSION, emission);

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

   glDisable(GL_LIGHTING);
   glColor3f(1.0,1.0,1.0);
   ball(Position[0], Position[1], Position[2], 0.125);

   r = glutGet(GLUT_ELAPSED_TIME)*rate;
   r = fmod(r, 360*24*18.4);
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
   //adjust projection
   if (mode == 0)
      glOrtho(-2*w2h, 2*w2h, -2, 2, -2, 2);
   else //(mode == 1 or 2)
      gluPerspective(60, w2h, 5/4, 5*4);

   //switch back to model matrix
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
}

void special(int key, int mousex, int mousey)
{
   switch(key)
   {
      case GLUT_KEY_UP:    // in Perspective vs FP mode
         if (mode == 2)    // up and down intuitively mean
            ph -= 5;       // opposite directions of rotation
         else              // in this implementation
            ph += 5;
         break;
      case GLUT_KEY_DOWN:
         if (mode == 2)
            ph += 5;
         else
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
//      case '.':
//         rate *= 2;
//         break;
//      case ',':
//         rate /= 2;
//         break;
      case 'm':
         mode += 1;
         mode %= 3;
         if (mode == 0)
            mode = 1;
         reshape(w, h);
         break;
      if(mode == 2)
      {
         case 'w': //move forward
            ex -= (ex-vx)/8;
            ey -= (ey-vy)/8;
            ez -= (ez-vz)/8;
            break;
         case 's': //move back
            ex += (ex-vx)/8;
            ey += (ey-vy)/8;
            ez += (ez-vz)/8;
            break;
         case 'a': //strafe right
            ex -= (ez-vz)/8;
            ez += (ex-vx)/8;
            break;
         case 'd': //strafe left
            ex += (ez-vz)/8;
            ez -= (ex-vx)/8;
            break;
      }
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
