// hw4 - scene.c
// manually draws various objects in a simple scene
// Jordan Dick
// jordan.dick@colorado.edu

#ifndef STDIncludes
#define STDIncludes
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#ifdef __APPLE__
#include <OpenGL/glu.h>
#else
#include <GL/GLU.h>
#endif
#endif

//#include "CSCIx229.h"
#include <SDL.h>
#include <SDL_opengl.h>
#include "objects.c"

using namespace std;

//GLOBAL VARIABLES//
//running or not
bool quit = false;

//View Angles
double th = 0;
double ph = 0;
//Window Size
int w = 600;
int h = 800;

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

//Textures
//unsigned int texture[5];


//SDL Window/OpenGL Context
SDL_Window* window = NULL;
SDL_GLContext context;

//Timing
double r = 0;
double dr = 0;

////////////////////

//////// SDL Init Function ////////

bool init()
{
   bool success = true;

   if (SDL_Init(SDL_INIT_VIDEO) != 0)
   {
      cerr << "SDL failed to initialize: " << SDL_GetError() << endl;
      success = false;
   }

   SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
   SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);

   window = SDL_CreateWindow("Jordan Dick - FinalTD", 0,0 , w,h, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
   if (window == NULL)
   {
      cerr << "SDL failed to create a window: " << SDL_GetError() << endl;
      success = false;
   }

   context = SDL_GL_CreateContext(window);
   if (context == NULL)
   {
      cerr << "SDL failed to create OpenGL context: " << SDL_GetError() << endl;
      success = false;
   }
   
   //Vsync
   if (SDL_GL_SetSwapInterval(1) < 0)
   {
      cerr << "SDL could not set Vsync: " << SDL_GetError() << endl;
      success = false;
   }

   return success;
}

///////////////////////////////////

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

   glEnable(GL_TEXTURE_2D);
   glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

   //////////Lighting//////////

   // Light position and rendered marker (unlit)
   //glDisable(GL_LIGHTING);
//   Position[0] = 4*Cos(r/3.0); Position[1] = 3.0; Position[2] = 4*Sin(r/3.0);

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
//   //float emission[] = {0.0, 0.4, 0.9, 1.0};
//
   glMaterialfv(GL_FRONT, GL_SHININESS, shininess);
   glMaterialfv(GL_FRONT, GL_SPECULAR, white);
//   //glMaterialfv(GL_FRONT, GL_EMISSION, emission);
//
   glColor3f(1.0,1.0,1.0);
//   glBindTexture(GL_TEXTURE_2D, texture[0]);
   sphere(0, 0, 0, 1.5*r, 0.5); //Jupiter
//
//   glPushMatrix();
//   glRotated(r/2, 0,1,0);
//   //glBindTexture(GL_TEXTURE_2D, texture[1]);
//   cube(1, 0, 0, r, 0.25); //IO
//   glPopMatrix();
//
//   glPushMatrix();
//   glRotated(r/4, 0,1,0);
//   //glBindTexture(GL_TEXTURE_2D, texture[2]);
//   octahedron(-2, 0, 0, 4.0/3.0*r, 0.25); //Europa
//   glPopMatrix();
//
//   glPushMatrix();
//   glRotated(r/8, 0,1,0);
//   //glBindTexture(GL_TEXTURE_2D, texture[3]);
//   dodecahedron(3, 0, 0, -1.125*r, 0.25); //Ganymede
//   glPopMatrix();
//
//   glPushMatrix();
//   glRotated(r/18.4, 0,1,0);
//   //glBindTexture(GL_TEXTURE_2D, texture[4]);
//   icosahedron(4, 0, 0, 0.75*r, 0.25); //Callisto
//   glPopMatrix();
//
//   glDisable(GL_TEXTURE_2D);
//   glDisable(GL_LIGHTING);
   glColor3f(1.0,1.0,1.0);
   ball(Position[0], Position[1], Position[2], 0.125);

//   r = glutGet(GLUT_ELAPSED_TIME)*rate;
//   r = fmod(r, 360*24*18.4);
   glFlush();
//   glutSwapBuffers();
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
   gluPerspective(60, w2h, 5/4, 5*4);

   //switch back to model matrix
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
}

void keyboard(const Uint8* state)
{
   if (state[SDLK_q])
      quit = true;
}

int main(int argc, char *argv[])
{
//   glutInit(&argc, argv);
//   glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
//
//   glutInitWindowPosition(0,0);
//   glutInitWindowSize(w,h);
//   glutCreateWindow("Jordan Dick: HW3 - Galilean Moons");
//
//   glutDisplayFunc(display);
//   glutReshapeFunc(reshape);
//   glutSpecialFunc(special);
//   glutKeyboardFunc(keyboard);
//   //glutPassiveMotionFunc(motion);
//   glutIdleFunc(idle);

//   SDL_StartTextInput();

   if (init() != true)
   {
      cerr << "Shutting Down\n";
      return 1;
   }
   
   reshape(w,h);

   SDL_Event event;

   ////////Main Loop////////
   while (!quit)
   {
      while (SDL_PollEvent(&event))
      {
         switch(event.type)
         {
            case SDL_QUIT:
               cout << "x button pressed\n";
               quit = true;
               break;

            case SDL_KEYDOWN || SDL_KEYUP:
               cout << "key pressed\n";
               if (mode == 1)
               {
                  const Uint8* state = SDL_GetKeyboardState(NULL);
                  keyboard(state);
               }
               break;
         }
      }

      display();
   }

   cout << "Shutting Down\n";
   SDL_Quit();
   

//   //load texture
////   texture[0] = LoadTexBMP("jupiter.bmp");
////   texture[1] = LoadTexBMP("mercury.bmp");
////   texture[2] = LoadTexBMP("venus.bmp");
////   texture[3] = LoadTexBMP("earth.bmp");
////   texture[4] = LoadTexBMP("mars.bmp");
//
//   //check for errors
////   ErrCheck("init");
//
//   glutMainLoop();

   return EXIT_SUCCESS;
}
