// Final project - game.cpp
// Generates and plays a tower defense game
// Jordan Dick
// jordan.dick@colorado.edu

#ifndef STDIncludes
#define STDIncludes
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#ifdef __APPLE__
#include <OpenGL/glu.h>
#else
#include <GL/glu.h>
#endif
#endif

#include <iostream>

//#include "CSCIx229.h"
#include <SDL.h>
#include <SDL_opengl.h>
#include "structures.h"
#include "objects.h"

using namespace std;

//GLOBAL VARIABLES//
//running or not
bool quit = false;

//View Angles
double th = 0;
double ph = 40;
//Window Size
int w = 1920;
int h = 1080;

//perspective mode
int mode = 1;  // 0 = ortho, 1 = perspective, 2 = first person

//eye position and orientation
double ex = 0;
double ey = 0;
double ez = 0;

double vx = 0;
double vy = 0;
double vz = 0;

double zoom = 16;

//lighting arrays
float Ambient[4];
float Diffuse[4];
float Specular[4];
float shininess[1];
float Position[4]; 
float ltheta = 0.0;

//Textures
//unsigned int texture[5];


//SDL Window/OpenGL Context
SDL_Window* window = NULL;
SDL_GLContext context;

//Timing
int r = 0;
int dr = 0;
int oldr = 0;

//Game Objects
Floor F;
Enemy* enemies[32];
Tower* towers[32];
Bullet* bullets[128];

////////////////////

//////// SDL Init Function ////////

bool init()
{
   bool success = true;

   if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0)
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
//   if (SDL_GL_SetSwapInterval(1) < 0)
//   {
//      cerr << "SDL could not set Vsync: " << SDL_GetError() << endl;
//      success = false;
//   }

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
   if (mode == 1) //rotation for perspective mode
   {
      ex = Sin(-th)*Cos(ph)*zoom;
      ey = Sin(ph)*zoom;
      ez = Cos(-th)*Cos(ph)*zoom;

      gluLookAt(ex,ey,ez , 0,0,0 , 0,Cos(ph),0);
   }
   else //mode == 2              // rotation and movement for FP mode
   {                             // occur in keyboard & special
      vx = ex - Sin(th)*Cos(ph); // here we simply update
      vy = ey - Sin(ph);         // location of view target
      vz = ez - Cos(th)*Cos(ph);

      gluLookAt(ex,ey,ez , vx,vy,vz , 0,Cos(ph),0);
   }

//   glEnable(GL_TEXTURE_2D);
//   glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

   //////////Lighting//////////

   // Light position and rendered marker (unlit)

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

   glColor3f(0.0,1.0,1.0);
//   glBindTexture(GL_TEXTURE_2D, texture[0]);
   ball(0, 2, 0, 0.5); //Jupiter

   glColor3f(0.25,0.25,0.3);
   F.render();

   //glDisable(GL_TEXTURE_2D);
   glDisable(GL_LIGHTING);
   glColor3f(1.0,1.0,1.0);
   ball(Position[0], Position[1], Position[2], 0.125);

//   r = glutGet(GLUT_ELAPSED_TIME)*rate;
//   r = fmod(r, 360*24*18.4);
   glFlush();
   SDL_GL_SwapWindow(window);
//   glutSwapBuffers();
}

void physics()
{
   while (dr >= 17)
   {
      //set timing stuf
      oldr = r;
      dr -= 17;

      //actually do all the animation and physics
      ltheta += M_PI/60;
      ltheta = fmod(ltheta, 2*M_PI);
      Position[0] = 3.0*sin(ltheta);
      Position[2] = 3.0*cos(ltheta);
   }
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
   gluPerspective(60, w2h, 20/4, 20*4);

   //switch back to model matrix
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
}

void keyboard(const Uint8* state)
{
   if (state[SDL_SCANCODE_ESCAPE])
      quit = true;
   
   if (state[SDL_SCANCODE_LEFT])
      th += 5;
   if (state[SDL_SCANCODE_RIGHT])
      th -= 5;
   if (state[SDL_SCANCODE_UP])
      ph += 5;
   if (state[SDL_SCANCODE_DOWN])
      ph -= 5;

   if (state[SDL_SCANCODE_Z])
      zoom = max(1.0, zoom-1);
   if (state[SDL_SCANCODE_X])
      zoom = zoom+1;
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
   
   //compile shaders


   reshape(w,h);

   Position[0] = 0.0; Position[1] = 5.0; Position[2] = 3.0; Position[3] = 1.0;

   SDL_Event event;

   ////////Main Loop////////
   while (!quit)
   {
      while (SDL_PollEvent(&event))
      {
         switch(event.type)
         {
            case SDL_QUIT:
               cout << "quit command\n";
               quit = true;
               break;

            case SDL_KEYDOWN:
               if (mode == 1)
               {
                  const Uint8* state = SDL_GetKeyboardState(NULL);
                  keyboard(state);
               }
               break;
         }
      }
      r = SDL_GetTicks();
      dr = r - oldr;
      physics();
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
