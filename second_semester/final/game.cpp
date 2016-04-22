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

using namespace std;

//GLOBAL VARIABLES//
//running or not
bool quit = false;

//Window Size
int w = 800;
int h = 800;

//play mode
int mode = 1;  // 1 = play, 0 = place

//tower placement cursor position

//eye position and orientation

//lighting arrays

//Textures
//unsigned int texture[5];

//Shaders
unsigned int img;
int id;

//SDL Window/OpenGL Context
SDL_Window* window = NULL;
SDL_GLContext context;

//Timing
int r = 0;
int dr = 0;
int oldr = 0;
//int frames = 0;

//Game Objects

////////////////////
//functions that are called ahead of when they're defined
//because C

void reshape(int width, int height);
void keyboard(const Uint8* state);

//////// SDL Init Function ////////

bool init()
{
   bool success = true;

   if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0)
   {
      cerr << "SDL failed to initialize: " << SDL_GetError() << endl;
      success = false;
   }

   window = SDL_CreateWindow("Jordan Dick - FinalTD", 0,0 , w,h, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
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


void display(float* target)
{
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   reshape(w,h);

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();

   glEnable(GL_TEXTURE_2D);

   glBindTexture(GL_TEXTURE_2D, img);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 28, 28, 0, GL_RED, GL_FLOAT, target);

   glBegin(GL_QUADS);
   glTexCoord2f(0,0);
   glVertex2f(-1,-1);

   glTexCoord2f(1,0);
   glVertex2f(1,-1);

   glTexCoord2f(1,1);
   glVertex2f(1,1);

   glTexCoord2f(0,1);
   glVertex2f(-1,1);
   glEnd();

   glDisable(GL_TEXTURE_2D);

   //swap the buffers
   glFlush();
   SDL_GL_SwapWindow(window);
}

// this function stolen from class examples
char* ReadText(char* file)
{
   int n;
   char* buffer;
   FILE* f = fopen(file,"r");
   if (!f) {cerr << "Cannot open text file " << file << endl; quit = true;}
   fseek(f, 0, SEEK_END);
   n = ftell(f);
   rewind(f);
   buffer = (char*) malloc(n+1);
   if (!buffer) {cerr << "Cannot allocate " << n+1 << " bytes for text file " << file << endl; quit = true;}
   int h = fread(buffer, n, 1, f);
   if (h != 1) {cerr << h << " Cannot read " << n << " bytes for text file " << file << endl; quit = true;}
   buffer[n] = 0;
   fclose(f);
   return buffer;
}

// this function stolen from class examples
int CreateShader(GLenum type, char* file)
{
   // Create the shader
   int shader = glCreateShader(type);
   // Load source code from file
   char* source = ReadText(file);
   glShaderSource(shader, 1, (const char**) &source, NULL);
   free(source);
   // Compile the shader
   fprintf(stderr, "Compile %s\n", file);
   glCompileShader(shader);
   // Return name (int)
   return shader;
}

// this function stolen (mostly) from class examples
int CreateShaderProg(char* VertFile, char* FragFile)
{
   // Create program
   int prog = glCreateProgram();
   // Create and compile vertex and fragment shaders
   int vert, frag;
   if (VertFile) vert = CreateShader(GL_VERTEX_SHADER,  VertFile);
   if (FragFile) frag = CreateShader(GL_FRAGMENT_SHADER,FragFile);
   // Attach vertex and fragment shaders
   if (VertFile) glAttachShader(prog,vert);
   if (FragFile) glAttachShader(prog,frag);
   // Link Program
   glLinkProgram(prog);
   // Return name (int)
   return prog;
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
   gluPerspective(60, w2h, 0.5, 20*4);

   //switch back to model matrix
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
}

void draw_a_dot(float* arr, float x, float y)
{
   float radius = 2.0;
   for (int i=0; i < 28; ++i)
   {
      for (int j=0; j < 28; ++j)
      {
         float dist = sqrt((i-x)*(i-x) + (j-y)*(j-y));
         float value = min(radius/2.0 - dist/2.0, 1.0);
         arr[i*28 + j] = max(arr[i*28 + j], value);
      }
   }
}

int main(int argc, char *argv[])
{
   //Initialize
   if (init() != true)
   {
      cerr << "Shutting Down\n";
      return 1;
   }

   glGenTextures(1, &img);
   glBindTexture(GL_TEXTURE_2D, img);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

   float* target = (float*)malloc(28*28*sizeof(float));
   memset(target, 0.0, 28*28*sizeof(float));

   SDL_Event event;

   ////////Main Loop////////
   while (!quit)
   {
      //// PHYSICS TIMING ////
      r = SDL_GetTicks();
      dr += r - oldr;

      while (dr >= 16)
      {
         while (SDL_PollEvent(&event))
         {
            int x, y;
            switch(event.type)
            {
               case SDL_QUIT:
                  quit = true;
                  break;

               case SDL_WINDOWEVENT:
                  if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
                  {
                     reshape(event.window.data1, event.window.data2);
                  }
                  break;

               case SDL_KEYDOWN:
                  if (event.key.keysym.scancode == SDL_SCANCODE_ESCAPE)
                     quit = true;
                  break;

               case SDL_MOUSEBUTTONDOWN:
                  if (SDL_GetMouseState(&x, &y) & SDL_BUTTON(SDL_BUTTON_LEFT))
                     draw_a_dot(target, (w-(float)y)*28/w, (float)x*28/h);
                  break;

               case SDL_MOUSEMOTION:
                  if (SDL_GetMouseState(&x, &y) & SDL_BUTTON(SDL_BUTTON_LEFT))
                     draw_a_dot(target, (w-(float)y)*28/w, (float)x*28/h);
                  break;
            }
         }
         dr -= 16;
      }
      oldr = r;
      display(target);
   }

   //cout << "Shutting Down\n";
   free(target);
   SDL_Quit();

   return 0;
}
