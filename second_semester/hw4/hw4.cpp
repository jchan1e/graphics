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
#include "objects.h"

using namespace std;

//GLOBAL VARIABLES//
//running or quit
bool quit = false;

//View Angles
double th =  0.0;
double ph =  0.0;
double dth = 0.0;
double dph = 0.0;
//Window Size
int w = 1920;
int h = 1080;
//View Mode
int perspective = 1;

//eye position and orientation
double ex = 0.0;
double ey = 0.0;
double ez = 0.0;

double vx = 0.0;
double vy = 0.0;
double vz = 0.0;
double zoom = 4.0;
double dzoom = 0.0;

//lighting arrays
float Ambient[4];
float Diffuse[4];
float Specular[4];
float shininess[1];
float Position[4]; 
float ltheta = 0.0;

//Shaders
int mode = 0;
int pixlit = 0;
int shader = 0;
float C = 0.0;
float D = 0.0;

//SDL Window/OpenGL Context
SDL_Window* window = NULL;
SDL_GLContext context;

//Timing
int r = 0;
int dr = 0;
int oldr = 0;
int pause = 0;
int frames = 0;

//Game Objects
float t = 0.0;
float dt = 0.0625;

//Vertex Arrays
int sphere_size = 6*2592;
unsigned int SphereBuf;
float SpherePoints[6*2592][12]; //12 floats/vertex, 6 vertices/quad, 2592 quads/sphere
float* SphereData; //12 floats/vertex, 6 vertices/quad, 2592 quads/sphere

////////////////////
//functions that are called ahead of when they're defined
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

   window = SDL_CreateWindow("Jordan Dick - Procedural Textures", 0,0 , w,h, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
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
   if (SDL_GL_SetSwapInterval(0) < 0)
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

   reshape(w,h);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();

   //view angle
   if (perspective)
   {
      ex = Sin(-th)*Cos(ph)*zoom;
      ey = Sin(ph)*zoom;
      ez = Cos(-th)*Cos(ph)*zoom;
      gluLookAt(ex,ey,ez , 0,0,0 , 0,Cos(ph),0);
   }
   else
   {
      glRotatef(ph, 1,0,0);
      glRotatef(th, 0,1,0);
      glScalef(4/zoom, 4/zoom, 4/zoom);
   }

   //////////Lighting//////////

   // Light position and rendered marker (unlit)

   // lighting colors/types
   Ambient[0] = 0.15; Ambient[1] = 0.17; Ambient[2] = 0.20; Ambient[3] = 1.0;
   Diffuse[0] = 0.75; Diffuse[1] = 0.75; Diffuse[2] = 0.70; Diffuse[3] = 1.0;
   Specular[0] = 0.6; Specular[1] = 0.6; Specular[2] = 0.65; Specular[3] = 1.0;
   shininess[0] = 128;

   // normally normalize normals
   glEnable(GL_NORMALIZE);

   // enable lighting
   glEnable(GL_LIGHTING);

   // set light model with ciewer location for specular lights
   glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, 1);

   glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
   glEnable(GL_COLOR_MATERIAL);

   // enable the light and position it
   //glEnable(GL_LIGHT0);
   //glLightfv(GL_LIGHT0, GL_AMBIENT, Ambient);
   //glLightfv(GL_LIGHT0, GL_DIFFUSE, Diffuse);
   //glLightfv(GL_LIGHT0, GL_SPECULAR, Specular);
   //glLightfv(GL_LIGHT0, GL_POSITION, Position);

   ///////////////////////////

   //float white[] = {1.0, 1.0, 1.0, 1.0};
   //float emission[] = {0.0, 0.2, 0.15, 1.0};

   //glMaterialfv(GL_FRONT, GL_SHININESS, shininess);
   //glMaterialfv(GL_FRONT, GL_SPECULAR, white);
   //glMaterialfv(GL_FRONT, GL_EMISSION, emission);

   // Use hw3 shader
   glUseProgram(shader);
   float ModelView[16];
   float Projection[16];
   glGetFloatv(GL_PROJECTION_MATRIX, Projection);
   glGetFloatv(GL_MODELVIEW_MATRIX, ModelView);

   int id;
   id = glGetUniformLocation(shader, "C");
   if (id >= 0) glUniform1f(id, C);
   id = glGetUniformLocation(shader, "D");
   if (id >= 0) glUniform1f(id, D);
   id = glGetUniformLocation(shader, "ModelViewMatrix");
   if (id >= 0) glUniformMatrix4fv(id, 1, GL_FALSE, ModelView);
   id = glGetUniformLocation(shader, "ProjectionMatrix");
   if (id >= 0) glUniformMatrix4fv(id, 1, GL_FALSE, Projection);
   id = glGetUniformLocation(shader, "LightPos");
   if (id >= 0) glUniform4fv(id, 4, Position);
   id = glGetUniformLocation(shader, "Ambient");
   if (id >= 0) glUniform4fv(id, 4, Ambient);
   id = glGetUniformLocation(shader, "Diffuse");
   if (id >= 0) glUniform4fv(id, 4, Diffuse);
   id = glGetUniformLocation(shader, "Specular");
   if (id >= 0) glUniform4fv(id, 4, Specular);
   id = glGetUniformLocation(shader, "shininess");
   if (id >= 0) glUniform1f(id, shininess[0]);

   glBindBuffer(GL_ARRAY_BUFFER, SphereBuf);
   glEnableVertexAttribArray(0);
   glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 12*sizeof(float), (void*)0);
   glEnableVertexAttribArray(1);
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12*sizeof(float), (void*)(4*sizeof(float)));
   glEnableVertexAttribArray(2);
   glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 12*sizeof(float), (void*)(7*sizeof(float)));
   glEnableVertexAttribArray(3);
   glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 12*sizeof(float), (void*)(10*sizeof(float)));
   
   //sphere( 0.0, 0.0, 0.0, 180.0, 2.0);
   glDrawArrays(GL_TRIANGLES, 0, sphere_size);
   
   glDisableVertexAttribArray(0);
   glDisableVertexAttribArray(1);

   //Stop Using Custom Shader
   glUseProgram(0);

   //glDisable(GL_TEXTURE_2D);
   glDisable(GL_LIGHTING);
   glColor3f(1.0,1.0,1.0);
   ball(Position[0], Position[1], Position[2], 0.125);

   //cout << gluErrorString(glGetError()) << endl;

   //swap the buffers
   glFlush();
   SDL_GL_SwapWindow(window);
}

void physics()
{
   while (dr >= 16)
   {
      const Uint8* state = SDL_GetKeyboardState(NULL);
      keyboard(state);

      //adjust the eye position
      th += dth;
      ph += dph;
      zoom = zoom+dzoom<2.1?2.1:zoom+dzoom;

      if (!pause)
      {
         //move the light

         t += dt;
         t = fmod(t, 180.0);

         //Position[0] = 4.5*Sin(4*t);
         //Position[2] = 4.5*Cos(4*t);

         //C = 0.7*Cos(t) - 0.3;
         //D = 0.7*Sin(t);
         C =-0.5*(Sin(t) - Sin(3*t)) + 0.25;
         D = 0.5*(Cos(t) - Cos(3*t));
      }

      //Timing Variables
      dr -= 16;
   }
   oldr = r;
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
   if (perspective)
      gluPerspective(60, w2h, 1.0/32.0, 32.0);
   else
      glOrtho(-4*w2h, 4*w2h, -4, 4, -4, 4);

   //switch back to model matrix
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
}

// Per frame keyboard input here, per keypress input in main()
void keyboard(const Uint8* state)
{
   if (state[SDL_SCANCODE_ESCAPE])
      quit = true;
   
   if (state[SDL_SCANCODE_LEFT])
      dth = 0.5;
   else if (state[SDL_SCANCODE_RIGHT])
      dth = -0.5;
   else
      dth = 0;
   if (state[SDL_SCANCODE_UP])
      dph = 0.5;
   else if (state[SDL_SCANCODE_DOWN])
      dph = -0.5;
   else
      dph = 0;
   if (state[SDL_SCANCODE_Z])
      dzoom = -0.10;
   else if (state[SDL_SCANCODE_X])
      dzoom = 0.10;
   else
      dzoom = 0;
}

int main(int argc, char *argv[])
{
   //Initialize
   if (init() != true)
   {
      cerr << "Failed to initialize SDL\n";
      return 2;
   }

   //initialize sphere vertex array
   sphere(SpherePoints, 0.15, 0.75, 0.90);
   SphereData = &SpherePoints[0][0];
   
   glGenBuffers(1,&SphereBuf);
   glBindBuffer(GL_ARRAY_BUFFER, SphereBuf);
   glBufferData(GL_ARRAY_BUFFER, sizeof(SphereData), SphereData, GL_STATIC_DRAW);
   glBindBuffer(GL_ARRAY_BUFFER, 0);
   
   //compile shaders
   //pixlit = CreateShaderProg((char*)"pixlight.vert",(char*)"pixlight.frag");
   shader = CreateShaderProg((char*)"hw4.vert", (char*)"hw4.frag");
   
   reshape(w,h);

   Position[0] = -4.5; Position[1] = 4.5; Position[2] = 4.5; Position[3] = 1.0;

   SDL_Event event;

   int startuptime = oldr = SDL_GetTicks();

   ////////Main Loop////////
   while (!quit)
   {
      while (SDL_PollEvent(&event))
      {
         switch(event.type)
         {
            case SDL_QUIT:
               quit = true;
               break;

            case SDL_KEYDOWN:
               if (event.key.keysym.scancode == SDL_SCANCODE_SPACE)
                  pause = 1 - pause;
               else if (event.key.keysym.scancode == SDL_SCANCODE_0)
               {
                  th = 0;
                  ph = 0;
               }
               //else if (event.key.keysym.scancode == SDL_SCANCODE_M)
               //   mode = 1 - mode;
               else if (event.key.keysym.scancode == SDL_SCANCODE_N)
                  perspective = 1 - perspective;
               else if (event.key.keysym.scancode == SDL_SCANCODE_PERIOD)
                  dt *= 2.0;
               else if (event.key.keysym.scancode == SDL_SCANCODE_COMMA)
                  dt *= 0.5;
               break;

            case SDL_WINDOWEVENT:
               if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
               {
                  //cerr << event.window.data1 << " " << event.window.data2 << endl;
                  reshape(event.window.data1, event.window.data2);
               }
               break;
         }
      }
      //// PHYSICS TIMING ////
      r = SDL_GetTicks();
      dr += r - oldr;
      physics();
      display();
      frames += 1;
   }

   cout << "Shutting Down\n";
   cout << "average framerate: " << 1000*(float)frames/(r - startuptime) << endl;
   //cout << glGetString(GL_VERSION) << endl;

   SDL_Quit();

   return 0;
}
