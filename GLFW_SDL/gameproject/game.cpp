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

//Shaders
int shader = 0;
int filter = 0;
unsigned int img;
int id;

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
Floor F;
Enemy* enemies[64] = {NULL};
Tower* towers[64] = {NULL};
Bullet* bullets[128] {NULL};

////////////////////

void reshape(int width, int height);

//////// SDL Init Function ////////

bool init()
{
   bool success = true;

   if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0)
   {
      cerr << "SDL failed to initialize: " << SDL_GetError() << endl;
      success = false;
   }

//   SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
//   SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);

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

void display()
{
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glEnable(GL_DEPTH_TEST);
   glEnable(GL_CULL_FACE);

   reshape(w,h);
   glMatrixMode(GL_MODELVIEW);
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
   Ambient[0] = 0.05; Ambient[1] = 0.07; Ambient[2] = 0.08; Ambient[3] = 1.0;
   Diffuse[0] = 0.75; Diffuse[1] = 0.75; Diffuse[2] = 0.70; Diffuse[3] = 1.0;
   Specular[0] = 0.7; Specular[1] = 0.7; Specular[2] = 1.0; Specular[3] = 1.0;
   shininess[0] = 1024;

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
   float emission[] = {0.0, 0.4, 0.25, 1.0};

   glMaterialfv(GL_FRONT, GL_SHININESS, shininess);
   glMaterialfv(GL_FRONT, GL_SPECULAR, white);
   glMaterialfv(GL_FRONT, GL_EMISSION, emission);

   // Use Custom Shader
   glUseProgram(shader);

   //Draw stuff
   //glColor3f((float)rand()/(float)RAND_MAX,(float)rand()/(float)RAND_MAX,(float)rand()/(float)RAND_MAX);
   glColor3f(0.0,0.8,0.8);
//   glBindTexture(GL_TEXTURE_2D, texture[0]);
   //octahedron(0,2,0, Position[0], 0.75);
   sphere(0,2,0, Position[0], 0.75);

   glColor3f(0.25,0.25,0.30);
   emission[0] = 0.0; emission[1] = 0.0; emission[2] = 0.0;
   glMaterialfv(GL_FRONT, GL_EMISSION, emission);
   F.render();

   for(int i=0; i<64; ++i)
   {
      if (enemies[i] != NULL)
         enemies[i]->render();
   }
   for(int i=0; i<64; ++i)
   {
      if (towers[i] != NULL)
         towers[i]->render();
   }
   for(int i=0; i<128; ++i)
   {
      if (bullets[i] != NULL)
         bullets[i]->render();
   }

   //Stop using Custom Shader
   glUseProgram(0);

   //glDisable(GL_TEXTURE_2D);
   glDisable(GL_LIGHTING);
   glColor3f(1.0,1.0,1.0);
   ball(Position[0], Position[1], Position[2], 0.125);

   //Apply Blur Filter
   glUseProgram(filter);

   //set offsets
   id = glGetUniformLocation(filter,"dX");
   if (id >= 0) glUniform1f(id,1.0/w);
   id = glGetUniformLocation(filter,"dY");
   if (id >= 0) glUniform1f(id,1.0/h);
   id = glGetUniformLocation(filter,"img");
   if (id >= 0) glUniform1i(id,0);

   glDisable(GL_DEPTH_TEST);
   glDisable(GL_CULL_FACE);

   glMatrixMode(GL_PROJECTION);
   //glPushMatrix();
   glLoadIdentity();
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();

   for (int l=0; l<0; ++l)
   {
      glBindTexture(GL_TEXTURE_2D,img);
      glCopyTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,0,0,w,h,0);

      glClear(GL_COLOR_BUFFER_BIT);

      glEnable(GL_TEXTURE_2D);
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

   }
   glUseProgram(0);

   //swap the buffers
   glFlush();
   SDL_GL_SwapWindow(window);
}

void physics()
{
   while (dr >= 16)
   {
      if (!pause)
      {
         //move the light
         ltheta += M_PI/60;
         ltheta = fmod(ltheta, 2*M_PI);
         Position[0] = 4.5*sin(ltheta);
         Position[2] = 4.5*cos(ltheta);

         //Manage the Spawning of the Waves
         int newenemy = F.animate();
         if (newenemy)
         {
            int i = 0;
            while (enemies[i]!=NULL)
               ++i;
            enemies[i] = new Enemy(-8,6, F.currentwave==0 ? 25 : 25*F.currentwave, newenemy);
         }

         //animate the enemies
         for (int i=0; i<64; ++i)
         {
            if (enemies[i] != NULL)
            {
               //cout << "animating enemy " << i << " at address " << enemies[i] << endl;
               enemies[i]->animate();
            
               if (enemies[i]->x == 8.0 && enemies[i]->y == 6.0)
               {
                  // subtract player lives
                  delete enemies[i];
                  enemies[i] = NULL;
               }
            }
         }
         //animate the towers
         for (int i=0; i<64; ++i)
         {
            if (towers[i] != NULL)
            {
               //select closest target
               float dist = INFINITY;
               for (int j=0; j<64; ++j)
               {
                  if (enemies[j] != NULL)
                  {
                     if (dist > towers[i]->distance(&enemies[j]))
                     {
                        dist = towers[i]->distance(&enemies[j]);
                        if (dist <= towers[i]->range)
                           towers[i]->target = &enemies[j];
                        else
                           towers[i]->target = NULL;
                     }
                  }
               }
               if (dist == INFINITY)
                  towers[i]->target = NULL;
               //manage firing loop
               if (towers[i]->cooldown < towers[i]->maxcooldown)
                  towers[i]->cooldown += dr;
               if (towers[i]->target != NULL && towers[i]->cooldown >= towers[i]->maxcooldown)
               {
                  Bullet* bullet = towers[i]->fire();
                  int k = 0;
                  while (bullets[k] != NULL)
                     ++k;
                  bullets[k] = bullet;
                  //cout << bullets[i] << " Target acquired: " << bullet->target << endl;
                  towers[i]->cooldown -= towers[i]->maxcooldown;
               }
               towers[i]->animate();
            }
         }
         for (int i=0; i<128; ++i)
         {
            if (bullets[i] != NULL)
            {
               //cout << "animating bullet " << i << " at address " << bullets[i] << endl;
               //cout << bullets[i]->target << " " << *(bullets[i]->target) << endl;
               if (*(bullets[i]->target) == NULL)
               {
                  //cout << bullets[i] << " Target lost...\n";
                  delete bullets[i];
                  bullets[i] = NULL;
               }
               else
               {
                  bullets[i]->animate();
                  if (bullets[i]->distance() <= 0.5)
                  {
                     //cout << bullets[i] << " Target hit! " << bullets[i]->target << endl;
                     //(*(bullets[i]->target))->damage(bullets[i]->dmg);
                     bullets[i]->collide();
                     //cout << (*bullets[i]->target)->health << endl;;
                     if ((*bullets[i]->target)->health <= 0)
                     {
                        //cout << bullets[i]->target << " " << *(bullets[i]->target) << endl;
                        //cout << "ded\n";
                        delete (*bullets[i]->target);
                        *(bullets[i]->target) = NULL;
                        //cout << bullets[i]->target << " " << *(bullets[i]->target) << endl;
                     }
                     delete bullets[i];
                     bullets[i] = NULL;
                  }
               }
            }
         }
      }

      //set timing stuf
      oldr = r;
      dr -= 16;
   }
}

// this function stolen from ex27
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

// this function stolen from ex27
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

// this function stolen from ex27
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

   //if (state[SDL_SCANCODE_SPACE])
   //   pause = 1-pause;
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
   shader = CreateShaderProg((char*)"pixlight.vert",(char*)"pixlight.frag");
   filter = CreateShaderProg(NULL, (char*)"gaussian.frag");
   
   //create and configure  texture for blur filter
   glGenTextures(1,&img);
   glBindTexture(GL_TEXTURE_2D,img);
   glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);

   reshape(w,h);

   Position[0] = 0.0; Position[1] = 6.0; Position[2] = 4.5; Position[3] = 1.0;

   SDL_Event event;

//   enemies[0] = new Enemy(-8, 6, 100, 0);
//   enemies[1] = new Enemy(-2, 2, 100, 1);
//   towers[0] = new Tower(0, 4);

   int startuptime = SDL_GetTicks();

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
               if (event.key.keysym.scancode == SDL_SCANCODE_SPACE)
                  pause = 1 - pause;
               else if (event.key.keysym.scancode == SDL_SCANCODE_E)
               {
                  if (F.currentwave < 6)
                     F.currentwave = 5;
                  F.spawnwave();

                  if (towers[0] == NULL)
                     towers[0] = new Tower(0.0, 4.0);
                  if (towers[1] == NULL)
                     towers[1] = new Tower(-4.0, -4.0);
                  if (towers[2] == NULL)
                     towers[2] = new Tower(4.0, -4.0);
//                  else
//                  {
//                     delete towers[0];
//                     towers[0] = NULL;
//                  }
               }
               else if (mode == 1)
               {
                  const Uint8* state = SDL_GetKeyboardState(NULL);
                  keyboard(state);
               }
               break;
            case SDL_WINDOWEVENT:
               switch(event.window.event)
               {
               case SDL_WINDOWEVENT_SIZE_CHANGED:
                  //cerr << event.window.data1 << " " << event.window.data2 << endl;
                  reshape(event.window.data1, event.window.data2);
                  break;
               }
               break;
         }
      }
      r = SDL_GetTicks();
      dr = r - oldr;
      physics();
      display();
      frames += 1;
   }

   cout << "Shutting Down\n";
   cout << "average framerate: " << 1000*(float)frames/(r - startuptime) << endl;
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
