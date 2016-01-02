/*
 *  Draped Textures
 *
 *  Combines a large texture with a quadrangles to make a simple terrain viewer.
 *
 *  Key bindings:
 *  m           Cycle display modes
 *  +/-         Increase/decrease vertical exaggeration
 *  a           Toggle axes
 *  arrows      Change view angle
 *  PgDn/PgUp   Zoom in and out
 *  LeftMouse   Pan horizontally
 *  RightMouse  Pan vertically
 *  0           Reset view angle
 *  ESC         Exit
 */
#include "CSCIx229.h"
int mode=0;            //  Display mode
int axes=1;            //  Display axes
int th=0;              //  Azimuth of view angle
int ph=90;             //  Elevation of view angle
double asp=1;          //  Aspect ratio
int dim=500;           //  Size of world
double Ox=0,Oy=0,Oz=0; //  LookAt Location
int move=0;            //  Move mode
int X,Y;               //  Last mouse location
float z[65][65];       //  DEM data
float zmin=+1e8;       //  DEM lowest location
float zmax=-1e8;       //  DEM highest location
float zmag=1;          //  DEM magnification

/*
 *  Draw scene
 */
void DEM()
{
   int i,j;
   double z0 = (zmin+zmax)/2;
   //  Apply texture to one large quad
   if (mode==0)
   {
      glColor3f(1,1,1);
      glEnable(GL_TEXTURE_2D);
      glBegin(GL_QUADS);
      glTexCoord2f(0,0); glVertex2d(-512,-512);
      glTexCoord2f(1,0); glVertex2d(+512,-512);
      glTexCoord2f(1,1); glVertex2d(+512,+512);
      glTexCoord2f(0,1); glVertex2d(-512,+512);
      glEnd();
      glDisable(GL_TEXTURE_2D);
   }
   //  Show DEM wire frame
   else if (mode==1)
   {
      glColor3f(1,1,0);
      for (i=0;i<64;i++)
         for (j=0;j<64;j++)
         {
            float x=16*i-512;
            float y=16*j-512;
            glBegin(GL_LINE_LOOP);
            glVertex3d(x+ 0,y+ 0,zmag*(z[i+0][j+0]-z0));
            glVertex3d(x+16,y+ 0,zmag*(z[i+1][j+0]-z0));
            glVertex3d(x+16,y+16,zmag*(z[i+1][j+1]-z0));
            glVertex3d(x+ 0,y+16,zmag*(z[i+0][j+1]-z0));
            glEnd();
         }
   }
   //  Apply texture to DEM wireframe
   else
   {
      glColor3f(1,1,1);
      glEnable(GL_TEXTURE_2D);
      glEnable(GL_DEPTH_TEST);
      glEnable(GL_CULL_FACE);
      for (i=0;i<64;i++)
         for (j=0;j<64;j++)
         {
            float x=16*i-512;
            float y=16*j-512;
            glBegin(GL_QUADS);
            glTexCoord2f((i+0)/64.,(j+0)/64.); glVertex3d(x+ 0,y+ 0,zmag*(z[i+0][j+0]-z0));
            glTexCoord2f((i+1)/64.,(j+0)/64.); glVertex3d(x+16,y+ 0,zmag*(z[i+1][j+0]-z0));
            glTexCoord2f((i+1)/64.,(j+1)/64.); glVertex3d(x+16,y+16,zmag*(z[i+1][j+1]-z0));
            glTexCoord2f((i+0)/64.,(j+1)/64.); glVertex3d(x+ 0,y+16,zmag*(z[i+0][j+1]-z0));
            glEnd();
         }
      glDisable(GL_CULL_FACE);
      glDisable(GL_DEPTH_TEST);
      glDisable(GL_TEXTURE_2D);
   }
}

/*
 *  OpenGL (GLUT) calls this routine to display the scene
 */
void display()
{
   //  Length of axes
   const double len=100;
   //  Eye position
   double Ex = Ox-2*dim*Sin(th)*Cos(ph);
   double Ey = Oy+2*dim        *Sin(ph);
   double Ez = Oz+2*dim*Cos(th)*Cos(ph);
   //  Erase the window and the depth buffer
   glClearColor(0.3,0.5,1.0,1);
   glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
   //  Set perspective
   glLoadIdentity();
   gluLookAt(Ex,Ey,Ez , Ox,Oy,Oz , 0,1,0);
   //  Draw scene
   glPushMatrix();
   glRotated(-90,1,0,0);
   DEM();
   //  Draw axes - no lighting from here on
   glColor3f(1,1,1);
   if (axes)
   {
      glBegin(GL_LINES);
      glVertex3d(Ox,Oy,Oz);
      glVertex3d(Ox+len,Oy,Oz);
      glVertex3d(Ox,Oy,Oz);
      glVertex3d(Ox,Oy+len,Oz);
      glVertex3d(Ox,Oy,Oz);
      glVertex3d(Ox,Oy,Oz+len);
      glEnd();
      //  Label axes
      glRasterPos3d(Ox+len,Oy,Oz);
      Print("X");
      glRasterPos3d(Ox,Oy+len,Oz);
      Print("Y");
      glRasterPos3d(Ox,Oy,Oz+len);
      Print("Z");
   }
   glPopMatrix();
   //  Display parameters
   glWindowPos2i(5,5);
   Print("Angle=%d,%d  Dim=%d  Vertical Magnification=%.1f",th,ph,dim,zmag);
   //  Render the scene and make it visible
   ErrCheck("display");
   glFlush();
   glutSwapBuffers();
}

/*
 *  GLUT calls this routine when an arrow key is pressed
 */
void special(int key,int x,int y)
{
   //  Right arrow key - increase angle by 5 degrees
   if (key == GLUT_KEY_RIGHT)
      th += 5;
   //  Left arrow key - decrease angle by 5 degrees
   else if (key == GLUT_KEY_LEFT)
      th -= 5;
   //  Up arrow key - increase elevation by 5 degrees
   else if (key == GLUT_KEY_UP && ph<90)
      ph += 5;
   //  Down arrow key - decrease elevation by 5 degrees
   else if (key == GLUT_KEY_DOWN && ph>0)
      ph -= 5;
   //  PageUp key - increase dim
   else if (key == GLUT_KEY_PAGE_DOWN)
      dim += 10;
   //  PageDown key - decrease dim
   else if (key == GLUT_KEY_PAGE_UP && dim>10)
      dim -= 10;
   //  Keep angles to +/-360 degrees
   th %= 360;
   ph %= 360;
   //  Update projection
   Project(60,asp,dim);
   //  Tell GLUT it is necessary to redisplay the scene
   glutPostRedisplay();
}

/*
 *  GLUT calls this routine when a key is pressed
 */
void key(unsigned char ch,int x,int y)
{
   //  Exit on ESC
   if (ch == 27)
      exit(0);
   //  Reset view angle
   else if (ch == '0')
   {
      th =  0;
      ph = 90;
      Ox = Oy = Oz = 0;
   }
   //  Toggle texture mode
   else if (ch == 'm')
      mode = (mode+1)%3;
   else if (ch == 'M')
      mode = (mode+2)%3;
   //  Toggle axes
   else if (ch == 'a' || ch == 'A')
      axes = 1-axes;
   //  Vertical magnification
   else if (ch == '+')
      zmag += 0.1;
   else if (ch == '-' && zmag>1)
      zmag -= 0.1;
   //  Reproject
   Project(60,asp,dim);
   //  Tell GLUT it is necessary to redisplay the scene
   glutPostRedisplay();
}

/*
 *  GLUT calls this routine when the window is resized
 */
void reshape(int width,int height)
{
   //  Ratio of the width to the height of the window
   asp = (height>0) ? (double)width/height : 1;
   //  Set the viewport to the entire window
   glViewport(0,0, width,height);
   //  Set projection
   Project(60,asp,dim);
}

/*
 *  GLUT calls this routine when a mouse is moved
 */
void motion(int x,int y)
{
   //  Do only when move is set
   //  WARNING:  this only works because by coincidence 1m = 1pixel
   if (move)
   {
      //  Left/right movement
      Ox += X-x;
      //  Near/far or Up/down movement
      if (move<0)
         Oy -= Y-y;
      else
         Oz += Y-y;
      //  Remember location
      X = x;
      Y = y;
      glutPostRedisplay();
   }
}

/*
 *  GLUT calls this routine when a mouse button is pressed or released
 */
void mouse(int key,int status,int x,int y)
{
   //  On button down, set 'move' and remember location
   if (status==GLUT_DOWN)
   {
      move = (key==GLUT_LEFT_BUTTON) ? 1 : -1;
      X = x;
      Y = y;
   }
   //  On button up, unset move
   else if (status==GLUT_UP)
      move = 0;
}

/*
 *  Read DEM from file
 */
void ReadDEM(char* file)
{
   int i,j;
   FILE* f = fopen(file,"r");
   if (!f) Fatal("Cannot open file %s\n",file);
   for (j=0;j<=64;j++)
      for (i=0;i<=64;i++)
      {
         if (fscanf(f,"%f",&z[i][j])!=1) Fatal("Error reading saddleback.dem\n");
         if (z[i][j] < zmin) zmin = z[i][j];
         if (z[i][j] > zmax) zmax = z[i][j];
      }
   fclose(f);
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
   glutCreateWindow("Draped Textures");
   glutFullScreen();
   //  Set callbacks
   glutDisplayFunc(display);
   glutReshapeFunc(reshape);
   glutSpecialFunc(special);
   glutKeyboardFunc(key);
   glutMouseFunc(mouse);
   glutMotionFunc(motion);
   //  Load texture
   LoadTexBMP("saddleback.bmp");
   //  Load DEM
   ReadDEM("saddleback.dem");
   //  Pass control to GLUT so it can interact with the user
   ErrCheck("init");
   glutMainLoop();
   return 0;
}
