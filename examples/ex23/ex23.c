/*
 *  Mixed transparent and opaque objects
 *
 *  Demonstrates how to draw transparent and opaque objects in the same scene
 *
 *  Key bindings:
 *  m/M        Toggle oredered/unordered drawing
 *  l/L        Toggle lighting on/off
 *  1          Toggle blend function destination GL_ONE_MINUS_SRC_ALPHA/GL_ONE
 *  +/-        Increase/decrease alpha
 *  a          Toggle axes
 *  arrows     Change view angle
 *  PgDn/PgUp  Zoom in and out
 *  0          Reset view angle
 *  ESC        Exit
 */
#include "CSCIx229.h"
int axes=0;       //  Display axes
int mode=1;       //  Projection mode
int th=0;         //  Azimuth of view angle
int ph=0;         //  Elevation of view angle
int zh=0;         //  Azimuth of light
int fov=55;       //  Field of view (for perspective)
double asp=1;     //  Aspect ratio
double dim=3.0;   //  Size of world
double alpha=0.75; //  Alpha
int    light=1;    //  Lighting
int    aone=1;     //  Alpha-one

/*
 *  Draw a ball
 *     at (x,y,z)
 *     radius r
 */
static void ball(double x,double y,double z,double r)
{
   //  Save transformation
   glPushMatrix();
   //  Offset, scale and rotate
   glTranslated(x,y,z);
   glScaled(r,r,r);
   //  White ball
   glColor3f(1,1,1);
   glutSolidSphere(1.0,16,16);
   //  Undo transofrmations
   glPopMatrix();
}

/*
 *  Draw a cube
 *     at (x,y,z)
 *     opaque or "stained glass"
 */
static void cube(double x,double y,double z,int opaque)
{
   //  Save transformation
   glPushMatrix();
   //  Offset
   glTranslated(x,y,z);
   glScaled(0.8,0.8,0.8);
   //  Set transparency
   if (!opaque)
   {
      glEnable(GL_BLEND);
      glColor4f(1,1,1,alpha);
      if (aone)
         glBlendFunc(GL_SRC_ALPHA,GL_ONE);
      else
         glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
      glDepthMask(0);
      glEnable(GL_TEXTURE_2D);
   }
   //  Cube
   glBegin(GL_QUADS);
   //  Front
   if (opaque) glColor3f(1,0,0);
   glNormal3f(0,0,+1);
   glTexCoord2f(0,0); glVertex3f(-1,-1,+1);
   glTexCoord2f(1,0); glVertex3f(+1,-1,+1);
   glTexCoord2f(1,1); glVertex3f(+1,+1,+1);
   glTexCoord2f(0,1); glVertex3f(-1,+1,+1);
   //  Back
   if (opaque) glColor3f(0,0,1);
   glNormal3f(0,0,-1);
   glTexCoord2f(0,0); glVertex3f(+1,-1,-1);
   glTexCoord2f(1,0); glVertex3f(-1,-1,-1);
   glTexCoord2f(1,1); glVertex3f(-1,+1,-1);
   glTexCoord2f(0,1); glVertex3f(+1,+1,-1);
   //  Right
   if (opaque) glColor3f(1,1,0);
   glNormal3f(+1,0,0);
   glTexCoord2f(0,0); glVertex3f(+1,-1,+1);
   glTexCoord2f(1,0); glVertex3f(+1,-1,-1);
   glTexCoord2f(1,1); glVertex3f(+1,+1,-1);
   glTexCoord2f(0,1); glVertex3f(+1,+1,+1);
   //  Left
   if (opaque) glColor3f(0,1,0);
   glNormal3f(-1,0,0);
   glTexCoord2f(0,0); glVertex3f(-1,-1,-1);
   glTexCoord2f(1,0); glVertex3f(-1,-1,+1);
   glTexCoord2f(1,1); glVertex3f(-1,+1,+1);
   glTexCoord2f(0,1); glVertex3f(-1,+1,-1);
   //  Top
   if (opaque) glColor3f(0,1,1);
   glNormal3f(0,+1,0);
   glTexCoord2f(0,0); glVertex3f(-1,+1,+1);
   glTexCoord2f(1,0); glVertex3f(+1,+1,+1);
   glTexCoord2f(1,1); glVertex3f(+1,+1,-1);
   glTexCoord2f(0,1); glVertex3f(-1,+1,-1);
   //  Bottom
   if (opaque) glColor3f(1,0,1);
   glNormal3f(0,-1,0);
   glTexCoord2f(0,0); glVertex3f(-1,-1,-1);
   glTexCoord2f(1,0); glVertex3f(+1,-1,-1);
   glTexCoord2f(1,1); glVertex3f(+1,-1,+1);
   glTexCoord2f(0,1); glVertex3f(-1,-1,+1);
   //  End
   glEnd();
   if (!opaque)
   {
      glDisable(GL_BLEND);
      glDepthMask(1);
      glDisable(GL_TEXTURE_2D);
   }
   //  Undo transofrmations
   glPopMatrix();
}

/*
 *  OpenGL (GLUT) calls this routine to display the scene
 */
void display()
{
   int i,j,k;
   const double len=1.5;  //  Length of axes
   double Ex = -2*dim*Sin(th)*Cos(ph);
   double Ey = +2*dim        *Sin(ph);
   double Ez = +2*dim*Cos(th)*Cos(ph);
   //  Erase the window and the depth buffer
   glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
   //  Enable Z-buffering in OpenGL
   glEnable(GL_DEPTH_TEST);
   //  Undo previous transformations
   glLoadIdentity();
   //  Perspective - set eye position
   gluLookAt(Ex,Ey,Ez , 0,0,0 , 0,Cos(ph),0);
   //  Light switch
   if (light)
   {
      //  Translate intensity to color vectors
      float Ambient[]   = {0.3,0.3,0.3,1.0};
      float Diffuse[]   = {1,1,1,1};
      float Specular[]  = {1,1,1,1};
      float white[]     = {1,1,1,1};
      //  Light direction
      float Position[]  = {5*Cos(zh),0,5*Sin(zh),1};
      //  Draw light position as ball (still no lighting here)
      ball(Position[0],Position[1],Position[2] , 0.1);
      //  Enable lighting with normalization
      glEnable(GL_LIGHTING);
      glEnable(GL_NORMALIZE);
      //  glColor sets ambient and diffuse color materials
      glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE);
      glEnable(GL_COLOR_MATERIAL);
      //  Enable light 0
      glEnable(GL_LIGHT0);
      glLightfv(GL_LIGHT0,GL_AMBIENT ,Ambient);
      glLightfv(GL_LIGHT0,GL_DIFFUSE ,Diffuse);
      glLightfv(GL_LIGHT0,GL_SPECULAR,Specular);
      glLightfv(GL_LIGHT0,GL_POSITION,Position);
      glMaterialf(GL_FRONT_AND_BACK,GL_SHININESS,32);
      glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,white);

   }
   else
      glDisable(GL_LIGHTING);
   if (mode)
   {
      //  Draw opaque cubes
      for (i=-1;i<=1;i+=2)
         for (j=-1;j<=1;j+=2)
            for (k=-1;k<=1;k+=2)
               if (((i+j+k)&2)!=0)
                  cube(i,j,k , 1);
      //  Draw transparent cubes
      for (i=-1;i<=1;i+=2)
         for (j=-1;j<=1;j+=2)
            for (k=-1;k<=1;k+=2)
               if (((i+j+k)&2)==0)
                  cube(i,j,k , 0);
   }
   else
   {
      for (i=-1;i<=1;i+=2)
         for (j=-1;j<=1;j+=2)
            for (k=-1;k<=1;k+=2)
                cube(i,j,k , (i+j+k)&2);
   }
   //  Draw axes
   glDisable(GL_LIGHTING);
   glColor3f(1,1,1);
   if (axes)
   {
      glBegin(GL_LINES);
      glVertex3d(0.0,0.0,0.0);
      glVertex3d(len,0.0,0.0);
      glVertex3d(0.0,0.0,0.0);
      glVertex3d(0.0,len,0.0);
      glVertex3d(0.0,0.0,0.0);
      glVertex3d(0.0,0.0,len);
      glEnd();
      //  Label axes
      glRasterPos3d(len,0.0,0.0);
      Print("X");
      glRasterPos3d(0.0,len,0.0);
      Print("Y");
      glRasterPos3d(0.0,0.0,len);
      Print("Z");
   }
   //  Display parameters
   glWindowPos2i(5,5);
   Print("Angle=%d,%d  Dim=%.1f alpha=%.2f Sequence=%s Blend Dest=%s",
      th,ph,dim,alpha,mode?"Ordered":"Unordered",aone?"GL_ONE":"GL_ONE_MINUS_SRC_ALPHA");
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
   else if (key == GLUT_KEY_UP)
      ph += 5;
   //  Down arrow key - decrease elevation by 5 degrees
   else if (key == GLUT_KEY_DOWN)
      ph -= 5;
   //  PageUp key - increase dim
   else if (key == GLUT_KEY_PAGE_DOWN)
      dim += 0.1;
   //  PageDown key - decrease dim
   else if (key == GLUT_KEY_PAGE_UP && dim>1)
      dim -= 0.1;
   //  Keep angles to +/-360 degrees
   th %= 360;
   ph %= 360;
   //  Update projection
   Project(fov,asp,dim);
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
      th = ph = 0;
   //  Toggle axes
   else if (ch == 'a' || ch == 'A')
      axes = 1-axes;
   //  Toggle alpha mode
   else if (ch == '1')
      aone = 1-aone;
   //  Switch display mode
   else if (ch == 'm' || ch == 'M')
      mode = 1-mode;
   //  Toggle light
   else if (ch == 'l' || ch == 'L')
      light = 1-light;
   //  Change field of view angle
   else if (ch == '-' && alpha>0.01)
      alpha -= 0.05;
   else if (ch == '+' && alpha<1)
      alpha += 0.05;
   //  Reproject
   Project(fov,asp,dim);
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
   Project(fov,asp,dim);
}

/*
 *  GLUT calls this routine when the window is resized
 */
void idle()
{
   //  Elapsed time in seconds
   double t = glutGet(GLUT_ELAPSED_TIME)/1000.0;
   zh = fmod(90*t,360.0);
   //  Tell GLUT it is necessary to redisplay the scene
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
   glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE | GLUT_ALPHA);
   glutInitWindowSize(600,600);
   glutCreateWindow("Mixed Objects");
   //  Set callbacks
   glutDisplayFunc(display);
   glutReshapeFunc(reshape);
   glutSpecialFunc(special);
   glutKeyboardFunc(key);
   glutIdleFunc(idle);
   //  Load textures
   LoadTexBMP("glass.bmp");
   //  Pass control to GLUT so it can interact with the user
   ErrCheck("init");
   glutMainLoop();
   return 0;
}
