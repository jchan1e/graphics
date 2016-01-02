/*
 *  Cockpit
 *
 *  Demonstrates how to draw a cockpit view for a 3D scene.
 *
 *  Key bindings:
 *  m          Toggle cockpit mode
 *  a          Toggle axes
 *  arrows     Change view angle
 *  PgDn/PgUp  Zoom in and out
 *  0          Reset view angle
 *  ESC        Exit
 */
#include "CSCIx229.h"
int axes=0;       //  Display axes
int mode=0;       //  Display cockpit
int th=0;         //  Azimuth of view angle
int ph=0;         //  Elevation of view angle
int fov=25;       //  Field of view
double asp=1;     //  Aspect ratio
double dim=5.0;   //  Size of world

/*
 *  Draw a cube
 *     at (x,y,z)
 *     dimentions (dx,dy,dz)
 *     rotated th about the y axis
 */
static void cube(double x,double y,double z,
                 double dx,double dy,double dz,
                 double th)
{
   //  Save transformation
   glPushMatrix();
   //  Offset
   glTranslated(x,y,z);
   glRotated(th,0,1,0);
   glScaled(dx,dy,dz);
   //  Cube
   glBegin(GL_QUADS);
   //  Front
   glColor3f(1,0,0);
   glVertex3f(-1.0,-1.0, 1.0);
   glVertex3f(+1.0,-1.0, 1.0);
   glVertex3f(+1.0,+1.0, 1.0);
   glVertex3f(-1.0,+1.0, 1.0);
   //  Back
   glColor3f(0,0,1);
   glVertex3f(+1.0,-1.0,-1.0);
   glVertex3f(-1.0,-1.0,-1.0);
   glVertex3f(-1.0,+1.0,-1.0);
   glVertex3f(+1.0,+1.0,-1.0);
   //  Right
   glColor3f(1,1,0);
   glVertex3f(+1.0,-1.0,+1.0);
   glVertex3f(+1.0,-1.0,-1.0);
   glVertex3f(+1.0,+1.0,-1.0);
   glVertex3f(+1.0,+1.0,+1.0);
   //  Left
   glColor3f(0,1,0);
   glVertex3f(-1.0,-1.0,-1.0);
   glVertex3f(-1.0,-1.0,+1.0);
   glVertex3f(-1.0,+1.0,+1.0);
   glVertex3f(-1.0,+1.0,-1.0);
   //  Top
   glColor3f(0,1,1);
   glVertex3f(-1.0,+1.0,+1.0);
   glVertex3f(+1.0,+1.0,+1.0);
   glVertex3f(+1.0,+1.0,-1.0);
   glVertex3f(-1.0,+1.0,-1.0);
   //  Bottom
   glColor3f(1,0,1);
   glVertex3f(-1.0,-1.0,-1.0);
   glVertex3f(+1.0,-1.0,-1.0);
   glVertex3f(+1.0,-1.0,+1.0);
   glVertex3f(-1.0,-1.0,+1.0);
   //  End
   glEnd();
   //  Undo transofrmations
   glPopMatrix();
}

/*
 *  Draw the cockpit as an overlay
 *  Must be called last
 */
void Cockpit()
{
   //  Screen edge
   float w = asp>2 ? asp : 2;
   //  Save transform attributes (Matrix Mode and Enabled Modes)
   glPushAttrib(GL_TRANSFORM_BIT|GL_ENABLE_BIT);
   //  Save projection matrix and set unit transform
   glMatrixMode(GL_PROJECTION);
   glPushMatrix();
   glLoadIdentity();
   glOrtho(-asp,+asp,-1,1,-1,1);
   //  Save model view matrix and set to indentity
   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();
   glLoadIdentity();
   //  Draw instrument panel with texture
   glColor3f(1,1,1);
   glEnable(GL_TEXTURE_2D);
   glBegin(GL_QUADS);
   glTexCoord2d(0,0);glVertex2f(-2,-1);
   glTexCoord2d(1,0);glVertex2f(+2,-1);
   glTexCoord2d(1,1);glVertex2f(+2, 0);
   glTexCoord2d(0,1);glVertex2f(-2, 0);
   glEnd();
   glDisable(GL_TEXTURE_2D);
   //  Draw the inside of the cockpit in grey
   glColor3f(0.6,0.6,0.6);
   glBegin(GL_QUADS);
   //  Port
   glVertex2f(-2,-1);
   glVertex2f(-2,+1);
   glVertex2f(-w,+1);
   glVertex2f(-w,-1);
   //  Starboard
   glVertex2f(+2,-1);
   glVertex2f(+2,+1);
   glVertex2f(+w,+1);
   glVertex2f(+w,-1);
   //  Port overhead
   glVertex2f(-2.00,+0.8);
   glVertex2f(-2.00,+1);
   glVertex2f(-0.03,+1);
   glVertex2f(-0.03,+0.9);
   //  Starboard overhead
   glVertex2f(+2.00,+0.8);
   glVertex2f(+2.00,+1);
   glVertex2f(+0.03,+1);
   glVertex2f(+0.03,+0.9);
   //  Windshield divide
   glVertex2f(-0.03,+1);
   glVertex2f(+0.03,+1);
   glVertex2f(+0.03,+0);
   glVertex2f(-0.03,+0);
   glEnd();
   //  Reset model view matrix
   glPopMatrix();
   //  Reset projection matrix
   glMatrixMode(GL_PROJECTION);
   glPopMatrix();
   //  Pop transform attributes (Matrix Mode and Enabled Modes)
   glPopAttrib();
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
   //  Undo previous transformations
   glLoadIdentity();
   //  Perspective - set eye position
   gluLookAt(Ex,Ey,Ez , 0,0,0 , 0,Cos(ph),0);
   //  Enable Z-buffering in OpenGL
   glEnable(GL_DEPTH_TEST);
   //  Draw cubes
   for (i=-1;i<=1;i++)
      for (j=-1;j<=1;j++)
         for (k=-1;k<=1;k++)
            cube(i,j,k , 0.3,0.3,0.3 , 0);
   //  Draw axes
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
   //  Disable Z-buffering in OpenGL
   glDisable(GL_DEPTH_TEST);
   //  Draw cockpit
   if (mode)
      Cockpit();
   //  Display parameters
   else
   {
      glWindowPos2i(5,5);
      Print("Angle=%d,%d  Dim=%.1f FOV=%d",th,ph,dim,fov);
   }
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
   else if (key == GLUT_KEY_PAGE_UP)
      dim += 0.1;
   //  PageDown key - decrease dim
   else if (key == GLUT_KEY_PAGE_DOWN && dim>1)
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
   //  Switch display mode
   else if (ch == 'm' || ch == 'M')
      mode = 1-mode;
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
 *  Start up GLUT and tell it what to do
 */
int main(int argc,char* argv[])
{
   //  Initialize GLUT
   glutInit(&argc,argv);
   //  Request double buffered, true color window with Z buffering at 600x600
   glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
   glutInitWindowSize(600,600);
   glutCreateWindow("Cockpit");
   //  Set callbacks
   glutDisplayFunc(display);
   glutReshapeFunc(reshape);
   glutSpecialFunc(special);
   glutKeyboardFunc(key);
   //  Load cockpit
   LoadTexBMP("737.bmp");
   //  Pass control to GLUT so it can interact with the user
   ErrCheck("init");
   glutMainLoop();
   return 0;
}
