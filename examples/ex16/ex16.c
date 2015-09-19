/*
 *  Lego
 *
 *  Draw a  Lego brick to demonstrate lighting.
 *  Lego is a trademark of the Lego Group.
 *
 *  Key bindings:
 *  l          Cycles between wirefame, and lighting modes
 *  c          Cycles through colors
 *  s/S        Toggle light movement
 *  b/B        Toggle blank/studded
 *  m/M        Increase/decrease studs in the X-direction
 *  n/N        Increase/decrease studs in the Y-direction
 *  t/T        Toggle between thin/thick bricks
 *  []         Lower/rise light
 *  arrows     Change view angle
 *  PgDn/PgUp  Zoom in and out
 *  0          Reset view angle
 *  ESC        Exit
 */
#include "CSCIx229.h"

int mode=2;        //  Lighting mode
int move=1;        //  Move light
int blank=0;       //  Blank or studded
int th=0;          //  Azimuth of view angle
int ph=0;          //  Elevation of view angle
int zh=90;         //  Light azimuth
double asp=1;      //  Aspect ratio
double dim=4;      //  Size of world
float ylight=5;    //  Elevation of light
int L=1,M=2,N=4;   //  Brick size
int K=5;           //  Color
typedef struct {float r,g,b;} color_t;
color_t color[7] = {{1.0,1.0,1.0},{0.5,0.5,0.5},{0.5,0.0,0.0},{1.0,0.5,0.0},{1.0,1.0,0.0},{0.0,0.5,0.0},{0.0,0.0,0.5}};
char* text[] = {"Wireframe","Filled no lighting","Filled with Lighting"};

/*
 *  Draw a stud
 */
static void stud(float x,float y, float z)
{
   int th;
   glPushMatrix();
   glTranslatef(x,y,z);
   glScaled(1.5,1.5,1);
   //  Top
   glBegin(GL_TRIANGLE_FAN);
   glNormal3f(0,0,1);
   glVertex3f(0,0,1);
   for (th=0;th<=360;th+=15)
      glVertex3f(Cos(th),Sin(th),1);
   glEnd();
   //  Sides
   glBegin(GL_QUAD_STRIP);
   for (th=0;th<=360;th+=15)
   {
      glNormal3f(Cos(th),Sin(th),0);
      glVertex3f(Cos(th),Sin(th),1);
      glVertex3f(Cos(th),Sin(th),0);
   }
   glEnd();
   glPopMatrix();
}

/*
 *  Draw a peg
 */
static void peg(float x,float y,float Z0,float Z1)
{
   int th;
   float Or = 2.5*sqrt(2.0)-1.5; 
   float Ir = Or-0.5;
   glPushMatrix();
   glTranslatef(x,y,0);
   //  Outside
   glBegin(GL_QUAD_STRIP);
   for (th=0;th<=360;th+=15)
   {
      glNormal3f(Cos(th),Sin(th),0);
      glVertex3f(Or*Cos(th),Or*Sin(th),Z1);
      glVertex3f(Or*Cos(th),Or*Sin(th),Z0);
   }
   glEnd();
   //  Inside
   glBegin(GL_QUAD_STRIP);
   for (th=0;th<=360;th+=15)
   {
      glNormal3f(-Cos(th),-Sin(th),0);
      glVertex3f(Ir*Cos(th),Ir*Sin(th),Z0);
      glVertex3f(Ir*Cos(th),Ir*Sin(th),Z1);
   }
   glEnd();
   //  Edge
   glNormal3f(0,0,-1);
   glBegin(GL_QUAD_STRIP);
   for (th=0;th<=360;th+=15)
   {
      glVertex3f(Or*Cos(th),Or*Sin(th),Z0);
      glVertex3f(Ir*Cos(th),Ir*Sin(th),Z0);
   }
   glEnd();
   glPopMatrix();
}

/*
 *  Draw a MxN brick
 */
static void brick(int m, int n,int thick,float R,float G,float B)
{
   int i,j;
   //  Set all colors
   float black[] = {0,0,0,1};
   float color[] = {R,G,B,1};
   float shinyvec[] = {16};
   glColor3f(R,G,B);
   glMaterialfv(GL_FRONT_AND_BACK,GL_SHININESS,shinyvec);
   glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT,color);
   glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,color);
   glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,color);
   glMaterialfv(GL_FRONT_AND_BACK,GL_EMISSION,black);

   //  Save transformation
   glPushMatrix();
   glScaled(0.2,0.2,0.2);
   //  Outside dimensions
   float Ox = 2.5*m;
   float Oy = 2.5*n;
   float Oz = thick?3:1;
   //  Inside dimensions
   float Ix = Ox-1;
   float Iy = Oy-1;
   float Iz = Oz-1;

   //  Top and sides
   glBegin(GL_QUADS);
   //  Outside front
   glNormal3f( 0, 1, 0);
   glVertex3f(-Ox,+Oy,+Oz);
   glVertex3f(+Ox,+Oy,+Oz);
   glVertex3f(+Ox,+Oy,-Oz);
   glVertex3f(-Ox,+Oy,-Oz);
   //  Outside back
   glNormal3f( 0,-1, 0);
   glVertex3f(+Ox,-Oy,+Oz);
   glVertex3f(-Ox,-Oy,+Oz);
   glVertex3f(-Ox,-Oy,-Oz);
   glVertex3f(+Ox,-Oy,-Oz);
   //  Outside right
   glNormal3f( 1, 0, 0);
   glVertex3f(+Ox,+Oy,+Oz);
   glVertex3f(+Ox,-Oy,+Oz);
   glVertex3f(+Ox,-Oy,-Oz);
   glVertex3f(+Ox,+Oy,-Oz);
   //  Outside left
   glNormal3f(-1, 0, 0);
   glVertex3f(-Ox,+Oy,-Oz);
   glVertex3f(-Ox,-Oy,-Oz);
   glVertex3f(-Ox,-Oy,+Oz);
   glVertex3f(-Ox,+Oy,+Oz);
   //  Outside top
   glNormal3f( 0, 0, 1);
   glVertex3f(-Ox,+Oy,+Oz);
   glVertex3f(-Ox,-Oy,+Oz);
   glVertex3f(+Ox,-Oy,+Oz);
   glVertex3f(+Ox,+Oy,+Oz);
   //  Inside front
   glNormal3f( 0,-1, 0);
   glVertex3f(-Ix,+Iy,-Oz);
   glVertex3f(+Ix,+Iy,-Oz);
   glVertex3f(+Ix,+Iy,+Iz);
   glVertex3f(-Ix,+Iy,+Iz);
   //  Inside back
   glNormal3f( 0,+1, 0);
   glVertex3f(+Ix,-Iy,-Oz);
   glVertex3f(-Ix,-Iy,-Oz);
   glVertex3f(-Ix,-Iy,+Iz);
   glVertex3f(+Ix,-Iy,+Iz);
   //  Inside right
   glNormal3f(-1, 0, 0);
   glVertex3f(+Ix,+Iy,-Oz);
   glVertex3f(+Ix,-Iy,-Oz);
   glVertex3f(+Ix,-Iy,+Iz);
   glVertex3f(+Ix,+Iy,+Iz);
   //  Inside left
   glNormal3f(+1, 0, 0);
   glVertex3f(-Ix,-Iy,-Oz);
   glVertex3f(-Ix,+Iy,-Oz);
   glVertex3f(-Ix,+Iy,+Iz);
   glVertex3f(-Ix,-Iy,+Iz);
   //  Inside top
   glNormal3f( 0, 0,-1);
   glVertex3f(+Ix,+Iy,+Iz);
   glVertex3f(+Ix,-Iy,+Iz);
   glVertex3f(-Ix,-Iy,+Iz);
   glVertex3f(-Ix,+Iy,+Iz);
   glEnd();
   //  Bottom
   glBegin(GL_QUAD_STRIP);
   glNormal3f( 0, 0,-1);
   glVertex3f(+Ix,+Iy,-Oz);
   glVertex3f(+Ox,+Oy,-Oz);
   glVertex3f(+Ix,-Iy,-Oz);
   glVertex3f(+Ox,-Oy,-Oz);
   glVertex3f(-Ix,-Iy,-Oz);
   glVertex3f(-Ox,-Oy,-Oz);
   glVertex3f(-Ix,+Iy,-Oz);
   glVertex3f(-Ox,+Oy,-Oz);
   glVertex3f(+Ix,+Iy,-Oz);
   glVertex3f(+Ox,+Oy,-Oz);
   glEnd();

   if (!blank)
   {
      //  Draw studs
      for (i=0;i<m;i++)
         for (j=0;j<n;j++)
            stud(5*i-Ox+2.5,5*j-Oy+2.5,Oz);
   }

   //  Draw pegs
   for (i=0;i<m-1;i++)
      for (j=0;j<n-1;j++)
         peg(5*i-Ox+5,5*j-Oy+5,-Oz,Iz);

   //  Undo transofrmations
   glPopMatrix();
}

/*
 *  Draw vertex in polar coordinates with normal
 */
static void Vertex(double th,double ph)
{
   double x = Sin(th)*Cos(ph);
   double y = Cos(th)*Cos(ph);
   double z =         Sin(ph);
   //  For a sphere at the origin, the position
   //  and normal vectors are the same
   glNormal3d(x,y,z);
   glVertex3d(x,y,z);
}

/*
 *  Draw a ball
 *     at (x,y,z)
 *     radius (r)
 */
static void ball(double x,double y,double z,double r)
{
   int th,ph;
   const int inc=15;
   //  Save transformation
   glPushMatrix();
   //  Offset, scale and rotate
   glTranslated(x,y,z);
   glScaled(r,r,r);
   //  Bands of latitude
   for (ph=-90;ph<90;ph+=inc)
   {
      glBegin(GL_QUAD_STRIP);
      for (th=0;th<=360;th+=2*inc)
      {
         Vertex(th,ph);
         Vertex(th,ph+inc);
      }
      glEnd();
   }
   //  Undo transofrmations
   glPopMatrix();
}

/*
 *  OpenGL (GLUT) calls this routine to display the scene
 */
void display()
{
   const double len=3.0;  //  Length of axes
   //  Erase the window and the depth buffer
   glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
   //  Enable Z-buffering in OpenGL
   glEnable(GL_DEPTH_TEST);

   //  Undo previous transformations
   glLoadIdentity();
   //  Set view
   glRotatef(ph,1,0,0);
   glRotatef(th,0,1,0);
   glRotatef(-90,1,0,0);

   //  Light switch
   if (mode==2)
   {
        //  Translate intensity to color vectors
        float Ambient[]   = {0.3,0.3,0.3,1.0};
        float Diffuse[]   = {0.8,0.8,0.8,1.0};
        float Specular[]  = {1.0,1.0,1.0,1.0};
        //  Light position
        float Position[]  = {5*Cos(zh),5*Sin(zh),ylight,1.0};
        //  Draw light position as ball (still no lighting here)
        glColor3f(1,1,1);
        ball(Position[0],Position[1],Position[2] , 0.1);
        //  OpenGL should normalize normal vectors
        glEnable(GL_NORMALIZE);
        //  Enable lighting
        glEnable(GL_LIGHTING);
        //  Enable light 0
        glEnable(GL_LIGHT0);
        //  Set ambient, diffuse, specular components and position of light 0
        glLightfv(GL_LIGHT0,GL_AMBIENT ,Ambient);
        glLightfv(GL_LIGHT0,GL_DIFFUSE ,Diffuse);
        glLightfv(GL_LIGHT0,GL_SPECULAR,Specular);
        glLightfv(GL_LIGHT0,GL_POSITION,Position);
   }
   else
     glDisable(GL_LIGHTING);

   //  Wireframe
   if (mode==0)
   {
      //  Draw polygon outlines
      glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
      brick(M,N,L,color[K].r,color[K].g,color[K].b);
      //  Draw polygon interior
      glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
      glEnable(GL_POLYGON_OFFSET_FILL);
      glPolygonOffset(1,1);
      brick(M,N,L,0,0,0);
      glDisable(GL_POLYGON_OFFSET_FILL);
   }
   //  Regular fill
   else
      brick(M,N,L,color[K].r,color[K].g,color[K].b);

   //  Draw axes - no lighting from here on
   glDisable(GL_LIGHTING);
   glColor3f(1,1,1);
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

   //  Display parameters
   glWindowPos2i(5,5);
   Print("Angle=%d,%d  Dim=%.1f M=%d N=%d T=%d Mode=%s",
      th,ph,dim,M,N,L,text[mode]);

   //  Render the scene and make it visible
   ErrCheck("display");
   glFlush();
   glutSwapBuffers();
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
   Project(0,asp,dim);
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
   //  Lighting mode
   else if (ch == 'l')
      mode = (mode+1)%3;
   else if (ch == 'L')
      mode = (mode+2)%3;
   //  Toggle light movement
   else if (ch == 's' || ch == 'S')
      move = 1-move;
   //  Toggle blank/studded
   else if (ch == 'b' || ch == 'B')
      blank = 1-blank;
   //  Move light
   else if (ch == '<')
      zh += 1;
   else if (ch == '>')
      zh -= 1;
   //  Light elevation
   else if (ch=='[')
      ylight -= 0.1;
   else if (ch==']')
      ylight += 0.1;
   //  Change width
   else if (ch=='m')
      M++;
   else if (ch=='M' && M>1)
      M--;
   //  Change length
   else if (ch=='n')
      N++;
   else if (ch=='N' && N>1)
      N--;
   //  Change thickness
   else if (ch=='t' || ch=='T')
      L = !L;
   else if (ch=='c' || ch=='C')
      K = (K+1)%7;
   //  Animate if requested
   glutIdleFunc(move?idle:NULL);
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
   Project(0,asp,dim);
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
   glutInitWindowSize(600,400);
   glutCreateWindow("Lego");
   //  Set callbacks
   glutDisplayFunc(display);
   glutReshapeFunc(reshape);
   glutSpecialFunc(special);
   glutKeyboardFunc(key);
   glutIdleFunc(idle);
   //  Pass control to GLUT so it can interact with the user
   ErrCheck("init");
   glutMainLoop();
   return 0;
}
