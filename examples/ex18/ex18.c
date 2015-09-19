/*
 *  Solar System
 *
 *  Shows celestial bodies in from the local solar system.
 *  Also shows the solar system to scale.
 *
 *  Key bindings:
 *  m          Cycle through celectial bodies
 *  t          Toggle ring transparency
 *  v          View solar system
 *  o          Toggle orbits in solar system mode
 *  l          Toggle lighting in solar system mode
 *  +/-        Change magnification of planets in solar system mode
 *  []         Adjust time scale in solar system mode
 *  s          Start/stop motion in solar system mode
 *  a          Toggle axes
 *  arrows     Change view angle
 *  PgDn/PgUp  Zoom in and out
 *  0          Reset view angle
 *  ESC        Exit
 */
#include "CSCIx229.h"
int mode;      //  Display mode
int axes=0;    //  Display axes
int th=0;      //  Azimuth of view angle
int ph=0;      //  Elevation of view angle
int zh=0;      //  Rotation
int mag=1;     //  Magnification for solar system
int fov=30;    //  Field of view
int move=1;    //  Movement
int orbit=0;   //  Show orbits
int light=0;   //  Lighting
int trans=1;   //  Transparency
double asp=1;  //  Aspect ratio
double day=0;  //  Day
double dim=5;  //  Size of world

#define TWOPI (2*M_PI)

typedef struct
{
   char* name;
   char* ball;
   char* ring;
   unsigned int balltex;
   unsigned int ringtex;
   double  R;
   double rot;
   double th;
   double ph;
   double i,o,p,a,n,e,l;
}  Planet;
const int N=11;
Planet planet[] = {
//                                                                                     ---------------   Keplearian elements in radians  ----------------
//   Name      Texture      RingTexture            Radius   Rotation      th     ph     i        o        p        a       n            e         l
   {"Sun"    ,"sun.bmp"    ,NULL             ,0,0, 695000 ,    24.60 ,    0.0,   0.0,  0.0     ,0.0     ,0.0     , 0.0   ,0.0         ,0.0      ,0.0     },
   {"Mercury","mercury.bmp",NULL             ,0,0,   2440 ,    58.60 ,    0.0,   0.0,  0.122262,0.843586,1.351827, 0.3871,0.0714250340,0.2056324,5.487729},
   {"Venus"  ,"venus.bmp"  ,NULL             ,0,0,   6052 ,  -243.00 , -180.0,   0.0,  0.059249,1.338474,2.299663, 0.7233,0.0279629322,0.0067933,4.135391},
   {"Earth"  ,"earth.bmp"  ,NULL             ,0,0,   6378 ,     0.99 ,  -23.5,   0.0,  0.000007,6.094690,1.795101, 1.0000,0.0172016091,0.0166967,5.731723},
   {"Mars"   ,"mars.bmp"   ,NULL             ,0,0,   3397 ,     1.03 ,  -25.2,   0.0,  0.032287,0.865097,5.865846, 1.5236,0.0091465952,0.0934231,4.580230},
   {"Jupiter","jupiter.bmp",NULL             ,0,0,  71492 ,     0.41 ,   -3.1,   0.0,  0.022770,1.753555,0.273978, 5.2026,0.0014503019,0.0484646,5.629731},
   {"Saturn" ,"saturn.bmp" ,"saturnrings.bmp",0,0,  60268 ,     0.45 ,  -26.7,  10.0,  0.043376,1.983319,1.550952, 9.5719,0.0005809601,0.0531651,0.365779},
   {"Uranus" ,"uranus.bmp" ,"uranusrings.bmp",0,0,  25559 ,    -0.72 ,    0.0,  98.0,  0.013499,1.293209,3.066207,19.3018,0.0002028587,0.0428959,5.291658},
   {"Neptune","neptune.bmp",NULL             ,0,0,  24766 ,     0.67 ,  -29.0,   0.0,  0.030859,2.300213,0.125768,30.2666,0.0001033110,0.0102981,5.233616},
   {"Pluto"  ,"pluto.bmp"  ,NULL             ,0,0,   1150 ,    -6.39 ,    0.0,  90.0,  0.298824,1.926552,3.923544,39.5804,0.0000690814,0.2501272,4.114886},
   {"Moon"   ,"moon.bmp"   ,NULL             ,0,0,   1738 ,27.321582 ,   -6.6,   0.0,  0.0     ,0.0     ,0.0     ,0.00257,0.0         ,0.0      ,0.0     },
   };

/*
 *  Draw vertex in polar coordinates
 */
static void Vertex(int th,int ph)
{
   double x = -Sin(th)*Cos(ph);
   double y =  Cos(th)*Cos(ph);
   double z =          Sin(ph);
   glNormal3d(x,y,z);
   glTexCoord2d(th/360.0,ph/180.0+0.5);
   glVertex3d(x,y,z);
}

/*
 *  Draw planet
 */
void DrawPlanet(int n)
{
   int th,ph;

   /*
    *  Draw surface of the planet
    */
   //  Set texture
   glEnable(GL_TEXTURE_2D);
   glBindTexture(GL_TEXTURE_2D,planet[n].balltex);
   //  Latitude bands
   glColor3f(1,1,1);
   for (ph=-90;ph<90;ph+=5)
   {
      glBegin(GL_QUAD_STRIP);
      for (th=0;th<=360;th+=5)
      {
         Vertex(th,ph);
         Vertex(th,ph+5);
      }
      glEnd();
   }

   /*
    *  Draw rings for planets that have them
    *  We naively assume the ring width equals the radius
    */
   if (planet[n].ring)
   {
      int k;
      //  Make rings transparent grey (if enabled)
      if (trans) glEnable(GL_BLEND);
      glBlendFunc(GL_ONE,GL_ONE_MINUS_SRC_COLOR);
      glColor3f(0.5,0.5,0.5);
      //  Set ring texture
      glBindTexture(GL_TEXTURE_2D,planet[n].ringtex);
      //  Draw ring plane
      glBegin(GL_QUAD_STRIP);
      for (k=0;k<=360;k+=2)
      {
         glTexCoord2f(1,0);glVertex2d(1.0*Cos(k),1.0*Sin(k));
         glTexCoord2f(0,0);glVertex2d(2.2*Cos(k),2.2*Sin(k));
      }
      glEnd();
      glDisable(GL_BLEND);
   }
   glDisable(GL_TEXTURE_2D);
}

/*
 *  Draw the solar system
 */
static void SolarSystem()
{
   int k;

   //  Set up lighting
   float black[]   = {0.0 , 0.0 , 0.0 , 1.0};
   float white[]   = {1.0 , 1.0 , 1.0 , 1.0};
   float pos[]     = {0.0 , 0.0 , 0.0 , 1.0};
   glEnable(GL_NORMALIZE);
   glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE);
   glEnable(GL_COLOR_MATERIAL);
   glEnable(GL_LIGHT0);
   glLightModelfv(GL_LIGHT_MODEL_AMBIENT,black);
   glLightfv(GL_LIGHT0,GL_AMBIENT ,black);
   glLightfv(GL_LIGHT0,GL_DIFFUSE ,white);
   glLightfv(GL_LIGHT0,GL_SPECULAR,black);
   glLightfv(GL_LIGHT0,GL_POSITION,pos);

   //  Draw the sun and planets
   for (k=0;k<=9;k++)
   {
      //  Copy variables to simplify expresions
      double i = planet[k].i;  //  Inclination
      double o = planet[k].o;  //  Longitude of Ascending Node
      double p = planet[k].p;  //  Longitude of Perihelion
      double a = planet[k].a;  //  Mean distance (AU)
      double n = planet[k].n;  //  Daily modtion
      double e = planet[k].e;  //  Eccentricity
      double l = planet[k].l;  //  Mean longitude

      //  Radius of planet (AU)
      double r = 6.7e-9*planet[k].R;  //  Radius (km->AU)
      if (mode<0 && mag>0) r *= mag;  //  Magnification
      if (r>0.3) r = 0.3;             //  Limit maximum size

      //  Apply keplerian elements to determine orbit
      glPushMatrix();
      if (k>0)
      {
         //  Calculate true anomaly
         double M = fmod(day*n+l-p,TWOPI); //  Mean anomaly in radians
         double delta;
         double v = M;
         do
         {
            delta = v - e*sin(v) - M;
            v -= delta/(1-e*cos(v));
         } while (fabs(delta)>1e-12);
         //  True anomaly
         v = 2*atan(sqrt((1+e)/(1-e))*tan(0.5*v));
         double R = a*(1-e*e)/(1+e*cos(v)); //  Orbit radius (AU)
         double h = v + p - o;              //  Elliptical angle
         //  Compute and apply location
         double x = R * (cos(o)*cos(h) - sin(o)*sin(h)*cos(i));
         double y = R * (sin(o)*cos(h) + cos(o)*sin(h)*cos(i));
         double z = R *                        (sin(h)*sin(i));
         glTranslated(x,y,z);
      }

      //  Transform and draw planet
      glPushMatrix();
      glRotated(planet[k].th,1,0,0);           //  Declination
      glRotated(360*day/planet[k].rot,0,0,1);  //  Siderial rotation
      glScaled(r,r,r);                         //  Radius of planet
      if (k>0 && light) glEnable(GL_LIGHTING); //  Lighting for planets
      DrawPlanet(k);                           //  Draw planet
      glPopMatrix();

      //  Draw the moon relative to earth
      if (k==3)
      {
         double R = 1.5*r>planet[10].a ? 1.5*r : planet[10].a;
         glPushMatrix();
         glRotated(360*day/planet[10].rot,0,0,1); //  Siderial rotation and orbit is the same
         glTranslated(R,0,0);                     //  Orbital radius
         r *= planet[10].R/planet[k].R;           //  Adjust radius
         glScaled(r,r,r);
         DrawPlanet(10);                          //  Draw planet
         glPopMatrix();
      }
      glPopMatrix();
      glDisable(GL_LIGHTING);

      //  Draw orbit
      if (orbit)
      {
         double v;
         glColor3f(0.3,0.3,0.3);
         glBegin(GL_LINE_LOOP);
         for (v=0;v<TWOPI;v+=0.01)
         {
            double R = a*(1-e*e)/(1+e*cos(v));
            double h = v + p - o;
            double x = R * (cos(o)*cos(h) - sin(o)*sin(h)*cos(i));
            double y = R * (sin(o)*cos(h) + cos(o)*sin(h)*cos(i));
            double z = R *                        (sin(h)*sin(i));
            glVertex3d(x,y,z);
         }
         glEnd();
      }
   }
}

/*
 *  OpenGL (GLUT) calls this routine to display the scene
 */
void display()
{
   //  Length of axes
   const double len=1.2;
   //  Eye position
   double Ex = -2*dim*Cos(ph);
   double Ey = +2*dim*Sin(ph);
   double Ez = 0;
   //  Erase the window and the depth buffer
   glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
   //  Set perspective
   glLoadIdentity();
   gluLookAt(Ex,Ey,Ez , 0,0,0 , 0,Cos(ph),0);
   //  Draw scene
   glEnable(GL_DEPTH_TEST);
   //  Rotate Z up
   glRotated(-90,1,0,0);

   /*
    * Draw solar system
    */
   if (mode<0)
   {
      glRotated(th,0,0,1);  //  View angle
      SolarSystem();
   }
   /*
    *  Draw planet
    */
   else
   {
      glRotated(th,1,0,0);  // Declination
      glRotated(zh,0,0,1);  // Spin around axes
      DrawPlanet(mode);
   }

   /*
    *  Draw axes - no textures from here
    */
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
   Print("Angle=%d,%d  Dim=%.1f Object=%s",th,ph,2*dim,mode<0?"Solar System":planet[mode].name);
   if (mode<0) Print(" Magnification %d Year %.1f",mag,2000+day/365.25);
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
   else if (key == GLUT_KEY_PAGE_UP && dim>0.5)
      dim -= 0.1;
   //  Keep angles to +/-360 degrees
   th %= 360;
   ph %= 360;
   //  Update projection
   Project(fov,asp,dim);
   //  Tell GLUT it is necessary to redisplay the scene
   glutPostRedisplay();
}

void SetMode(int k)
{
   mode = k;
   th = planet[k].th;
   ph = planet[k].ph;
}

/*
 *  GLUT calls this routine when the window is resized
 */
void idle()
{
   zh = 15*glutGet(GLUT_ELAPSED_TIME)/1000.0;
   day  = 0.1*glutGet(GLUT_ELAPSED_TIME);
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
      mag = 1;
      dim = 5;
      th = planet[mode].th;
      ph = planet[mode].ph;
   }
   //  Select display mode
   else if (ch == 'm')
      SetMode((mode+1)%N);
   else if (ch == 'M')
      SetMode((mode+N-1)%N);
   else if (ch == 'v')
      mode = -1;
   //  Magnification
   else if (ch == '-' && mag>1)
      mag = mag/2;
   else if (ch == '+' && mag<65536)
      mag = 2*mag;
   //  Toggle axes
   else if (ch == 'a' || ch == 'A')
      axes = 1-axes;
   //  Toggle movement
   else if (ch == 's' || ch == 'S')
      move = 1-move;
   //  Toggle orbits
   else if (ch == 'o' || ch == 'O')
      orbit = 1-orbit;
   //  Toggle lighting
   else if (ch == 'l' || ch == 'L')
      light = 1-light;
   //  Toggle transparency
   else if (ch == 't' || ch == 'T')
      trans = 1-trans;
   //  Move
   else if (ch == '[')
      day -= 0.1;
   else if (ch == ']')
      day += 0.1;
   //  Reproject
   Project(fov,asp,dim);
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
   Project(fov,asp,dim);
}

/*
 *  Start up GLUT and tell it what to do
 */
int main(int argc,char* argv[])
{
   int k;

   //  Initialize GLUT
   glutInit(&argc,argv);
   //  Request double buffered, true color window with Z buffering at 600x600
   glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE | GLUT_ALPHA);
   glutInitWindowSize(1000,700);
   glutCreateWindow("Solar System");
   //  Set callbacks
   glutDisplayFunc(display);
   glutReshapeFunc(reshape);
   glutSpecialFunc(special);
   glutKeyboardFunc(key);
   glutIdleFunc(idle);
   //  Load textures
   for (k=0;k<N;k++)
   {
      planet[k].balltex = LoadTexBMP(planet[k].ball);
      planet[k].ringtex = planet[k].ring ? LoadTexBMP(planet[k].ring) : 0;
   }
   SetMode(3);

   //  Pass control to GLUT so it can interact with the user
   ErrCheck("init");
   glutMainLoop();
   return 0;
}
