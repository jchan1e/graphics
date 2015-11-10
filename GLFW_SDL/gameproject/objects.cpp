// Jordan Dick 2015
// functions for drawing various objects
//

//#ifndef STDIncludes
//#define STDIncludes
//#include <stdlib.h>
//#include <stdio.h>
//#include <math.h>
//#ifdef __APPLE__
//#include <OpenGL/glu.h>
//#else
//#include <GL/glu.h>
//#endif
//#endif
//
//#ifndef Jobjects
//#define Jobjects
//
//
////Cos and Sin in degrees - stolen from ex8
//#define Sin(x) (sin((x) * 3.1415927/180))
//#define Cos(x) (cos((x) * 3.1415927/180))

#include "objects.h"

//Vertex Polar-Cartesian tranformation function
void VertexC(double th, double ph)
{
   glTexCoord2d(th/360, (ph-90)/180);
   glNormal3d(Sin(th)*Cos(ph), Sin(ph), Cos(th)*Cos(ph));
   glVertex3d(Sin(th)*Cos(ph), Sin(ph), Cos(th)*Cos(ph));
}

//without the coloration
void VertexS(double th, double ph)
{
   glNormal3d(Sin(th)*Cos(ph), Sin(ph), Cos(th)*Cos(ph));
   glVertex3d(Sin(th)*Cos(ph), Sin(ph), Cos(th)*Cos(ph));
}

//without the spherical normal
void Vertex(double th, double ph)
{
   glVertex3d(Sin(th)*Cos(ph), Sin(ph), Cos(th)*Cos(ph));
}

//Normal Polar-Cartesian transformation and summation function
void pNormal(double th1, double ph1, double th2, double ph2, double th3, double ph3)
{
   glNormal3d(Sin(th1)*Cos(ph1) + Sin(th2)*Cos(th2) + Sin(th3)*Cos(ph3),
              Sin(ph1) + Sin(ph2) + Sin(ph3),
              Cos(th1)*Cos(ph1) + Cos(th2)*Cos(ph2) + Cos(th3)*Cos(ph3));
}
//Sphere function nabbed from ex8 and slightly modified
void ball(double x, double y, double z,
          double s)
{
   glPushMatrix();

   glTranslated(x, y, z);
   glScaled(s, s, s);

   //top fan
   glBegin(GL_TRIANGLE_FAN);
   VertexS(0,90);
   for(int th = 0; th <= 360; th += 5)
   {
      VertexS(th, 85);
   }
   glEnd();

   //latitude rings
   for (int ph = 85; ph >= -80; ph -= 5)
   {
      glBegin(GL_QUAD_STRIP);
      for (int th = 0; th <= 360; th += 5)
      {
         VertexS(th, ph);
         VertexS(th, ph-5);
      }
      glEnd();
   }
   
   //bottom fan
   glBegin(GL_TRIANGLE_FAN);
   VertexS(0,-90);
   for(int th = 360; th >= 0; th -= 5)
   {
      VertexS(th, -85);
   }
   glEnd();

   glPopMatrix();
}

//Sphere function nabbed from ex8 and slightly modified
void sphere(double x, double y, double z,
            double r,
            double s)
{
   glPushMatrix();

   glTranslated(x, y, z);
   glRotated(r, 0,1,0);
   glScaled(s, s, s);

   //top fan
   glBegin(GL_TRIANGLE_FAN);
   VertexC(0,90);
   for(int th = 0; th <= 360; th += 5)
   {
      VertexC(th, 85);
   }
   glEnd();

   //latitude rings
   for (int ph = 85; ph >= -80; ph -= 5)
   {
      glBegin(GL_QUAD_STRIP);
      for (int th = 0; th <= 360; th += 5)
      {
         VertexC(th, ph);
         VertexC(th, ph-5);
      }
      glEnd();
   }
   
   //bottom fan
   glBegin(GL_TRIANGLE_FAN);
   VertexC(0,-90);
   for(int th = 360; th >= 0; th -= 5)
   {
      VertexC(th, -85);
   }
   glEnd();

   glPopMatrix();
}

void cube(double x, double y, double z,
          double r,
          double s)
{
   glPushMatrix();
   glTranslated(x,y,z);
   glRotated(r, 0,1,0);
   glScaled(s/sqrt(2), s/sqrt(2), s/sqrt(2));

   //glColor3f(0.8,0.2,0.2);

   glBegin(GL_QUADS);
   glNormal3d(0,1,0);
   glTexCoord2d(0.0, 0.2);
   glVertex3d(1,1,1);
   glTexCoord2d(0.25, 0.2);
   glVertex3d(1,1,-1);
   glTexCoord2d(0.5, 0.2);
   glVertex3d(-1,1,-1);
   glTexCoord2d(0.75, 0.2);
   glVertex3d(-1,1,1);

   glNormal3d(0,0,1);
   glTexCoord2d(0.0, 0.2);
   glVertex3d(1,1,1);
   glTexCoord2d(0.75, 0.2);
   glVertex3d(-1,1,1);
   glTexCoord2d(0.75, 0.8);
   glVertex3d(-1,-1,1);
   glTexCoord2d(0.0, 0.8);
   glVertex3d(1,-1,1);

   glNormal3d(1,0,0);
   glTexCoord2d(0.0, 0.2);
   glVertex3d(1,1,1);
   glTexCoord2d(0.0, 0.8);
   glVertex3d(1,-1,1);
   glTexCoord2d(0.25, 0.8);
   glVertex3d(1,-1,-1);
   glTexCoord2d(0.25, 0.2);
   glVertex3d(1,1,-1);

   glNormal3d(0,0,-1);
   glTexCoord2d(0.5, 0.8);
   glVertex3d(-1,-1,-1);
   glTexCoord2d(0.5, 0.2);
   glVertex3d(-1,1,-1);
   glTexCoord2d(0.25, 0.2);
   glVertex3d(1,1,-1);
   glTexCoord2d(0.25, 0.8);
   glVertex3d(1,-1,-1);

   glNormal3d(-1,0,0);
   glTexCoord2d(0.5, 0.8);
   glVertex3d(-1,-1,-1);
   glTexCoord2d(0.75, 0.8);
   glVertex3d(-1,-1,1);
   glTexCoord2d(0.75, 0.2);
   glVertex3d(-1,1,1);
   glTexCoord2d(0.5, 0.2);
   glVertex3d(-1,1,-1);

   glNormal3d(0,-1,0);
   glTexCoord2d(0.5, 0.8);
   glVertex3d(-1,-1,-1);
   glTexCoord2d(0.25, 0.8);
   glVertex3d(1,-1,-1);
   glTexCoord2d(0.0, 0.8);
   glVertex3d(1,-1,1);
   glTexCoord2d(0.75, 0.8);
   glVertex3d(-1,-1,1);
   glEnd();


   // WIREFRAME
   //glColor3f(1,1,1);

   //glBegin(GL_LINES);
   //glNormal3d(1,1,0);
   //glVertex3d(1,1,1);
   //glVertex3d(1,1,-1);

   //glNormal3d(0,1,-1);
   //glVertex3d(1,1,-1);
   //glVertex3d(-1,1,-1);

   //glNormal3d(-1,1,0);
   //glVertex3d(-1,1,-1);
   //glVertex3d(-1,1,1);

   //glNormal3d(0,1,1);
   //glVertex3d(-1,1,1);
   //glVertex3d(1,1,1);

   //glNormal3d(1,0,1);
   //glVertex3d(1,1,1);
   //glVertex3d(1,-1,1);

   //glNormal3d(1,-1,0);
   //glVertex3d(1,-1,1);
   //glVertex3d(1,-1,-1);

   //glNormal3d(0,-1,-1);
   //glVertex3d(1,-1,-1);
   //glVertex3d(-1,-1,-1);

   //glNormal3d(-1,-1,0);
   //glVertex3d(-1,-1,-1);
   //glVertex3d(-1,-1,1);

   //glNormal3d(0,-1,1);
   //glVertex3d(-1,-1,1);
   //glVertex3d(1,-1,1);

   //glNormal3d(-1,0,1);
   //glVertex3d(-1,1,1);
   //glVertex3d(-1,-1,1);

   //glNormal3d(-1,0,-1);
   //glVertex3d(-1,1,-1);
   //glVertex3d(-1,-1,-1);

   //glNormal3d(1,0,-1);
   //glVertex3d(1,1,-1);
   //glVertex3d(1,-1,-1);
   //glEnd();

   glPopMatrix();
}

void octahedron(double x, double y, double z,
                 double r,
                 double s)
{
   glPushMatrix();
   glTranslated(x,y,z);
   glRotated(r, 0,1,0);
   glRotated(54.7356, 1,0,1);
   glScaled(s, s, s);

   //glColor3f(0.6,0.4,0.2);

   glBegin(GL_TRIANGLES);
   glNormal3d(1,1,1);
   glVertex3d(1,0,0);
   glVertex3d(0,1,0);
   glVertex3d(0,0,1);

   glNormal3d(-1,1,1);
   glVertex3d(0,0,1);
   glVertex3d(0,1,0);
   glVertex3d(-1,0,0);

   glNormal3d(-1,-1,1);
   glVertex3d(0,0,1);
   glVertex3d(-1,0,0);
   glVertex3d(0,-1,0);

   glNormal3d(-1,-1,-1);
   glVertex3d(0,-1,0);
   glVertex3d(-1,0,0);
   glVertex3d(0,0,-1);

   glNormal3d(1,-1,-1);
   glVertex3d(0,-1,0);
   glVertex3d(0,0,-1);
   glVertex3d(1,0,0);

   glNormal3d(1,1,-1);
   glVertex3d(1,0,0);
   glVertex3d(0,0,-1);
   glVertex3d(0,1,0);

   glNormal3d(1,-1,1);
   glVertex3d(1,0,0);
   glVertex3d(0,0,1);
   glVertex3d(0,-1,0);

   glNormal3d(-1,1,-1);
   glVertex3d(-1,0,0);
   glVertex3d(0,1,0);
   glVertex3d(0,0,-1);
   glEnd();


   //WIREFRAME
   //glColor3f(1,1,1);

   //glBegin(GL_LINE_STRIP);
   //glVertex3d(1,0,0);
   //glVertex3d(0,0,1);
   //glVertex3d(0,-1,0);
   //glVertex3d(1,0,0);
   //glVertex3d(0,1,0);
   //glVertex3d(-1,0,0);
   //glVertex3d(0,0,-1);
   //glVertex3d(0,1,0);
   //glVertex3d(0,0,1);
   //glVertex3d(-1,0,0);
   //glVertex3d(0,-1,0);
   //glVertex3d(0,0,-1);
   //glVertex3d(1,0,0);
   //glEnd();

   glPopMatrix();
}

void dodecahedron(double x, double y, double z,
                 double r,
                 double s)
{
   //points plotted using cartesian coordinates
   double phi = 0.5 + sqrt(5.0)/2; //the golden ratio, not the latitude angle
   double points[20][3];
   //cube corners
   points[0][0] = 1;       points[0][1] = 1;       points[0][2] = 1;       
   points[1][0] = -1;      points[1][1] = 1;       points[1][2] = 1;       
   points[2][0] = -1;      points[2][1] = 1;       points[2][2] = -1;       
   points[3][0] = 1;       points[3][1] = 1;       points[3][2] = -1;       
   points[4][0] = 1;       points[4][1] = -1;      points[4][2] = 1;       
   points[5][0] = -1;      points[5][1] = -1;      points[5][2] = 1;       
   points[6][0] = -1;      points[6][1] = -1;      points[6][2] = -1;       
   points[7][0] = 1;       points[7][1] = -1;      points[7][2] = -1;       
   //top and bottom
   points[8][0] = 1/phi;   points[8][1] = phi;     points[8][2] = 0;       
   points[9][0] = -1/phi;  points[9][1] = phi;     points[9][2] = 0;       
   points[10][0] = 1/phi;  points[10][1] = -phi;   points[10][2] = 0;      
   points[11][0] = -1/phi; points[11][1] = -phi;   points[11][2] = 0;      
   //front and back
   points[12][0] = 0;      points[12][1] = 1/phi;  points[12][2] = phi;      
   points[13][0] = 0;      points[13][1] = -1/phi; points[13][2] = phi;      
   points[14][0] = 0;      points[14][1] = 1/phi;  points[14][2] = -phi;      
   points[15][0] = 0;      points[15][1] = -1/phi; points[15][2] = -phi;      
   //left and right
   points[16][0] = phi;    points[16][1] = 0;      points[16][2] = 1/phi;      
   points[17][0] = phi;    points[17][1] = 0;      points[17][2] = -1/phi;      
   points[18][0] = -phi;   points[18][1] = 0;      points[18][2] = 1/phi;      
   points[19][0] = -phi;   points[19][1] = 0;      points[19][2] = -1/phi;      

   glPushMatrix();
   glTranslated(x, y, z);
   glRotated(r, 0,1,0);
   glRotated(-58.2825, 0,0,1);
   glScaled(s/sqrt(3), s/sqrt(3), s/sqrt(3)); //vertices are sqrt(3) units from center

   //glColor3f(0.6,0.2,0.4);

   // pentagons from the top down

   //top front
   glBegin(GL_POLYGON);
   glNormal3d(points[0][0]+points[8][0]+points[9][0]+points[1][0]+points[12][0],
              points[0][1]+points[8][1]+points[9][1]+points[1][1]+points[12][1],
              points[0][2]+points[8][2]+points[9][2]+points[1][2]+points[12][2]);
   glVertex3d(points[0][0], points[0][1], points[0][2]);
   glVertex3d(points[8][0], points[8][1], points[8][2]);
   glVertex3d(points[9][0], points[9][1], points[9][2]);
   glVertex3d(points[1][0], points[1][1], points[1][2]);
   glVertex3d(points[12][0], points[12][1], points[12][2]);
   glEnd();

   //top back
   glBegin(GL_POLYGON);
   glNormal3d(points[2][0]+points[9][0]+points[8][0]+points[3][0]+points[14][0],
              points[2][1]+points[9][1]+points[8][1]+points[3][1]+points[14][1],
              points[2][2]+points[9][2]+points[8][2]+points[3][2]+points[14][2]);
   glVertex3d(points[2][0], points[2][1], points[2][2]);
   glVertex3d(points[9][0], points[9][1], points[9][2]);
   glVertex3d(points[8][0], points[8][1], points[8][2]);
   glVertex3d(points[3][0], points[3][1], points[3][2]);
   glVertex3d(points[14][0], points[14][1], points[14][2]);
   glEnd();

   //left top
   glBegin(GL_POLYGON);
   glNormal3d(points[2][0]+points[19][0]+points[18][0]+points[1][0]+points[9][0],
              points[2][1]+points[19][1]+points[18][1]+points[1][1]+points[9][1],
              points[2][2]+points[19][2]+points[18][2]+points[1][2]+points[9][2]);
   glVertex3d(points[2][0], points[2][1], points[2][2]);
   glVertex3d(points[19][0], points[19][1], points[19][2]);
   glVertex3d(points[18][0], points[18][1], points[18][2]);
   glVertex3d(points[1][0], points[1][1], points[1][2]);
   glVertex3d(points[9][0], points[9][1], points[9][2]);
   glEnd();

   //right top
   glBegin(GL_POLYGON);
   glNormal3d(points[0][0]+points[16][0]+points[17][0]+points[3][0]+points[8][0],
              points[0][1]+points[16][1]+points[17][1]+points[3][1]+points[8][1],
              points[0][2]+points[16][2]+points[17][2]+points[3][2]+points[8][2]);
   glVertex3d(points[0][0], points[0][1], points[0][2]);
   glVertex3d(points[16][0], points[16][1], points[16][2]);
   glVertex3d(points[17][0], points[17][1], points[17][2]);
   glVertex3d(points[3][0], points[3][1], points[3][2]);
   glVertex3d(points[8][0], points[8][1], points[8][2]);
   glEnd();

   //front right
   glBegin(GL_POLYGON);
   glNormal3d(points[0][0]+points[12][0]+points[13][0]+points[4][0]+points[16][0],
              points[0][1]+points[12][1]+points[13][1]+points[4][1]+points[16][1],
              points[0][2]+points[12][2]+points[13][2]+points[4][2]+points[16][2]);
   glVertex3d(points[0][0], points[0][1], points[0][2]);
   glVertex3d(points[12][0], points[12][1], points[12][2]);
   glVertex3d(points[13][0], points[13][1], points[13][2]);
   glVertex3d(points[4][0], points[4][1], points[4][2]);
   glVertex3d(points[16][0], points[16][1], points[16][2]);
   glEnd();

   //front left
   glBegin(GL_POLYGON);
   glNormal3d(points[5][0]+points[13][0]+points[12][0]+points[1][0]+points[18][0],
              points[5][1]+points[13][1]+points[12][1]+points[1][1]+points[18][1],
              points[5][2]+points[13][2]+points[12][2]+points[1][2]+points[18][2]);
   glVertex3d(points[5][0], points[5][1], points[5][2]);
   glVertex3d(points[13][0], points[13][1], points[13][2]);
   glVertex3d(points[12][0], points[12][1], points[12][2]);
   glVertex3d(points[1][0], points[1][1], points[1][2]);
   glVertex3d(points[18][0], points[18][1], points[18][2]);
   glEnd();

   //back left
   glBegin(GL_POLYGON);
   glNormal3d(points[2][0]+points[14][0]+points[15][0]+points[6][0]+points[19][0],
              points[2][1]+points[14][1]+points[15][1]+points[6][1]+points[19][1],
              points[2][2]+points[14][2]+points[15][2]+points[6][2]+points[19][2]);
   glVertex3d(points[2][0], points[2][1], points[2][2]);
   glVertex3d(points[14][0], points[14][1], points[14][2]);
   glVertex3d(points[15][0], points[15][1], points[15][2]);
   glVertex3d(points[6][0], points[6][1], points[6][2]);
   glVertex3d(points[19][0], points[19][1], points[19][2]);
   glEnd();

   //back right
   glBegin(GL_POLYGON);
   glNormal3d(points[7][0]+points[15][0]+points[14][0]+points[3][0]+points[17][0],
              points[7][1]+points[15][1]+points[14][1]+points[3][1]+points[17][1],
              points[7][2]+points[15][2]+points[14][2]+points[3][2]+points[17][2]);
   glVertex3d(points[7][0], points[7][1], points[7][2]);
   glVertex3d(points[15][0], points[15][1], points[15][2]);
   glVertex3d(points[14][0], points[14][1], points[14][2]);
   glVertex3d(points[3][0], points[3][1], points[3][2]);
   glVertex3d(points[17][0], points[17][1], points[17][2]);
   glEnd();

   //left bottom
   glBegin(GL_POLYGON);
   glNormal3d(points[5][0]+points[18][0]+points[19][0]+points[6][0]+points[11][0],
              points[5][1]+points[18][1]+points[19][1]+points[6][1]+points[11][1],
              points[5][2]+points[18][2]+points[19][2]+points[6][2]+points[11][2]);
   glVertex3d(points[5][0], points[5][1], points[5][2]);
   glVertex3d(points[18][0], points[18][1], points[18][2]);
   glVertex3d(points[19][0], points[19][1], points[19][2]);
   glVertex3d(points[6][0], points[6][1], points[6][2]);
   glVertex3d(points[11][0], points[11][1], points[11][2]);
   glEnd();

   //right bottom
   glBegin(GL_POLYGON);
   glNormal3d(points[7][0]+points[17][0]+points[16][0]+points[4][0]+points[10][0],
              points[7][1]+points[17][1]+points[16][1]+points[4][1]+points[10][1],
              points[7][2]+points[17][2]+points[16][2]+points[4][2]+points[10][2]);
   glVertex3d(points[7][0], points[7][1], points[7][2]);
   glVertex3d(points[17][0], points[17][1], points[17][2]);
   glVertex3d(points[16][0], points[16][1], points[16][2]);
   glVertex3d(points[4][0], points[4][1], points[4][2]);
   glVertex3d(points[10][0], points[10][1], points[10][2]);
   glEnd();

   //bottom front
   glBegin(GL_POLYGON);
   glNormal3d(points[5][0]+points[11][0]+points[10][0]+points[4][0]+points[13][0],
              points[5][1]+points[11][1]+points[10][1]+points[4][1]+points[13][1],
              points[5][2]+points[11][2]+points[10][2]+points[4][2]+points[13][2]);
   glVertex3d(points[5][0], points[5][1], points[5][2]);
   glVertex3d(points[11][0], points[11][1], points[11][2]);
   glVertex3d(points[10][0], points[10][1], points[10][2]);
   glVertex3d(points[4][0], points[4][1], points[4][2]);
   glVertex3d(points[13][0], points[13][1], points[13][2]);
   glEnd();

   //bottom back
   glBegin(GL_POLYGON);
   glNormal3d(points[7][0]+points[10][0]+points[11][0]+points[6][0]+points[15][0],
              points[7][1]+points[10][1]+points[11][1]+points[6][1]+points[15][1],
              points[7][2]+points[10][2]+points[11][2]+points[6][2]+points[15][2]);
   glVertex3d(points[7][0], points[7][1], points[7][2]);
   glVertex3d(points[10][0], points[10][1], points[10][2]);
   glVertex3d(points[11][0], points[11][1], points[11][2]);
   glVertex3d(points[6][0], points[6][1], points[6][2]);
   glVertex3d(points[15][0], points[15][1], points[15][2]);
   glEnd();


   //WIREFRAME
   //glColor3f(1,1,1);
   //glBegin(GL_LINE_STRIP);
   //glVertex3d(points[8][0], points[8][1], points[8][2]);
   //glVertex3d(points[0][0], points[0][1], points[0][2]);
   //glVertex3d(points[16][0], points[16][1], points[16][2]);
   //glVertex3d(points[17][0], points[17][1], points[17][2]);
   //glVertex3d(points[3][0], points[3][1], points[3][2]);
   //glVertex3d(points[14][0], points[14][1], points[14][2]);
   //glVertex3d(points[15][0], points[15][1], points[15][2]);
   //glVertex3d(points[7][0], points[7][1], points[7][2]);
   //glVertex3d(points[10][0], points[10][1], points[10][2]);
   //glVertex3d(points[4][0], points[4][1], points[4][2]);
   //glVertex3d(points[13][0], points[13][1], points[13][2]);
   //glVertex3d(points[12][0], points[12][1], points[12][2]);
   //glVertex3d(points[1][0], points[1][1], points[1][2]);
   //glVertex3d(points[18][0], points[18][1], points[18][2]);
   //glVertex3d(points[5][0], points[5][1], points[5][2]);
   //glVertex3d(points[11][0], points[11][1], points[11][2]);
   //glVertex3d(points[6][0], points[6][1], points[6][2]);
   //glVertex3d(points[19][0], points[19][1], points[19][2]);
   //glVertex3d(points[2][0], points[2][1], points[2][2]);
   //glVertex3d(points[9][0], points[9][1], points[9][2]);
   //glVertex3d(points[8][0], points[8][1], points[8][2]);
   //glEnd();

   //glBegin(GL_LINES);

   //glVertex3d(points[8][0], points[8][1], points[8][2]);
   //glVertex3d(points[3][0], points[3][1], points[3][2]);

   //glVertex3d(points[2][0], points[2][1], points[2][2]);
   //glVertex3d(points[14][0], points[14][1], points[14][2]);

   //glVertex3d(points[6][0], points[6][1], points[6][2]);
   //glVertex3d(points[15][0], points[15][1], points[15][2]);

   //glVertex3d(points[10][0], points[10][1], points[10][2]);
   //glVertex3d(points[11][0], points[11][1], points[11][2]);

   //glVertex3d(points[5][0], points[5][1], points[5][2]);
   //glVertex3d(points[13][0], points[13][1], points[13][2]);

   //glVertex3d(points[7][0], points[7][1], points[7][2]);
   //glVertex3d(points[17][0], points[17][1], points[17][2]);

   //glVertex3d(points[4][0], points[4][1], points[4][2]);
   //glVertex3d(points[16][0], points[16][1], points[16][2]);

   //glVertex3d(points[0][0], points[0][1], points[0][2]);
   //glVertex3d(points[12][0], points[12][1], points[12][2]);

   //glVertex3d(points[1][0], points[1][1], points[1][2]);
   //glVertex3d(points[9][0], points[9][1], points[9][2]);

   //glVertex3d(points[18][0], points[18][1], points[18][2]);
   //glVertex3d(points[19][0], points[19][1], points[19][2]);

   //glEnd();

   glPopMatrix();
}

void icosahedron(double x, double y, double z,
                 double r,
                 double s)
{
   //points plotted using polar coordinates
   double lat = atan(0.5)*180/3.1415927;
   double points[12][2];
   points[0][0] = 0;       points[0][1] = 90; 
   points[1][0] = 0;       points[1][1] = lat; 
   points[2][0] = 36;      points[2][1] = -lat; 
   points[3][0] = 72;      points[3][1] = lat; 
   points[4][0] = 108;     points[4][1] = -lat; 
   points[5][0] = 144;     points[5][1] = lat; 
   points[6][0] = 180;     points[6][1] = -lat; 
   points[7][0] = 216;     points[7][1] = lat; 
   points[8][0] = 252;     points[8][1] = -lat; 
   points[9][0] = 288;     points[9][1] = lat; 
   points[10][0]= 324;     points[10][1]= -lat; 
   points[11][0]= 0;       points[11][1]= -90; 

   glPushMatrix();
   glTranslated(x, y, z);
   glRotated(r, 0,1,0);
   glScaled(s, s, s);

   //glColor3f(0.2,0.4,0.6);

   glBegin(GL_TRIANGLES);
   pNormal(points[3][0], points[3][1],
           points[0][0], points[0][1],
           points[1][0], points[1][0]);
   Vertex(points[3][0], points[3][1]);
   Vertex(points[0][0], points[0][1]);
   Vertex(points[1][0], points[1][1]);
   glEnd();

   glBegin(GL_TRIANGLES);
   pNormal(points[5][0], points[5][1],
           points[0][0], points[0][1],
           points[3][0], points[3][1]);
   Vertex(points[5][0], points[5][1]);
   Vertex(points[0][0], points[0][1]);
   Vertex(points[3][0], points[3][1]);
   glEnd();

   glBegin(GL_TRIANGLES);
   pNormal(points[7][0], points[7][1],
           points[0][0], points[0][1],
           points[5][0], points[5][1]);
   Vertex(points[7][0], points[7][1]);
   Vertex(points[0][0], points[0][1]);
   Vertex(points[5][0], points[5][1]);
   glEnd();

   glBegin(GL_TRIANGLES);
   pNormal(points[9][0], points[9][1],
           points[0][0], points[0][1],
           points[7][0], points[7][1]);
   Vertex(points[9][0], points[9][1]);
   Vertex(points[0][0], points[0][1]);
   Vertex(points[7][0], points[7][1]);
   glEnd();

   glBegin(GL_TRIANGLES);
   pNormal(points[1][0], points[1][1],
           points[0][0], points[0][1],
           points[9][0], points[9][1]);
   Vertex(points[1][0], points[1][1]);
   Vertex(points[0][0], points[0][1]);
   Vertex(points[9][0], points[9][1]);
   glEnd();

   glBegin(GL_TRIANGLES);
   pNormal(points[1][0], points[1][1],
           points[2][0], points[2][1],
           points[3][0], points[3][1]);
   Vertex(points[1][0], points[1][1]);
   Vertex(points[2][0], points[2][1]);
   Vertex(points[3][0], points[3][1]);
   glEnd();

   glBegin(GL_TRIANGLES);
   pNormal(points[3][0], points[3][1],
           points[4][0], points[4][1],
           points[5][0], points[5][1]);
   Vertex(points[3][0], points[3][1]);
   Vertex(points[4][0], points[4][1]);
   Vertex(points[5][0], points[5][1]);
   glEnd();

   glBegin(GL_TRIANGLES);
   pNormal(points[5][0], points[5][1],
           points[6][0], points[6][1],
           points[7][0], points[7][1]);
   Vertex(points[5][0], points[5][1]);
   Vertex(points[6][0], points[6][1]);
   Vertex(points[7][0], points[7][1]);
   glEnd();

   glBegin(GL_TRIANGLES);
   pNormal(points[7][0], points[7][1],
           points[8][0], points[8][1],
           points[9][0], points[9][1]);
   Vertex(points[7][0], points[7][1]);
   Vertex(points[8][0], points[8][1]);
   Vertex(points[9][0], points[9][1]);
   glEnd();

   glBegin(GL_TRIANGLES);
   pNormal(points[9][0], points[9][1],
           points[10][0], points[10][1],
           points[1][0], points[1][1]);
   Vertex(points[9][0], points[9][1]);
   Vertex(points[10][0], points[10][1]);
   Vertex(points[1][0], points[1][1]);
   glEnd();

   glBegin(GL_TRIANGLES);
   pNormal(points[10][0], points[10][1],
           points[9][0], points[9][1],
           points[8][0], points[8][1]);
   Vertex(points[10][0], points[10][1]);
   Vertex(points[9][0], points[9][1]);
   Vertex(points[8][0], points[8][1]);
   glEnd();

   glBegin(GL_TRIANGLES);
   pNormal(points[8][0], points[8][1],
           points[7][0], points[7][1],
           points[6][0], points[6][1]);
   Vertex(points[8][0], points[8][1]);
   Vertex(points[7][0], points[7][1]);
   Vertex(points[6][0], points[6][1]);
   glEnd();

   glBegin(GL_TRIANGLES);
   pNormal(points[6][0], points[6][1],
           points[5][0], points[5][1],
           points[4][0], points[4][1]);
   Vertex(points[6][0], points[6][1]);
   Vertex(points[5][0], points[5][1]);
   Vertex(points[4][0], points[4][1]);
   glEnd();

   glBegin(GL_TRIANGLES);
   pNormal(points[4][0], points[4][1],
           points[3][0], points[3][1],
           points[2][0], points[2][1]);
   Vertex(points[4][0], points[4][1]);
   Vertex(points[3][0], points[3][1]);
   Vertex(points[2][0], points[2][1]);
   glEnd();

   glBegin(GL_TRIANGLES);
   pNormal(points[2][0], points[2][1],
           points[1][0], points[1][1],
           points[10][0], points[10][1]);
   Vertex(points[2][0], points[2][1]);
   Vertex(points[1][0], points[1][1]);
   Vertex(points[10][0], points[10][1]);
   glEnd();

   glBegin(GL_TRIANGLES);
   pNormal(points[2][0], points[2][1],
           points[11][0], points[11][1],
           points[4][0], points[4][1]);
   Vertex(points[2][0], points[2][1]);
   Vertex(points[11][0], points[11][1]);
   Vertex(points[4][0], points[4][1]);
   glEnd();

   glBegin(GL_TRIANGLES);
   pNormal(points[4][0], points[4][1],
           points[11][0], points[11][1],
           points[6][0], points[6][1]);
   Vertex(points[4][0], points[4][1]);
   Vertex(points[11][0], points[11][1]);
   Vertex(points[6][0], points[6][1]);
   glEnd();

   glBegin(GL_TRIANGLES);
   pNormal(points[6][0], points[6][1],
           points[11][0], points[11][1],
           points[8][0], points[8][1]);
   Vertex(points[6][0], points[6][1]);
   Vertex(points[11][0], points[11][1]);
   Vertex(points[8][0], points[8][1]);
   glEnd();

   glBegin(GL_TRIANGLES);
   pNormal(points[8][0], points[8][1],
           points[11][0], points[11][1],
           points[10][0], points[10][1]);
   Vertex(points[8][0], points[8][1]);
   Vertex(points[11][0], points[11][1]);
   Vertex(points[10][0], points[10][1]);
   glEnd();

   glBegin(GL_TRIANGLES);
   pNormal(points[10][0], points[10][1],
           points[11][0], points[11][1],
           points[2][0], points[2][1]);
   Vertex(points[10][0], points[10][1]);
   Vertex(points[11][0], points[11][1]);
   Vertex(points[2][0], points[2][1]);
   glEnd();


   //WIREFRAME
   //glColor3f(1,1,1);

   ////top fan of edges
   //glBegin(GL_LINES);
   //for (i = 1; i <= 9; i += 2)
   //{
   //   Vertex(points[0][0], points[0][1]);
   //   Vertex(points[i][0], points[i][1]);
   //}
   //glEnd();
   ////top ring of edges
   //glBegin(GL_LINE_STRIP);
   //for (i = 1; i <= 9; i += 2)
   //   Vertex(points[i][0], points[i][1]);
   ////middle zigzag edges
   //for (i = 1; i <= 10; ++i)
   //   Vertex(points[i][0], points[i][1]);
   //Vertex(points[1][0], points[1][1]);
   //glEnd();
   ////bottom ring of edges
   //glBegin(GL_LINE_STRIP);
   //for (i = 2; i <= 10; i += 2)
   //   Vertex(points[i][0], points[i][1]);
   //Vertex(points[2][0], points[2][1]);
   //glEnd();
   ////bottom fan of edges
   //glBegin(GL_LINES);
   //for (i = 2; i <= 10; i += 2)
   //{
   //   Vertex(points[11][0], points[11][1]);
   //   Vertex(points[i][0], points[i][1]);
   //}
   //glEnd();

   glPopMatrix();
}
