// 
// functions for drawing various objects
//

#ifndef STDIncludes
#define STDIncludes
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#endif

//Cos and Sin in degrees - stolen from ex8
#define Sin(x) (sin((x) * 3.1415927/180))
#define Cos(x) (cos((x) * 3.1415927/180))

//Vertex Polar-Cartesian tranformation function also pilfered from ex8 
void Vertex(double th, double ph)
{
   glColor3f(fabs(Cos(th)*Cos(ph)), fabs(Sin(th)*Cos(ph)), fabs(Sin(ph)));
   glVertex3d(Sin(th)*Cos(ph), Sin(ph), Cos(th)*Cos(ph));
}

//without the coloration
void Vertex2(double th, double ph)
{
   glVertex3d(Sin(th)*Cos(ph), Sin(ph), Cos(th)*Cos(ph));
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
   Vertex(0,90);
   for(int th = 0; th <= 360; th += 5)
   {
      Vertex(th, 85);
   }
   glEnd();

   //latitude rings
   for (int ph = 85; ph >= -80; ph -= 5)
   {
      glBegin(GL_QUAD_STRIP);
      for (int th = 0; th <= 360; th += 5)
      {
         Vertex(th, ph);
         Vertex(th, ph-5);
      }
      glEnd();
   }
   
   //bottom fan
   glBegin(GL_TRIANGLE_FAN);
   Vertex(0,-90);
   for(int th = 360; th >= 0; th -= 5)
   {
      Vertex(th, -85);
   }
   glEnd();

   glPopMatrix();
}

void icosahedron(double x, double y, double z,
                 double r,
                 double s)
{
   double lat = atan(0.5)*180/3.1415927;
   int i = 0;
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

   glColor3f(0.1,0.2,0.3);

   //top 5 triangles
   glBegin(GL_TRIANGLE_FAN);
   Vertex2(points[0][0], points[0][1]);
   for (i = 1; i < 10; i += 2)
      Vertex2(points[i][0], points[i][1]);
   Vertex2(points[1][0], points[1][1]);
   glEnd();

   //middle 10 triangles
   glBegin(GL_TRIANGLE_STRIP);
   for (i = 1; i <= 10; ++i)
      Vertex2(points[i][0], points[i][1]);
   Vertex2(points[1][0], points[1][1]);
   Vertex2(points[2][0], points[2][1]);
   glEnd();

   //bottom 5 triangles
   glBegin(GL_TRIANGLE_FAN);
   Vertex2(points[11][0], points[11][1]);
   for (i = 10; i > 1; i -= 2)
      Vertex2(points[i][0], points[i][1]);
   Vertex2(points[11][0], points[11][1]);
   glEnd();

   glColor3f(1,1,1);
   glLineWidth(1.5);

   //top fan of edges
   glBegin(GL_LINES);
   for (i = 1; i <= 9; i += 2)
   {
      Vertex2(points[0][0], points[0][1]);
      Vertex2(points[i][0], points[i][1]);
   }
   glEnd();
   //top ring of edges
   glBegin(GL_LINE_STRIP);
   for (i = 1; i <= 9; i += 2)
      Vertex2(points[i][0], points[i][1]);
   //middle zigzag edges
   for (i = 1; i <= 10; ++i)
      Vertex2(points[i][0], points[i][1]);
   Vertex2(points[1][0], points[1][1]);
   glEnd();
   //bottom ring of edges
   glBegin(GL_LINE_STRIP);
   for (i = 2; i <= 10; i += 2)
      Vertex2(points[i][0], points[i][1]);
   Vertex2(points[2][0], points[2][1]);
   glEnd();
   //bottom fan of edges
   glBegin(GL_LINES);
   for (i = 2; i <= 10; i += 2)
   {
      Vertex2(points[11][0], points[11][1]);
      Vertex2(points[i][0], points[i][1]);
   }
   glEnd();

   glPopMatrix();
}
