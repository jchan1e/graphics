//
//  OpenGL Ex02opengl widget
//
#include "ex02opengl.h"
#include <QtOpenGL>
#include <QMessageBox>
#include "Cube.h"
#include "WaveOBJ.h"
#include "Teapot.h"
#include <math.h>
#define Cos(th) cos(3.1415926/180*(th))
#define Sin(th) sin(3.1415926/180*(th))
#include <QVector3D>

//
//  Constructor
//
Ex02opengl::Ex02opengl(QWidget* parent)
    : QGLWidget(parent)
{
   mode = 0;
   init  = false;
   mouse = false;
   asp = 1;
   dim = 3;
   fov = 0;
   th = ph = 0;
   x0 = y0 = z0 = 0;
}

//
//  Set shader
//
void Ex02opengl::setShader(int sel)
{
   mode = sel;
   //  Request redisplay
   updateGL();
}

//
//  Set X position
//
void Ex02opengl::setX(int X)
{
   x0 = X;
   //  Request redisplay
   updateGL();
}

//
//  Set Y position
//
void Ex02opengl::setY(int Y)
{
   y0 = Y;
   //  Request redisplay
   updateGL();
}

//
//  Set Z position
//
void Ex02opengl::setZ(int Z)
{
   z0 = Z;
   //  Request redisplay
   updateGL();
}

//
//  Set projection
//
void Ex02opengl::setPerspective(int on)
{
   fov = on ? 55 : 0;
   Projection();
   //  Request redisplay
   updateGL();
}

//
//  Set object
//
void Ex02opengl::setObject(int type)
{
   obj = objects[type];
   //  Request redisplay
   updateGL();
}

//
//  Initialize
//
void Ex02opengl::initializeGL()
{
   if (init) return;
   init = true;

   //  Vertex and fragment shader
   if (!shader[1].addShaderFromSourceFile(QGLShader::Vertex,":/ex02.vert"))
      Fatal("Error compiling ex02.vert\n"+shader[1].log());
   if (!shader[1].addShaderFromSourceFile(QGLShader::Fragment,":/ex02a.frag"))
      Fatal("Error compiling ex02a.frag\n"+shader[1].log());
   if (!shader[1].link())
      Fatal("Error linking shader\n"+shader[1].log());

   //  Vertex Only Shader
   if (!shader[2].addShaderFromSourceFile(QGLShader::Vertex,":/ex02.vert"))
      Fatal("Error compiling ex02.vert\n"+shader[2].log());
   if (!shader[2].link())
      Fatal("Error linking shader\n"+shader[2].log());

   //  Fragment Only Shader
   if (!shader[3].addShaderFromSourceFile(QGLShader::Fragment,":/ex02b.frag"))
      Fatal("Error compiling ex02b.frag\n"+shader[2].log());
   if (!shader[3].link())
      Fatal("Error linking shader\n"+shader[3].log());

   // Cube
   objects.push_back(new Cube());

   // Teapot
   Teapot* pot = new Teapot(8);
   pot->scale(0.5);
   pot->color(0,1,1);
   objects.push_back(pot);

   // Suzanne
   WaveOBJ* suza=0;
   try
   {
      suza = new WaveOBJ("suzanne.obj",":/");
   }
   catch (QString err)
   {
      Fatal("Error loading object\n"+err);
   }
   if (suza)
   {
      suza->color(1,1,0);
      objects.push_back(suza);
   }

   //  Set initial object
   obj = objects[0];
}

//
//  Set projection when window is resized
//
void Ex02opengl::resizeGL(int width, int height)
{
   //  Window aspect ration
   asp = height ? width / (float)height : 1;
   //  Viewport is whole screen
   glViewport(0,0,width,height);
   //  Set projection
   Projection();
}

//
//  Draw the window
//
void Ex02opengl::paintGL()
{
   //  Clear screen and Z-buffer
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glEnable(GL_DEPTH_TEST);

   //  Set view
   glLoadIdentity();
   if (fov) glTranslated(0,0,-2*dim);
   glRotated(ph,1,0,0);
   glRotated(th,0,1,0);

   //  Apply shader
   if (mode)
   {
      shader[mode].bind();
      //  Dimensions
      QVector3D dim(width(),height(),1);
      shader[mode].setUniformValue("dim",dim);
   }

   //  Draw scene
   glPushMatrix();
   glTranslated(x0*dim/100,y0*dim/100,z0*dim/100);
   if (obj) obj->display();
   glPopMatrix();

   //  Release shader
   if (mode) shader[mode].release();

   //  Draw Axes
   glDisable(GL_DEPTH_TEST);
   glColor3f(1,1,1);
   glBegin(GL_LINES);
   glVertex3d(0,0,0);
   glVertex3d(1,0,0);
   glVertex3d(0,0,0);
   glVertex3d(0,1,0);
   glVertex3d(0,0,0);
   glVertex3d(0,0,1);
   glEnd();
   //  Label axes
   renderText(1,0,0,"X");
   renderText(0,1,0,"Y");
   renderText(0,0,1,"Z");

   
   //  Emit angles to display
   emit angles(QString::number(th)+","+QString::number(ph));
   //  Emit centers to display
   emit center(QString::number(x0)+","+QString::number(y0)+","+QString::number(z0));
}

//
//  Throw a fatal error and die
//
void Ex02opengl::Fatal(QString message)
{
   QMessageBox::critical(this,"Ex02opengl",message);
   QApplication::quit();
}

//
//  Set OpenGL projection
//
void Ex02opengl::Projection()
{
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   if (fov)
   {
      float zmin = dim/4;
      float zmax = 4*dim;
      float ydim = zmin*tan(fov*M_PI/360);
      float xdim = ydim*asp;
      glFrustum(-xdim,+xdim,-ydim,+ydim,zmin,zmax);
   }
   else
      glOrtho(-dim*asp, +dim*asp, -dim, +dim, -dim, +dim);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
}

/******************************************************************/
/*************************  Mouse Events  *************************/
/******************************************************************/
//
//  Mouse pressed
//
void Ex02opengl::mousePressEvent(QMouseEvent* e)
{
   mouse = true;
   pos = e->pos();  //  Remember mouse location
}

//
//  Mouse released
//
void Ex02opengl::mouseReleaseEvent(QMouseEvent*)
{
    mouse = false;
}

//
//  Mouse moved
//
void Ex02opengl::mouseMoveEvent(QMouseEvent* e)
{
   if (mouse)
   {
      QPoint d = e->pos()-pos;  //  Change in mouse location
      th = (th+d.x())%360;      //  Translate x movement to azimuth
      ph = (ph+d.y())%360;      //  Translate y movement to elevation
      pos = e->pos();           //  Remember new location
      updateGL();               //  Request redisplay
   }
}

//
//  Mouse wheel
//
void Ex02opengl::wheelEvent(QWheelEvent* e)
{
   //  Zoom out
   if (e->delta()<0)
      dim += 0.1;
   //  Zoom in
   else if (dim>1)
      dim -= 0.1;
   //  Request redisplay
   Projection();
   updateGL();
}
