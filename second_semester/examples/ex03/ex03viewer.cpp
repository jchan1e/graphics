//
//  Ex03viewer Widget
//

#include <QComboBox>
#include <QLabel>
#include <QGridLayout>
#include "ex03viewer.h"

//
//  Constructor
//
Ex03viewer::Ex03viewer()
{
   //  Set window title
   setWindowTitle(tr("Ex03:  Procedural Textures"));

   //  Create new Triangle widget
   ogl = new Ex03opengl;

   //  Select shader
   QComboBox* shader = new QComboBox();
   shader->addItem("None");
   shader->addItem("Stored");
   shader->addItem("Brick");
   shader->addItem("Mandelbrot");

   //  Select projection
   QComboBox* projection = new QComboBox();
   projection->addItem("Orthogonal");
   projection->addItem("Perspective");

   //  Select object
   QComboBox* object = new QComboBox();
   object->addItem("Cube");
   object->addItem("Teapot");
   object->addItem("Suzanne");

   //  Center position
   Xpos = new QDoubleSpinBox();
   Ypos = new QDoubleSpinBox();
   Zpos = new QSlider(Qt::Horizontal);
   Xpos->setRange(-100,100);Xpos->setDecimals(6);Xpos->setSingleStep(0.1);
   Ypos->setRange(-100,100);Ypos->setDecimals(6);Ypos->setSingleStep(0.1);
   Zpos->setRange(0,100);

   //  View angle and zoom
   QLabel* angles = new QLabel();
   QLabel* zoom  = new QLabel();

   //  Pause/resume button
   light = new QPushButton("Pause");

   //  Reset
   QPushButton* rst = new QPushButton("Reset");
   //  Quit
   QPushButton* quit = new QPushButton("Quit");

   //  Set layout of child widgets
   QGridLayout* layout = new QGridLayout;
   layout->addWidget(ogl,0,0,11,1);
   layout->addWidget(new QLabel("Shader"),0,1);
   layout->addWidget(shader,0,2);
   layout->addWidget(new QLabel("Projection"),1,1);
   layout->addWidget(projection,1,2);
   layout->addWidget(new QLabel("Object"),2,1);
   layout->addWidget(object,2,2);
   layout->addWidget(new QLabel("X Position"),3,1);
   layout->addWidget(Xpos,3,2);
   layout->addWidget(new QLabel("Y Position"),4,1);
   layout->addWidget(Ypos,4,2);
   layout->addWidget(new QLabel("Zoom Level"),5,1);
   layout->addWidget(Zpos,5,2);
   layout->addWidget(new QLabel("Angles"),6,1);
   layout->addWidget(angles,6,2);
   layout->addWidget(new QLabel("Zoom"),7,1);
   layout->addWidget(zoom,7,2);
   layout->addWidget(new QLabel("Light"),8,1);
   layout->addWidget(light,8,2);
   layout->addWidget(rst,10,1);
   layout->addWidget(quit,10,2);
   //  Manage resizing
   layout->setColumnStretch(0,100);
   layout->setColumnMinimumWidth(0,100);
   layout->setRowStretch(9,100);
   setLayout(layout);

   //  Connect valueChanged() signals to ogl
   connect(shader,SIGNAL(currentIndexChanged(int))     , ogl,SLOT(setShader(int)));
   connect(object,SIGNAL(currentIndexChanged(int))     , ogl,SLOT(setObject(int)));
   connect(projection,SIGNAL(currentIndexChanged(int)) , ogl,SLOT(setPerspective(int)));
   connect(Xpos,SIGNAL(valueChanged(double)) , ogl,SLOT(setX(double)));
   connect(Ypos,SIGNAL(valueChanged(double)) , ogl,SLOT(setY(double)));
   connect(Zpos,SIGNAL(valueChanged(int)) , this,SLOT(izoom(int)));
   //  Connect angles() and zoom() signal to labels
   //  Connect angles() and zoom() signal to labels
   connect(ogl,SIGNAL(angles(QString)) , angles,SLOT(setText(QString)));
   connect(ogl,SIGNAL(zoom(QString))   , zoom,SLOT(setText(QString)));
   //  Connect reset() and lmove() signals
   connect(rst  ,SIGNAL(pressed()),this,SLOT(reset()));
   connect(light,SIGNAL(pressed()),this,SLOT(lmove()));
   //  Connect quit() signal to qApp::quit()
   connect(quit,SIGNAL(pressed()) , qApp,SLOT(quit()));
}

//
//  Reset view
//
void Ex03viewer::reset()
{
   Xpos->setValue(0);
   Ypos->setValue(0);
   Zpos->setValue(0);
   ogl->reset();
}

//
//  Light pause/move
//
void Ex03viewer::izoom(int iz)
{
   //  Map coordinates
   float z = powf(2.0,0.2*iz);
   ogl->setZoom(z);
   //  Set Increments
   int dec = iz/16+1;
   float step = powf(10.0,-dec);
   Xpos->setSingleStep(step);
   Ypos->setSingleStep(step);
}

//
//  Light pause/move
//
void Ex03viewer::lmove()
{
   bool pause = (light->text()=="Pause");
   if (pause)
      light->setText("Animate");
   else
      light->setText("Pause");
   ogl->setLightMove(!pause);
}
