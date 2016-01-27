//
//  Ex02viewer Widget
//

#include <QSlider>
#include <QComboBox>
#include <QLabel>
#include <QGridLayout>
#include "ex02viewer.h"
#include "ex02opengl.h"

//
//  Constructor
//
Ex02viewer::Ex02viewer()
{
   //  Set window title
   setWindowTitle(tr("Ex02:  NDC to RGB Shader"));

   //  Create new Triangle widget
   Ex02opengl* ogl = new Ex02opengl;

   //  Select shader
   QComboBox* shader = new QComboBox();
   shader->addItem("None");
   shader->addItem("Vertex and Fragment");
   shader->addItem("Vertex Only");
   shader->addItem("Fragment Only");

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
   QSlider* Xpos = new QSlider(Qt::Horizontal);
   QSlider* Ypos = new QSlider(Qt::Horizontal);
   QSlider* Zpos = new QSlider(Qt::Horizontal);
   Xpos->setRange(-100,100);
   Ypos->setRange(-100,100);
   Zpos->setRange(-100,100);

   //  View angle and center
   QLabel* angles = new QLabel();
   QLabel* center = new QLabel();

   //  Quit
   QPushButton* quit = new QPushButton("Quit");

   //  Set layout of child widgets
   QGridLayout* layout = new QGridLayout;
   layout->addWidget(ogl,0,0,10,1);
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
   layout->addWidget(new QLabel("Z Position"),5,1);
   layout->addWidget(Zpos,5,2);
   layout->addWidget(new QLabel("Angles"),6,1);
   layout->addWidget(angles,6,2);
   layout->addWidget(new QLabel("Center"),7,1);
   layout->addWidget(center,7,2);
   layout->addWidget(quit,9,2);
   //  Manage resizing
   layout->setColumnStretch(0,100);
   layout->setColumnMinimumWidth(0,100);
   layout->setRowStretch(8,100);
   setLayout(layout);

   //  Connect valueChanged() signals to ogl
   connect(shader,SIGNAL(currentIndexChanged(int))     , ogl,SLOT(setShader(int)));
   connect(object,SIGNAL(currentIndexChanged(int))     , ogl,SLOT(setObject(int)));
   connect(projection,SIGNAL(currentIndexChanged(int)) , ogl,SLOT(setPerspective(int)));
   connect(Xpos,SIGNAL(valueChanged(int)) , ogl,SLOT(setX(int)));
   connect(Ypos,SIGNAL(valueChanged(int)) , ogl,SLOT(setY(int)));
   connect(Zpos,SIGNAL(valueChanged(int)) , ogl,SLOT(setZ(int)));
   //  Connect angles() and center() signal to labels
   connect(ogl,SIGNAL(angles(QString)) , angles,SLOT(setText(QString)));
   connect(ogl,SIGNAL(center(QString)) , center,SLOT(setText(QString)));
   //  Connect quit() signal to qApp::quit()
   connect(quit,SIGNAL(pressed()) , qApp,SLOT(quit()));
}
