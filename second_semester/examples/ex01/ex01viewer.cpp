//
//  Ex01viewer Widget
//

#include <QComboBox>
#include <QLabel>
#include <QGridLayout>
#include "ex01viewer.h"
#include "ex01opengl.h"

//
//  Constructor
//
Ex01viewer::Ex01viewer()
{
   //  Set window title
   setWindowTitle(tr("Ex01:  Basic Shader"));

   //  Create new OpenGL widget
   Ex01opengl* ex01opengl = new Ex01opengl;

   //  Select shader
   QComboBox* shader = new QComboBox();
   shader->addItem("None");
   shader->addItem("Basic");

   //  Select projection
   QComboBox* projection = new QComboBox();
   projection->addItem("Orthogonal");
   projection->addItem("Perspective");

   //  Select object
   QComboBox* object = new QComboBox();
   object->addItem("Cube");
   object->addItem("Teapot");
   object->addItem("Tyra");

   //  Lighting
   QCheckBox* lighting = new QCheckBox("");

   //  View angle
   QLabel* angles = new QLabel();

   //  Quit
   QPushButton* quit = new QPushButton("Quit");

   //  Set layout of child widgets
   QGridLayout* layout = new QGridLayout;
   layout->addWidget(ex01opengl,0,0,7,1);
   layout->addWidget(new QLabel("Shader"),0,1);
   layout->addWidget(shader,0,2);
   layout->addWidget(new QLabel("Projection"),1,1);
   layout->addWidget(projection,1,2);
   layout->addWidget(new QLabel("Object"),2,1);
   layout->addWidget(object,2,2);
   layout->addWidget(new QLabel("Lighting"),3,1);
   layout->addWidget(lighting,3,2);
   layout->addWidget(new QLabel("Angles"),4,1);
   layout->addWidget(angles,4,2);
   layout->addWidget(quit,6,2);
   //  Manage resizing
   layout->setColumnStretch(0,100);
   layout->setColumnMinimumWidth(0,100);
   layout->setRowStretch(5,100);
   setLayout(layout);

   //  Connect valueChanged() signals to ex01opengl
   connect(shader,SIGNAL(currentIndexChanged(int))     , ex01opengl,SLOT(setShader(int)));
   connect(object,SIGNAL(currentIndexChanged(int))     , ex01opengl,SLOT(setObject(int)));
   connect(projection,SIGNAL(currentIndexChanged(int)) , ex01opengl,SLOT(setPerspective(int)));
   connect(lighting,SIGNAL(stateChanged(int))          , ex01opengl,SLOT(setLighting(int)));
   //  Connect angles() signal to label
   connect(ex01opengl,SIGNAL(angles(QString)) , angles,SLOT(setText(QString)));
   //  Connect quit() signal to qApp::quit()
   connect(quit,SIGNAL(pressed()) , qApp,SLOT(quit()));
}
