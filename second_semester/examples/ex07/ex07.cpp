//
//  Ex07:  OpenGL 4 Lighting and Textures
//  Willem A. (Vlakkies) Schreuder
//  CSCI 4239/5239 Spring 2016
//

#include <QApplication>
#include "ex07viewer.h"

//
//  Main function
//
int main(int argc, char *argv[])
{
   //  Create the application
   QApplication app(argc,argv);
   //  Create and show view widget
   Ex07viewer view;
   view.show();
   //  Main loop for application
   return app.exec();
}
