//
//  Ex 01:  Basic Shader
//  Willem A. (Vlakkies) Schreuder
//  CSCI 4239/5239 Spring 2016
//

#include <QApplication>
#include "ex01viewer.h"

//
//  Main function
//
int main(int argc, char *argv[])
{
   //  Create the application
   QApplication app(argc,argv);
   //  Create and show view widget
   Ex01viewer view;
   view.show();
   //  Main loop for application
   return app.exec();
}
