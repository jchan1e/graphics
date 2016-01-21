//
//  Cube class
//  The constructor sets the position, size and rotation
//  All parameters are assigned default values
//
#ifndef CUBE_H
#define CUBE_H

#include "Object.h"

class Cube: public Object
{
private:
   float Sx,Sy,Sz;  //  Dimensions in X/Y/Z directions
   float Th;        //  Rotation around Y axis
public:
   Cube(float x=0,float y=0,float z=0,float dx=1,float dy=1,float dz=1,float th=0);
   void rotate(float th);                   //  Set rotation
   void scale(float dx,float dy,float dz);  //  Set scale
   void display();                          //  Render the cube
};

#endif
