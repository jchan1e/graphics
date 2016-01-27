//
//  Teapot class
//  The constructor sets the position, size and rotation
//  All parameters are assigned default values
//
#ifndef TEAPOT_H
#define TEAPOT_H

#include "Object.h"

class Teapot: public Object
{
private:
   int   n;            //  Number of patches
   float s;            // Scale
   float R,G,B;        // Color
   float th,dx,dy,dz;  //  Rotation (angle and axis)
public:
   Teapot(int patches);                               //  Constructor
   void color(float R,float G,float B);               //  Set color
   void rotate(float th,float dx,float dy,float dz);  //  Set rotation
   void scale(float S) ;                              //  Set scale
   void display();                                    //  Render the teapot
};

#endif
