//
//  Sphere class
//  The constructor sets the position, radius and color
//  All parameters are assigned default values
//
#ifndef SPHERE_H
#define SPHERE_H

#include "Object.h"

class Sphere: public Object
{
private:
   float r0;        //  Radius
   float R,G,B;     //  Color
public:
   Sphere(float x=0,float y=0,float z=0,float r=1,float R=1,float G=1,float B=1);
   void radius(float r);                    //  Set radius
   void display();                          //  Render the sphere
};

#endif
