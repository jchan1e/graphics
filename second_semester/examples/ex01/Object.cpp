//
//  Generic Object class
//
#include "Object.h"

//
//  Constructor
//
Object::Object(float x,float y,float z)
{
   x0 = x;
   y0 = y;
   z0 = z;
}

//
//  Set translation
//
void Object::translate(float x,float y,float z)
{
   x0 = x;
   y0 = y;
   z0 = z;
}

//
//  Apply ambient and diffuse color
//  Specular is set to white
//  Emission is set to black
//
void Object::setColor(Color c)
{
   setColor(c,c,Color(1,1,1),Color(0,0,0),16);
}

//
//  Apply colors
//
void Object::setColor(Color a,Color d,Color s,Color e,float Ns)
{
   glColor4fv(d.fv());
   glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT ,a.fv());
   glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE ,d.fv());
   glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,s.fv());
   glMaterialfv(GL_FRONT_AND_BACK,GL_EMISSION,e.fv());
   glMaterialfv(GL_FRONT_AND_BACK,GL_SHININESS,&Ns);
}

