// 
// manually draws various objects in a simple scene
//

#ifndef STDIncludes
#define STDIncludes
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#endif

void display()
{
}

void reshape(int width, int height)
{
}

void special(int key, int mousex, int mousey)
{
}

void keyboard(unsigned char key, int mousex, int mousey)
{
}

int main(int argc, char *argv[])
{
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);

   glutInitWindowPosition(0,0);
   glutInitWindowSize(500,500);
   glutCreateWindow("4229 - Jordan Dick: Simple Scene");

   glutDisplayFunc(display);
   glutReshapeFunc(reshape);
   glutSpecialFunc(special);
   glutKeyboardFunc(keyboard);
   //glutPassiveMotionFunc(motion);

   glutMainLoop();
}
