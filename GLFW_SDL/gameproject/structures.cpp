#include "structures.h"

Floor::Floor()
{
//   int newarr[81] = {0, 0, 0, 0, 0, 0, 0, 0, 0,  // it's a 2nd order Hilbert curve
//                    -1,-1,-1,-1, 0,-1,-1,-1,-1, 
//                     0, 0, 0,-1, 0,-1, 0, 0, 0, 
//                     0,-1,-1,-1, 0,-1,-1,-1, 0, 
//                     0,-1, 0, 0, 0, 0, 0,-1, 0, 
//                     0,-1, 0,-1,-1,-1, 0,-1, 0, 
//                     0,-1, 0,-1, 0,-1, 0,-1, 0, 
//                     0,-1,-1,-1, 0,-1,-1,-1, 0, 
//                     0, 0, 0, 0, 0, 0, 0, 0, 0};
//   arr = newarr;

   startx = 0;
   starty = 1;
}

void Floor::tile(float x, float y, float z, int direction)
{
   glPushMatrix();

   glTranslated(x, y, z);

   if (direction == 2)      //front
      glRotated(90, 1,0,0);
   else if (direction == 3) //left
      glRotated(90, 0,0,1);
   else if (direction == 4) //back
      glRotated(90, -1,0,0);
   else if (direction == 5) //right
      glRotated(90, 0,0,-1);
   else if (direction == 1) //down
      glRotated(180, 1,0,0);
                            //otherwise up
   glBegin(GL_TRIANGLES);
   glNormal3d(0.05,1,0);
   glVertex3d(0,1.05,0);
   glVertex3d(1,1,1);
   glVertex3d(1,1,-1);

   glNormal3d(0,1,-0.05);
   glVertex3d(0,1.05,0);
   glVertex3d(1,1,-1);
   glVertex3d(-1,1,-1);

   glNormal3d(-0.05,1,0);
   glVertex3d(0,1.05,0);
   glVertex3d(-1,1,-1);
   glVertex3d(-1,1,1);

   glNormal3d(0,1,0.05);
   glVertex3d(0,1.05,0);
   glVertex3d(-1,1,1);
   glVertex3d(1,1,1);
   glEnd();

   glPopMatrix();
}

void Floor::render()
{
   glPushMatrix();
   glTranslated(-8,0,-8);

   for (int i=0; i<9; ++i)
   {
      for (int j=0; j<9; ++j)
      {
         tile(2*j, 2*arr[9*i+j], 2*i, 0);
         //tile(2*i, -1, 2*j, 0);
         switch (i)
         {
         case 0:
            tile(2*j, 0, 2*i, 4);
            if (arr[9*i+j] == 0 && arr[9*(i+1)+j] == -1)
               tile(2*j, arr[9*i+j], 2*i, 2);
            break;
         case 8:
            tile(2*j, 0, 2*i, 2);
            if (arr[9*i+j] == 0 && arr[9*(i-1)+j] == -1)
               tile(2*j, arr[9*i+j], 2*i, 4);
            break;
         default:
            if (arr[9*i+j] == 0 && arr[9*(i+1)+j] == -1)
               tile(2*j, arr[9*i+j], 2*i, 2);
            if (arr[9*i+j] == 0 && arr[9*(i-1)+j] == -1)
               tile(2*j, arr[9*i+j], 2*i, 4);
            break;
         }
         switch (j)
         {
         case 0:
            tile(2*j, 0, 2*i, 3);
            if (arr[9*i+j] == 0 && arr[9*i+(j+1)] == -1)
               tile(2*j, arr[9*i+j], 2*i, 5);
            break;
         case 8:
            tile(2*j, 0, 2*i, 5);
            if (arr[9*i+j] == 0 && arr[9*i+(j-1)] == -1)
               tile(2*j, arr[9*i+j], 2*i, 3);
            break;
         default:
            if (arr[9*i+j] == 0 && arr[9*i+(j+1)] == -1)
               tile(2*j, arr[9*i+j], 2*i, 5);
            if (arr[9*i+j] == 0 && arr[9*i+(j-1)] == -1)
               tile(2*j, arr[9*i+j], 2*i, 3);
            break;
         }
      }
   }
   glPopMatrix();
}

Enemy::Enemy(float ix, float iy, int ihealth)
{
   health = ihealth;
   x = ix;
   y = iy;
   z = 0;
   theta = 0.0;
}

void Enemy::render()
{
   glColor3f(0.8,0.0,0.8);
   dodecahedron(x, z, -y, theta, 1.0);
   glColor3f(0.0,0.0,1.0);
   icosahedron(x, z, -y, theta, 1.0);
}

void Enemy::animate()
{
}

void Enemy::damage(int dmg)
{
}

Tower::Tower(float x, float y)
{
}

void Tower::render()
{
}

void Tower::animate()
{
}

void Tower::fire()
{
}

Bullet::Bullet(float x, float y, float z)
{
}

void Bullet::render()
{
}

void Bullet::animate()
{
}

void Bullet::collide(Enemy* target)
{
}
