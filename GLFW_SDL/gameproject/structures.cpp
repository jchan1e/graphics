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
            if (arr[9*i+j] == 0)
               tile(2*j, 0, 2*i, 3);
            if (arr[9*i+j] == 0 && arr[9*i+(j+1)] == -1)
               tile(2*j, arr[9*i+j], 2*i, 5);
            break;
         case 8:
            if (arr[9*i+j] == 0)
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

Enemy::Enemy(float X, float Y, int Health, int Type)
{
   type = Type;
   health = Health;
   x = X;
   y = Y;
   z = 0;
   theta = 0.0;
   if (type == 1)
   {
      s1 = 0.85;  ds1 = 0.02;
      s2 = 0.85;  ds2 = -0.02;
   }
   else
   {
      s1 = 0.8;  ds1 = 0.03;
      s2 = 0.8;  ds2 = -0.03;
   }
}

void Enemy::render()
{
   float emission[] = {0.0,0.0,0.0,1.0};

   //glColor3f(1.0,1.0,1.0);
   if (type == 1)
   {
      glColor3f(0.8,0.0,0.8);
      //glColor3f(0.0,0.8,0.0);
      emission[0] = 0.4; emission[1] = 0.0; emission[2] = 0.4;
      glMaterialfv(GL_FRONT, GL_EMISSION, emission);
      dodecahedron(x, z, -y, theta+16.6666, s1);

      glColor3f(0.0,0.0,1.0);
      //glColor3f(1.0,1.0,0.0);
      emission[0] = 0.0; emission[1] = 0.0; emission[2] = 0.6;
      glMaterialfv(GL_FRONT, GL_EMISSION, emission);
      icosahedron(x, z, -y, theta, s2);
   }
   else //type == 0
   {
      glColor3f(0.8,0.0,0.0);
      //glColor3f(0.0,0.8,0.8);
      emission[0] = 0.4; emission[1] = 0.0; emission[2] = 0.0;
      glMaterialfv(GL_FRONT, GL_EMISSION, emission);
      cube(x, z, -y, theta, s1*0.8);

      glColor3f(0.6,0.4,0.0);
      //glColor3f(0.4,0.6,1.0);
      emission[0] = 0.5; emission[1] = 0.3; emission[2] = 0.0;
      glMaterialfv(GL_FRONT, GL_EMISSION, emission);
      octahedron(x, z, -y, theta, s2);
   }
}

void Enemy::animate()
{
   theta += 2; fmod(theta, 360.0);
   if (type == 1)
   {
      if (s1 <= 0.7 || s1 >= 1.0)
         ds1 = -ds1;
      if (s2 <= 0.7 || s2 >= 1.0)
         ds2 = -ds2;
   }
   else
   {
      if (s1 <= 0.5 || s1 >= 1.1)
         ds1 = -ds1;
      if (s2 <= 0.5 || s2 >= 1.1)
         ds2 = -ds2;
   }
   s1 += ds1;
   s2 += ds2;

   //if (x
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

Bullet::Bullet(float ix, float iy, float iz, Enemy* Target)
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
