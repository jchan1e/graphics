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
   glNormal3d(0.10,1,0);
   glVertex3d(0,1.10,0);
   glVertex3d(1,1,1);
   glVertex3d(1,1,-1);

   glNormal3d(0,1,-0.10);
   glVertex3d(0,1.10,0);
   glVertex3d(1,1,-1);
   glVertex3d(-1,1,-1);

   glNormal3d(-0.10,1,0);
   glVertex3d(0,1.10,0);
   glVertex3d(-1,1,-1);
   glVertex3d(-1,1,1);

   glNormal3d(0,1,0.10);
   glVertex3d(0,1.10,0);
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
   speed = 0.05;
   movestate = 0;
   dx = speed; dy = 0;

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
   theta += 2; theta = fmod(theta, 360.0);
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

   x += dx; y += dy;

   switch (movestate)
   {
   case 0:
      if (x >= -2.0)
      {  movestate = 1; dx = 0; dy = -speed;}
      break;
   case 1:
      if (y <= 2.0)
      {  movestate = 2; dx = -speed; dy = 0;}
      break;
   case 2:
      if (x <= -6.0)
      {  movestate = 3; dx = 0; dy = -speed;}
      break;
   case 3:
      if (y <= -6.0)
      {  movestate = 4; dx = speed; dy = 0;}
      break;
   case 4:
      if (x >= -2.0)
      {  movestate = 5; dx = 0; dy = speed;}
      break;
   case 5:
      if (y >= -2.0)
      {  movestate = 6; dx = speed; dy = 0;}
      break;
   case 6:
      if (x >= 2.0)
      {  movestate = 7; dx = 0; dy = -speed;}
      break;
   case 7:
      if (y <= -6.0)
      {  movestate = 8; dx = speed; dy = 0;}
      break;
   case 8:
      if (x >= 6.0)
      {  movestate = 9; dx = 0; dy = speed;}
      break;
   case 9:
      if (y >= 2.0)
      {  movestate = 10; dx = -speed; dy = 0;}
      break;
   case 10:
      if (x <= 2.0)
      {  movestate = 11; dx = 0; dy = speed;}
      break;
   case 11:
      if (y >= 6.0)
      {  movestate = 12; dx = speed; dy = 0;}
      break;
   case 12:
      if (x >= 8.0)
      {  x = 8.0; y = 6.0; }
      break;
   default:
      movestate = 0;
      break;
   }
   //follow the metal grey road
   //if      (x < -2.0 && y >= 5.9)   //upper entry
   //{  dx = speed;    dy = 0;     }
   //else if (x >= -2.1 && y > 2.0)   //downward
   //{  dx = 0;        dy = -speed;}
   //else if (x > -6.0 && y <= 2.1)   //leftward
   //{  dx = -speed;   dy = 0;     }
   //else if (x <= -5.9 && y > -6.0)  //left corridor
   //{  dx = 0;        dy = -speed;}
   //else if (x < -2.0 && y <= -5.9)  //lower left horizontal
   //{  dx = speed;    dy = 0;     }
   //else if (x >= -2.1 && y < -2.0)  //left side upward
   //{  dx = 0;        dy = speed; }
   //else if (
}

void Enemy::damage(int dmg)
{
   health -= dmg;
}

Tower::Tower(float X, float Y)
{
   x = X;
   y = Y;
   z = 2.0;
   maxcooldown = 250;
   cooldown = 0;
   target = NULL;
}

void Tower::render()
{
}

Bullet* Tower::fire()
{
   Bullet* bullet = new Bullet(x, y, 3.0, target);
   return bullet;
}

float Tower::distance(Enemy** Target)
{
   return sqrt(((*Target)->x - x) * ((*Target)->x - x)
             + ((*Target)->y - y) * ((*Target)->y - y)
             + ((*Target)->z - z) * ((*Target)->z - z));
}

Bullet::Bullet(float X, float Y, float Z, Enemy** Target)
{
   x = X;
   y = Y;
   z = Z;
   target = Target;
   dmg = 10;
   speed = 0.125;
}

void Bullet::render()
{
   float emission[] = {0.0, 0.0, 0.0, 1.0};
   glColor3f(0.8,0.8,0.0);
   emission[0] = 0.7; emission[1] = 0.7; emission[2] = 0.0;
   glMaterialfv(GL_FRONT, GL_EMISSION, emission);
   ball(x, z, -y, 0.25);
}

void Bullet::animate()
{
   normalizeV();
   dx *= speed;
   dy *= speed;
   dz *= speed;
   x += dx;
   y += dy;
   z += dz;
}

void Bullet::collide(Enemy** target)
{
}

void Bullet::normalizeV()
{
   float a = distance();
   dx = ((*target)->x - x)/a;
   dy = ((*target)->y - y)/a;
   dz = ((*target)->z - z)/a;
}

float Bullet::distance()
{
   return sqrt(((*target)->x - x) * ((*target)->x - x)
             + ((*target)->y - y) * ((*target)->y - y)
             + ((*target)->z - z) * ((*target)->z - z));
}
