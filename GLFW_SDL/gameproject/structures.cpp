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

   if (lives > 0)
   {
      glColor3f(0.4,0.8,0.4);
      float emission[4] = {0.0,0.4,0.0,1.0};
      glMaterialfv(GL_FRONT, GL_EMISSION, emission);
      for (int i=0; i < lives; ++i)
      {
         if (i < 2)
            cube(8.5, 1.5*i, -6.0, 0, 1.0);
         else
            cube(10.0, 1.5*(i%2), -6.0 + 1.5*((i-2)/2), 0, 1.0);
      }
   }
   else
   {
      bool pixels[7][53] = {{0,1,1,1,0, 0, 0,1,1,1,0, 0, 0,1,0,1,0, 0, 1,1,1,1,1, 0, 0,0,0,0,0, 0, 0,1,1,1,0, 0, 1,0,0,0,1, 0, 1,1,1,1,1, 0, 1,1,1,1,0},
                            {1,0,0,0,1, 0, 1,0,0,0,1, 0, 1,0,1,0,1, 0, 1,0,0,0,0, 0, 0,0,0,0,0, 0, 1,0,0,0,1, 0, 1,0,0,0,1, 0, 1,0,0,0,0, 0, 1,0,0,0,1},
                            {1,0,0,0,0, 0, 1,0,0,0,1, 0, 1,0,1,0,1, 0, 1,0,0,0,0, 0, 0,0,0,0,0, 0, 1,0,0,0,1, 0, 1,0,0,0,1, 0, 1,0,0,0,0, 0, 1,0,0,0,1},
                            {1,0,1,1,1, 0, 1,1,1,1,1, 0, 1,0,1,0,1, 0, 1,1,1,1,0, 0, 0,0,0,0,0, 0, 1,0,0,0,1, 0, 1,0,0,0,1, 0, 1,1,1,1,0, 0, 1,1,1,1,0},
                            {1,0,0,0,1, 0, 1,0,0,0,1, 0, 1,0,1,0,1, 0, 1,0,0,0,0, 0, 0,0,0,0,0, 0, 1,0,0,0,1, 0, 1,0,0,0,1, 0, 1,0,0,0,0, 0, 1,0,1,0,0},
                            {1,0,0,0,1, 0, 1,0,0,0,1, 0, 1,0,1,0,1, 0, 1,0,0,0,0, 0, 0,0,0,0,0, 0, 1,0,0,0,1, 0, 0,1,0,1,0, 0, 1,0,0,0,0, 0, 1,0,0,1,0},
                            {1,1,1,1,1, 0, 1,0,0,0,1, 0, 1,0,1,0,1, 0, 1,1,1,1,1, 0, 0,0,0,0,0, 0, 0,1,1,1,0, 0, 0,0,1,0,0, 0, 1,1,1,1,1, 0, 1,0,0,0,1}};
      glColor3f(0.9,0.9,0.9);
      glPushMatrix();
      for (float th=0.0; th <= 271.0; th += 90)
      {
         glRotated(th, 0,1,0);
         for (int i=0; i<53; ++i)
         {
            for (int j=0; j<7;++j)
            {
               if (pixels[j][i])
                  cube(2.5-0.1*i,4-0.1*j,9.0, 0, 0.1/sqrt(2.0));
            }
         }
      }
      glPopMatrix();
   }
}

int Floor::animate()
{
   if (spawncount > 0)
   {
      currentwavetime += 16;
      if (currentwavetime >= wavetime)
      {
         currentwavetime -= wavetime;
         spawncount -= 1;
         return waves[currentwave][9-spawncount];
      }
   }
   return 0;
}

void Floor::spawnwave()
{
   if (spawncount <= 0)
   {
      if (currentwave < 8)
         currentwave += 1;
      spawncount = 10;
   }
}

int Floor::getlocation(float ex, float ey)
{
   int ix = (int)ex/2 + 4;
   int iy = -(int)ey/2 + 4;
   if (iy > 8 || iy < 0 || ix > 8 || ix < 0)
      return -1;
   return arr[9*iy + ix];
}

Enemy::Enemy(float X, float Y, int Health, int Type)
{
   type = Type;
   health = Health;
   x = X;
   y = Y;
   z = 0;
   theta = 0.0;
   speed = 0.06;
   if (type == 1)
   {
      speed = 0.08;
      health /= 2;
   }
   else if (type == 3)
   {
      speed = 0.04;
      health *= 2;
   }
   movestate = 0;
   dx = speed; dy = 0;

   if (type == 2)
   {
      s1 = 0.85;  ds1 = 0.02;
      s2 = 0.85;  ds2 = -0.02;
   }
   else if (type == 1)
   {
      s1 = 0.8;  ds1 = 0.03;
      s2 = 0.8;  ds2 = -0.03;
   }
   else //type == 3
   {
      s1 = 0.0;   ds1 = 5.12;
      s2 = 0.0;   ds2 = 3.0;
   }
}

void Enemy::render()
{
   float emission[] = {0.0,0.0,0.0,1.0};

   //glColor3f(1.0,1.0,1.0);
   if (type == 2)
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
   else if (type == 1)
   {
      glColor3f(0.8,0.0,0.0);
      //glColor3f(0.0,0.8,0.8);
      emission[0] = 0.5; emission[1] = 0.0; emission[2] = 0.0;
      glMaterialfv(GL_FRONT, GL_EMISSION, emission);
      cube(x, z, -y, theta, s1*0.8);

      glColor3f(0.6,0.4,0.0);
      //glColor3f(0.4,0.6,1.0);
      emission[0] = 0.5; emission[1] = 0.3; emission[2] = 0.0;
      glMaterialfv(GL_FRONT, GL_EMISSION, emission);
      octahedron(x, z, -y, theta, s2);
   }
   else // type == 3
   {
      glColor3f(0.0,0.8,0.8);
      emission[0] = 0.0;   emission[1] = 0.4;emission[2] = 0.25;
      glMaterialfv(GL_FRONT, GL_EMISSION, emission);

      sphere(x, z, -y, 0, 0.25);

      glPushMatrix();
      glTranslated(x, z, -y);
      glRotated(theta, 0,1,0);
      glRotated(s1, 1,0,0);
      torus (0, 0, 0, 0.100, 0.4);
      glPopMatrix();

      glPushMatrix();
      glTranslated(x, z, -y);
      glRotated(theta/2, 0,-1,0);
      glRotated(s2, 1,0,0);
      torus (0, 0, 0, 0.100, 0.65);
      glPopMatrix();
   }
}

void Enemy::animate()
{
   if (type == 2)
   {
      if (s1 <= 0.7 || s1 >= 1.0)
         ds1 = -ds1;
      if (s2 <= 0.7 || s2 >= 1.0)
         ds2 = -ds2;
      theta += 2; theta = fmod(theta, 360.0);
   }
   else if (type == 1)
   {
      if (s1 <= 0.5 || s1 >= 1.1)
         ds1 = -ds1;
      if (s2 <= 0.5 || s2 >= 1.1)
         ds2 = -ds2;
      theta += 2; theta = fmod(theta, 360.0);
   }
   else //type == 3
   {
      theta += 2; theta = fmod(theta, 720.0);
      s1 = fmod(s1, 360.0);
      s2 = fmod(s2, 360.0);
   }
   s1 += ds1;
   s2 += ds2;

   x += dx; y += dy;
   
   //Follow the metal grey road
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
}

void Enemy::damage(int dmg)
{
   health -= dmg;
}

Tower::Tower(float X, float Y, bool mode)
{
   x = X;
   y = Y;
   z = 3.0;
   dx = 1.0;
   dy = 0.0;
   dz = 0.0;
   maxcooldown = 500;
   cooldown = 0;
   target = NULL;
   range = 8.0;
   wireframe = mode;
}

void Tower::animate()
{
   if (!wireframe && target != NULL)
   {
      dx = x - (*target)->x;
      dy = y - (*target)->y;
      dz = z - (*target)->z;
   }
}

void Tower::render()
{
   if (!wireframe)
   {
      float emission[] = {0.0,0.0,0.0,1.0};
      glMaterialfv(GL_FRONT, GL_EMISSION, emission);
      glColor3f(0.25,0.25,0.3);
      octahedron(x,1   ,-y, 0, 1);
      octahedron(x,1.5 ,-y, 0, 0.75);
      octahedron(x,2.0 ,-y, 0, 0.625);
      octahedron(x,2.5 ,-y, 0, 0.5);

      float W0 = sqrt(dx*dx + dy*dy + dz*dz);
      float X0 = dx/W0;
      float Y0 = dy/W0;
      float Z0 = dz/W0;

      float W2 = sqrt(Y0*Y0 + X0*X0);
      float X2 = Y0/W2;
      float Y2 = -X0/W2;
      float Z2 = 0;

      //float W1;
      float X1 = Y2*Z0 - Y0*Z2;
      float Y1 = Z2*X0 - Z0*X2;
      float Z1 = X2*Y0 - X0*Y2;

      float mat[16];
      mat[0] = X0;   mat[4] = X1;    mat[8] = X2;   mat[12] = 0;
      mat[1] = Z0;   mat[5] = Z1;    mat[9] = Z2;   mat[13] = 0;
      mat[2] =-Y0;   mat[6] =-Y1;    mat[10]=-Y2;   mat[14] = 0;
      mat[3] = 0;    mat[7] = 0;     mat[11] = 0;   mat[15] = 1;

      glPushMatrix();
      glTranslated(x,3.0,-y);
      glMultMatrixf(mat);
      glScaled(1/3.0, 1/3.0, 1/3.0);

      emission[0] = 0.5; emission[1] = 0.5; emission[2] = 0.55;
      glMaterialfv(GL_FRONT, GL_EMISSION, emission);
      glColor3f(0.6,0.6,0.6);
      //octahedron(x,3.0 ,-y, 0, 0.4);

      glBegin(GL_TRIANGLES);
      glNormal3f(1.0, 1.0, 1.0);
      glVertex3f(1,0,0);
      glVertex3f(0,1,0);
      glVertex3f(0,0,1);

      glNormal3f(-1.0/2, 1.0, 1.0);
      glVertex3f(0,1,0);
      glVertex3f(-2,0,0);
      glVertex3f(0,0,1);

      glNormal3f(-1.0/2, -1.0, 1.0);
      glVertex3f(-2,0,0);
      glVertex3f(0,-1,0);
      glVertex3f(0,0,1);

      glNormal3f(1.0, -1.0, 1.0);
      glVertex3f(0,-1,0);
      glVertex3f(1,0,0);
      glVertex3f(0,0,1);

      glNormal3f(1.0, 1.0, -1.0);
      glVertex3f(0,1,0);
      glVertex3f(1,0,0);
      glVertex3f(0,0,-1);

      glNormal3f(-1.0/2, 1.0, -1.0);
      glVertex3f(-2,0,0);
      glVertex3f(0,1,0);
      glVertex3f(0,0,-1);

      glNormal3f(-1.0/2, -1.0, -1.0);
      glVertex3f(0,-1,0);
      glVertex3f(-2,0,0);
      glVertex3f(0,0,-1);

      glNormal3f(1.0, -1.0, -1.0);
      glVertex3f(1,0,0);
      glVertex3f(0,-1,0);
      glVertex3f(0,0,-1);
      glEnd();

      glPopMatrix();
   }
   else
   {
      float emission[] = {0.0,0.0,0.0,1.0};
      glMaterialfv(GL_FRONT, GL_EMISSION, emission);
      glColor3f(0.25,0.65,0.3);
      octahedron(x,1,-y, 0, 1.05);
   }
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
             + ((*Target)->z - 0) * ((*Target)->z - 0));
}

Bullet::Bullet(float X, float Y, float Z, Enemy** Target)
{
   x = X;
   y = Y;
   z = Z;
   target = Target;
   dmg = 10;
   speed = 0.25;
}

void Bullet::render()
{
   float W0 = sqrt(dx*dx + dy*dy + dz*dz);
   float X0 = dx/W0;
   float Y0 = dy/W0;
   float Z0 = dz/W0;

   float W2 = sqrt(Y0*Y0 + X0*X0);
   float X2 = Y0/W2;
   float Y2 = -X0/W2;
   float Z2 = 0;

   //float W1;
   float X1 = Y2*Z0 - Y0*Z2;
   float Y1 = Z2*X0 - Z0*X2;
   float Z1 = X2*Y0 - X0*Y2;

   float mat[16];
   mat[0] = X0;   mat[4] = X1;    mat[8] = X2;   mat[12] = 0;
   mat[1] = Z0;   mat[5] = Z1;    mat[9] = Z2;   mat[13] = 0;
   mat[2] =-Y0;   mat[6] =-Y1;    mat[10]=-Y2;   mat[14] = 0;
   mat[3] = 0;    mat[7] = 0;     mat[11] = 0;   mat[15] = 1;

   float emission[] = {0.0, 0.0, 0.0, 1.0};
   glColor3f(0.9,0.9,0.3);
   emission[0] = 0.6; emission[1] = 0.6; emission[2] = 0.0;
   glMaterialfv(GL_FRONT, GL_EMISSION, emission);
   //ball(x, z, -y, 0.125);
   
   glPushMatrix();
   glTranslatef(x, z, -y);
   glMultMatrixf(mat);
   glScaled(0.25,0.25,0.25);

   glBegin(GL_TRIANGLES);
   glNormal3f(1.0/2, 1.0, 1.0);
   glVertex3f(2,0,0);
   glVertex3f(0,1,0);
   glVertex3f(0,0,1);

   glNormal3f(-1.0/2, 1.0, 1.0);
   glVertex3f(0,1,0);
   glVertex3f(-2,0,0);
   glVertex3f(0,0,1);

   glNormal3f(-1.0/2, -1.0, 1.0);
   glVertex3f(-2,0,0);
   glVertex3f(0,-1,0);
   glVertex3f(0,0,1);

   glNormal3f(1.0/2, -1.0, 1.0);
   glVertex3f(0,-1,0);
   glVertex3f(2,0,0);
   glVertex3f(0,0,1);

   glNormal3f(1.0/2, 1.0, -1.0);
   glVertex3f(0,1,0);
   glVertex3f(2,0,0);
   glVertex3f(0,0,-1);

   glNormal3f(-1.0/2, 1.0, -1.0);
   glVertex3f(-2,0,0);
   glVertex3f(0,1,0);
   glVertex3f(0,0,-1);

   glNormal3f(-1.0/2, -1.0, -1.0);
   glVertex3f(0,-1,0);
   glVertex3f(-2,0,0);
   glVertex3f(0,0,-1);

   glNormal3f(1.0/2, -1.0, -1.0);
   glVertex3f(2,0,0);
   glVertex3f(0,-1,0);
   glVertex3f(0,0,-1);
   glEnd();

   glPopMatrix();
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

void Bullet::collide()
{
   (*target)->damage(dmg);
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
