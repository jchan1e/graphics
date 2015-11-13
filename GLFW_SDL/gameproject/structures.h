#include "objects.h"

class Floor
{
   float startx;
   float starty;
   int arr[81] = {0, 0, 0, 0, 0, 0, 0, 0, 0,  // it's a 2nd order Hilbert curve
                 -1,-1,-1,-1, 0,-1,-1,-1,-1,
                  0, 0, 0,-1, 0,-1, 0, 0, 0,
                  0,-1,-1,-1, 0,-1,-1,-1, 0,
                  0,-1, 0, 0, 0, 0, 0,-1, 0,
                  0,-1, 0,-1,-1,-1, 0,-1, 0,
                  0,-1, 0,-1, 0,-1, 0,-1, 0,
                  0,-1,-1,-1, 0,-1,-1,-1, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0};
public:
   Floor();
   void tile(float x, float y, float z, int direction);
   void render();
   float Startx() {return startx;}
   float Starty() {return starty;}
};

class Enemy
{
public:
   float x, y, z, dx, dy, theta, speed;
   int health, type;
   float s1, s2, ds1, ds2;
   int movestate;

   Enemy(float X, float Y, int Health, int Type);
   void render();
   void animate();
   void damage(int dmg);
};


class Bullet
{
public:
   float x, y, z, dx, dy, dz, speed;
   int dmg;
   Enemy* target;

   Bullet(float X, float Y, float Z, Enemy* Target);
   void render();
   void animate();
   void collide(Enemy* target);
   float distance();
   void normalizeV();
};

class Tower
{
public:
   float x, y, z;
   Enemy* target;
   int cooldown, maxcooldown;

   Tower(float X, float Y);
   void render();
   Bullet* fire();
   float distance(Enemy* Target);
};
