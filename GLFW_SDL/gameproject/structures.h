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

   Enemy(float X, float Y, int Health, int Type);
   void render();
   void animate();
   void damage(int dmg);
};

class Tower
{
public:
   float x, y;
   Enemy* target;
   float cooldown, maxcooldown;

   Tower(float x, float y);
   void render();
   void animate();
   void fire();
};

class Bullet
{
public:
   float x, y, z, dx, dy, dz;
   Enemy* target;

   Bullet(float ix, float iy, float iz, Enemy* Target);
   void render();
   void animate();
   void collide(Enemy* target);
};
