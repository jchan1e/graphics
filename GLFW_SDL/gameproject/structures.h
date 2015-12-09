#include "objects.h"

class Floor
{
   int arr[81] = {0, 0, 0, 0, 0, 0, 0, 0, 0,  // it's a 2nd order Hilbert curve
                 -1,-1,-1,-1, 0,-1,-1,-1,-1,
                  0, 0, 0,-1, 0,-1, 0, 0, 0,
                  0,-1,-1,-1, 0,-1,-1,-1, 0,
                  0,-1, 0, 0, 0, 0, 0,-1, 0,
                  0,-1, 0,-1,-1,-1, 0,-1, 0,
                  0,-1, 0,-1, 0,-1, 0,-1, 0,
                  0,-1,-1,-1, 0,-1,-1,-1, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0};
   int waves[9][10] = {{2, 2, 2, 2, 2, 0, 0, 0, 0, 0},
                       {2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                       {1, 1, 1, 1, 1, 1, 1, 1, 0, 0},
                       {2, 1, 2, 1, 2, 1, 2, 1, 2, 1},
                       {2, 2, 2, 2, 2, 1, 1, 1, 1, 1},
                       {3, 0, 3, 0, 3, 0, 3, 0, 3, 0},
                       {3, 2, 1, 3, 2, 1, 3, 2, 1, 3},
                       {3, 3, 3, 3, 2, 2, 2, 1, 1, 1},
                       {3, 3, 3, 3, 3, 3, 3, 3, 3, 3}};
   int wavetime = 600;
   int currentwavetime = 0;
   int spawncount = -1;
public:
   int lives = 20;
   int currentwave = -1;
   Floor();
   void tile(float x, float y, float z, int direction);
   void render();
   int animate();
   void spawnwave();
   int getlocation(float ex, float ey);
};

class Enemy
{
public:
   float x, y, z, dx, dy, theta, speed;
   //float a, b, c, w;
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
   Enemy** target;

   Bullet(float X, float Y, float Z, Enemy** Target);
   void render();
   void animate();
   void collide();
   float distance();
   void normalizeV();
};

class Tower
{
public:
   float x, y, z, dx, dy, dz, range;
   Enemy** target;
   int cooldown, maxcooldown;
   bool wireframe;

   Tower(float X, float Y, bool mode);
   void animate();
   void render();
   Bullet* fire();
   float distance(Enemy** Target);
};
