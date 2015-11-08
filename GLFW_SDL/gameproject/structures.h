#include "objects.h"

class Floor
{
   int arr[32];
   float startx;
   float starty;
   void tile(float x, float y, int direction);
public:
   Floor();
   void render();
   float Startx() {return startx;}
   float Starty() {return starty;}
};

class Enemy
{
public:
   float x, y, dx, dy, speed;
   int health;
   float s1, s2;

   Enemy(float x, float y);
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

   Bullet(float x, float y, float z);
   void render();
   void animate();
   void collide(Enemy* target);
};
