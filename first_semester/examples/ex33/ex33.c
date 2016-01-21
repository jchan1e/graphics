/*
 *  Utah Teapot
 *
 *  Demonstrates drawing an objects using bezier patches.
 *  Note:  This is not done efficiently, the
 *    idea is to make it understandable
 *
 *  Key bindings:
 *  m/M        Cycle through display modes
 *  o/O        Cycle through parts
 *  c/C        Toggle control points
 *  r/R        Toggle reflections
 *  s/S        Toggle lid size
 *  +/-        Increase/decrease number of slices
 *  []         Decrease/increase light elevation
 *  a          Toggle axes
 *  arrows     Change view angle
 *  PgDn/PgUp  Zoom in and out
 *  0          Reset view angle
 *  ESC        Exit
 */
#include "CSCIx229.h"
#define OBJ 10
#define MODE 9
int mode=0;    //  Display mode
int axes=1;    //  Display axes
int obj=OBJ;   //  Display objects
int ref=1;     //  Display reflect
int ctrl=0;    //  Display control points
int n = 8;     //  Number of slices
int th=-30;    //  Azimuth of view angle
int ph=+30;    //  Elevation of view angle
int zh=-30;    //  Light azimuth
int size=0;    //  Lid size
double asp=1;  //  Aspect ratio
double dim=3;  //  Size of world
typedef struct {float x,y,z;} Point;
char* part[] = {"Rim","Upper Body","Lower Body","Bottom","Lid Handle","Lid","Upper Handle","Lower Handle","Spout Body","Spout Tip","Teapot"};
char* text[] = {"Wireframe","Hidden Line Wireframe","Solid","Solid Lit","Textured","Textured Lit","Manual Solid","Manual with Normals","Manual with Textures"};

#define MAXN 64

//  Rim, body, lid, and bottom data must be reflected in x and y;
//  Handle and spout data reflected across the y axis only.
static int teapot[][4][4] =
{

   {{102,103,104,105},{  4,  5,  6,  7},{  8,  9, 10, 11},{ 12, 13, 14, 15}},  //  Rim
   {{ 12, 13, 14, 15},{ 16, 17, 18, 19},{ 20, 21, 22, 23},{ 24, 25, 26, 27}},  //  Upper body
   {{ 24, 25, 26, 27},{ 29, 30, 31, 32},{ 33, 34, 35, 36},{ 37, 38, 39, 40}},  //  Lower body
   {{118,118,118,118},{124,122,119,121},{123,126,125,120},{ 40, 39, 38, 37}},  //  Bottom
   {{ 96, 96, 96, 96},{ 97, 98, 99,100},{101,101,101,101},{  0,  1,  2,  3}},  //  Lid handle
   {{  0,  1,  2,  3},{106,107,108,109},{110,111,112,113},{114,115,116,117}},  //  Lid
   {{ 41, 42, 43, 44},{ 45, 46, 47, 48},{ 49, 50, 51, 52},{ 53, 54, 55, 56}},  //  Upper handle
   {{ 53, 54, 55, 56},{ 57, 58, 59, 60},{ 61, 62, 63, 64},{ 28, 65, 66, 67}},  //  Lower handle
   {{ 68, 69, 70, 71},{ 72, 73, 74, 75},{ 76, 77, 78, 79},{ 80, 81, 82, 83}},  //  Spout body
   {{ 80, 81, 82, 83},{ 84, 85, 86, 87},{ 88, 89, 90, 91},{ 92, 93, 94, 95}},  //  Spout tip
};

//  Data points
static Point data[] =
{
   { 0.2   ,  0     , 2.7    },
   { 0.2   , -0.112 , 2.7    },
   { 0.112 , -0.2   , 2.7    },
   { 0     , -0.2   , 2.7    },
   { 1.3375,  0     , 2.53125},
   { 1.3375, -0.749 , 2.53125},
   { 0.749 , -1.3375, 2.53125},
   { 0     , -1.3375, 2.53125},
   { 1.4375,  0     , 2.53125},
   { 1.4375, -0.805 , 2.53125},
   { 0.805 , -1.4375, 2.53125},
   { 0     , -1.4375, 2.53125},
   { 1.5   ,  0     , 2.4    },
   { 1.5   , -0.84  , 2.4    },
   { 0.84  , -1.5   , 2.4    },
   { 0     , -1.5   , 2.4    },
   { 1.75  ,  0     , 1.875  },
   { 1.75  , -0.98  , 1.875  },
   { 0.98  , -1.75  , 1.875  },
   { 0     , -1.75  , 1.875  },
   { 2     ,  0     , 1.35   },
   { 2     , -1.12  , 1.35   },
   { 1.12  , -2     , 1.35   },
   { 0     , -2     , 1.35   },
   { 2     ,  0     , 0.9    },
   { 2     , -1.12  , 0.9    },
   { 1.12  , -2     , 0.9    },
   { 0     , -2     , 0.9    },
   { -2    ,  0     , 0.9    },
   { 2     ,  0     , 0.45   },
   { 2     , -1.12  , 0.45   },
   { 1.12  , -2     , 0.45   },
   { 0     , -2     , 0.45   },
   { 1.5   ,  0     , 0.225  },
   { 1.5   , -0.84  , 0.225  },
   { 0.84  , -1.5   , 0.225  },
   { 0     , -1.5   , 0.225  },
   { 1.5   ,  0     , 0.15   },
   { 1.5   , -0.84  , 0.15   },
   { 0.84  , -1.5   , 0.15   },
   { 0     , -1.5   , 0.15   },
   {-1.6   ,  0     , 2.025  },
   {-1.6   , -0.3   , 2.025  },
   {-1.5   , -0.3   , 2.25   },
   {-1.5   ,  0     , 2.25   },
   {-2.3   ,  0     , 2.025  },
   {-2.3   , -0.3   , 2.025  },
   {-2.5   , -0.3   , 2.25   },
   {-2.5   ,  0     , 2.25   },
   {-2.7   ,  0     , 2.025  },
   {-2.7   , -0.3   , 2.025  },
   {-3     , -0.3   , 2.25   },
   {-3     ,  0     , 2.25   },
   {-2.7   ,  0     , 1.8    },
   {-2.7   , -0.3   , 1.8    },
   {-3     , -0.3   , 1.8    },
   {-3     ,  0     , 1.8    },
   {-2.7   ,  0     , 1.575  },
   {-2.7   , -0.3   , 1.575  },
   {-3     , -0.3   , 1.35   },
   {-3     ,  0     , 1.35   },
   {-2.5   ,  0     , 1.125  },
   {-2.5   , -0.3   , 1.125  },
   {-2.65  , -0.3   , 0.9375 },
   {-2.65  ,  0     , 0.9375 },
   {-2     , -0.3   , 0.9    },
   {-1.9   , -0.3   , 0.6    },
   {-1.9   ,  0     , 0.6    },
   { 1.7   ,  0     , 1.425  },
   { 1.7   , -0.66  , 1.425  },
   { 1.7   , -0.66  , 0.6    },
   { 1.7   ,  0     , 0.6    },
   { 2.6   ,  0     , 1.425  },
   { 2.6   , -0.66  , 1.425  },
   { 3.1   , -0.66  , 0.825  },
   { 3.1   ,  0     , 0.825  },
   { 2.3   ,  0     , 2.1    },
   { 2.3   , -0.25  , 2.1    },
   { 2.4   , -0.25  , 2.025  },
   { 2.4   ,  0     , 2.025  },
   { 2.7   ,  0     , 2.4    },
   { 2.7   , -0.25  , 2.4    },
   { 3.3   , -0.25  , 2.4    },
   { 3.3   ,  0     , 2.4    },
   { 2.8   ,  0     , 2.475  },
   { 2.8   , -0.25  , 2.475  },
   { 3.525 , -0.25  , 2.49375},
   { 3.525 ,  0     , 2.49375},
   { 2.9   ,  0     , 2.475  },
   { 2.9   , -0.15  , 2.475  },
   { 3.45  , -0.15  , 2.5125 },
   { 3.45  ,  0     , 2.5125 },
   { 2.8   ,  0     , 2.4    },
   { 2.8   , -0.15  , 2.4    },
   { 3.2   , -0.15  , 2.4    },
   { 3.2   ,  0     , 2.4    },
   { 0     ,  0     , 3.15   },
   { 0.8   ,  0     , 3.15   },
   { 0.8   , -0.45  , 3.15   },
   { 0.45  , -0.8   , 3.15   },
   { 0     , -0.8   , 3.15   },
   { 0     ,  0     , 2.85   },
   { 1.4   ,  0     , 2.4    },
   { 1.4   , -0.784 , 2.4    },
   { 0.784 , -1.4   , 2.4    },
   { 0     , -1.4   , 2.4    },
   { 0.4   ,  0     , 2.55   },
   { 0.4   , -0.224 , 2.55   },
   { 0.224 , -0.4   , 2.55   },
   { 0     , -0.4   , 2.55   },
   { 1.3   ,  0     , 2.55   },
   { 1.3   , -0.728 , 2.55   },
   { 0.728 , -1.3   , 2.55   },
   { 0     , -1.3   , 2.55   },
   { 1.3   ,  0     , 2.4    },
   { 1.3   , -0.728 , 2.4    },
   { 0.728 , -1.3   , 2.4    },
   { 0     , -1.3   , 2.4    },
   { 0     ,  0     , 0      },
   { 1.425 , -0.798 , 0      },
   { 1.5   ,  0     , 0.075  },
   { 1.425 ,  0     , 0      },
   { 0.798 , -1.425 , 0      },
   { 0     , -1.5   , 0.075  },
   { 0     , -1.425 , 0      },
   { 1.5   , -0.84  , 0.075  },
   { 0.84  , -1.5   , 0.075  },
};

//  Texture coordinates
static struct {float r,s;} tex[2][2] =
{
   {{1,1},{0,1}},
   {{1,0},{0,0}}
};

#define Bezier(x)  V*V*V*(U*U*U*p[0][0].x + 3*U*U*u*p[0][1].x + 3*U*u*u*p[0][2].x + u*u*u*p[0][3].x) \
               + 3*V*V*v*(U*U*U*p[1][0].x + 3*U*U*u*p[1][1].x + 3*U*u*u*p[1][2].x + u*u*u*p[1][3].x) \
               + 3*V*v*v*(U*U*U*p[2][0].x + 3*U*U*u*p[2][1].x + 3*U*u*u*p[2][2].x + u*u*u*p[2][3].x) \
               +   v*v*v*(U*U*U*p[3][0].x + 3*U*U*u*p[3][1].x + 3*U*u*u*p[3][2].x + u*u*u*p[3][3].x)
/*
 *  Evaluate 2D Bezier surface
 */
Point Bezier2D(Point p[4][4],float u,float v)
{
   float U = 1-u;
   float V = 1-v;
   Point P;
   P.x = Bezier(x);
   P.y = Bezier(y);
   P.z = Bezier(z);
   return P;
}

#define ddu(x)  -U*U*(V*V*V*p[0][0].x + 3*V*V*v*p[1][0].x + 3*V*v*v*p[2][0].x + v*v*v*p[3][0].x) \
         + (1-3*u)*U*(V*V*V*p[0][1].x + 3*V*V*v*p[1][1].x + 3*V*v*v*p[2][1].x + v*v*v*p[3][1].x) \
         + u*(2-3*u)*(V*V*V*p[0][2].x + 3*V*V*v*p[1][2].x + 3*V*v*v*p[2][2].x + v*v*v*p[3][2].x) \
         +       u*u*(V*V*V*p[0][3].x + 3*V*V*v*p[1][3].x + 3*V*v*v*p[2][3].x + v*v*v*p[3][3].x)
#define ddv(x)  -V*V*(U*U*U*p[0][0].x + 3*U*U*u*p[0][1].x + 3*U*u*u*p[0][2].x + u*u*u*p[0][3].x) \
         + (1-3*v)*V*(U*U*U*p[1][0].x + 3*U*U*u*p[1][1].x + 3*U*u*u*p[1][2].x + u*u*u*p[1][3].x) \
         + v*(2-3*v)*(U*U*U*p[2][0].x + 3*U*U*u*p[2][1].x + 3*U*u*u*p[2][2].x + u*u*u*p[2][3].x) \
         +       v*v*(U*U*U*p[3][0].x + 3*U*U*u*p[3][1].x + 3*U*u*u*p[3][2].x + u*u*u*p[3][3].x)

/*
 *  Evaluate 2D Bezier normal
 */
Point Normal2D(Point p[4][4],float u,float v)
{
   float tiny=1e-6;
   float U = 1-u;
   float V = 1-v;
   float D,Du,Dv;
   Point P,Pu,Pv;
   //  1/3 of derivative in the u direction
   Pu.x = ddu(x);
   Pu.y = ddu(y);
   Pu.z = ddu(z);
   Du = sqrt(Pu.x*Pu.x+Pu.y*Pu.y+Pu.z*Pu.z);
   //  1/3 of derivative in the v direction
   Pv.x = ddv(x);
   Pv.y = ddv(y);
   Pv.z = ddv(z);
   Dv = sqrt(Pv.x*Pv.x+Pv.y*Pv.y+Pv.z*Pv.z);
   //  Du=0
   if (Du<tiny && Dv>tiny)
   {
      u += 0.001;
      U -= 0.001;
      Pu.x = ddv(x);
      Pu.y = ddv(y);
      Pu.z = ddv(z);
   }
   //  Dv=0
   else if (Dv<tiny && Du>tiny)
   {
      v += 0.001;
      V -= 0.001;
      Pv.x = ddu(x);
      Pv.y = ddu(y);
      Pv.z = ddu(z);
   }
   //  Cross product
   P.x = Pu.y*Pv.z - Pu.z*Pv.y;
   P.y = Pu.z*Pv.x - Pu.x*Pv.z;
   P.z = Pu.x*Pv.y - Pu.y*Pv.x;
   //  Normalize
   D = sqrt(P.x*P.x+P.y*P.y+P.z*P.z);
   if (D>tiny)
   {
      P.x /= D;
      P.y /= D;
      P.z /= D;
   }
   return P;
}

/*
 *  Draw patch
 */
static void Patch(int patch[4][4],float Sx,float Sy , int type,int mesh,int norm , int R,int G,int B)
{
   int i,j,k;
   Point p[4][4],P[MAXN+1][MAXN+1],N[MAXN+1][MAXN+1],T[MAXN+1][MAXN+1];

   //  Copy data with reflection
   for (k=0;k<4;k++)
   {
      int K = Sx*Sy<0 ? 3-k : k;
      for (j=0;j<4;j++)
      {
         int l = patch[j][K];
         p[j][k].x = Sx*data[l].x;
         p[j][k].y = Sy*data[l].y;
         p[j][k].z =    data[l].z;
      }
   }

   //  Set color
   glColor3f(R,G,B);

   //  Evaluate using mesh
   if (mesh)
   {
      glMap2f(GL_MAP2_VERTEX_3,0,1,3,4, 0,1,12,4,(void*)p);
      glEvalMesh2(type,0,n,0,n);
   }

   //  Manually evaluate mesh
   else
   {
      //  Evaluate grid points
      for (i=0;i<=n;i++)
         for (j=0;j<=n;j++)
         {
            float u = (float)i/n;
            float v = (float)j/n;
            P[i][j] = Bezier2D(p,u,v);
            N[i][j] = Normal2D(p,u,v);
            T[i][j].x = 1-u;
            T[i][j].y = 1-v;
            T[i][j].z = 0;
         }

      //  Draw quads
      for (i=0;i<n;i++)
      {
         glBegin(GL_QUAD_STRIP);
         for (j=0;j<=n;j++)
         {
            //  Draw normals and vertexes
            glNormal3fv((void*)&N[i  ][j]); glTexCoord2fv((void*)&T[i  ][j]); glVertex3fv((void*)&P[i  ][j]);
            glNormal3fv((void*)&N[i+1][j]); glTexCoord2fv((void*)&T[i+1][j]); glVertex3fv((void*)&P[i+1][j]);
         }
         glEnd();
      }

      //  Show Normals
      if (norm)
      {
         glColor3f(1,0,0);
         glBegin(GL_LINES);
         for (i=0;i<=n;i++)
            for (j=0;j<=n;j++)
            {
               glVertex3f(P[i][j].x,P[i][j].y,P[i][j].z);
               glVertex3f(P[i][j].x+N[i][j].x/8,P[i][j].y+N[i][j].y/8,P[i][j].z+N[i][j].z/8);
            }
         glEnd();
      }
   }

   //  Draw Control Points
   if (ctrl)
   {
      glColor3f(1,0,0);
      glPointSize(5);
      glBegin(GL_POINTS);
      for (k=0;k<4;k++)
         for (j=0;j<4;j++)
            glVertex3f(p[j][k].x,p[j][k].y,p[j][k].z);
      glEnd();
   }
}

/*
 *  Draw teapot
 */
static void Teapot(int type,int mesh,int norm , float R,float G,float B)
{
   int i;
   int m0=(obj<OBJ)?obj:0;
   int m1=(obj<OBJ)?obj+1:OBJ;

   //  Set transform
   glPushMatrix();
   glRotated(-90,1,0,0);
   glTranslatef(0.0, 0.0, -1.5);

   if (mesh)
   {
      //  Enable 3D vertexes and 2D textures
      glEnable(GL_MAP2_VERTEX_3);
      glEnable(GL_MAP2_TEXTURE_COORD_2);
      //  Evaluate on n x n grid
      glMapGrid2f(n,0,1,n,0,1);
      //  Texture coordinates
      glMap2f(GL_MAP2_TEXTURE_COORD_2, 0,1,2,2, 0,1,4,2,(void*)tex);
   }
   else
   {
      //  Enable 3D vertexes and 2D textures
      glDisable(GL_MAP2_VERTEX_3);
      glDisable(GL_MAP2_TEXTURE_COORD_2);
   }

   //  Draw parts of teapot
   for (i=m0;i<m1;i++)
   {
      //  Draw just one patch
      if (!ref)
         Patch(teapot[i],+1,+1 , type,mesh,norm , R,G,B);
      //  Draw patches reflected to 4 quadrants
      else if (i<6)
      {
         Patch(teapot[i],+1,+1 , type,mesh,norm , R,G,B);
         Patch(teapot[i],+1,-1 , type,mesh,norm , R,G,B);
         Patch(teapot[i],-1,+1 , type,mesh,norm , R,G,B);
         Patch(teapot[i],-1,-1 , type,mesh,norm , R,G,B);
      }
      //  Draw patch reflected to 2 hemispheres
      else
      {
         Patch(teapot[i],+1,+1 , type,mesh,norm , R,G,B);
         Patch(teapot[i],+1,-1 , type,mesh,norm , R,G,B);
      }
   }

   //  Undo transformations
   glPopMatrix();
}

/*
 *  Enable lighting
 */
static void Light(int on)
{
   if (on)
   {
      float Ambient[]   = {0.3,0.3,0.3,1.0};
      float Diffuse[]   = {1,1,1,1};
      //  Light direction
      float Position[]  = {Sin(zh),0.33,Cos(zh),0};
      //  Enable lighting with normalization
      glEnable(GL_LIGHTING);
      glEnable(GL_NORMALIZE);
      //  glColor sets ambient and diffuse color materials
      glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE);
      glEnable(GL_COLOR_MATERIAL);
      //  Enable light 0
      glEnable(GL_LIGHT0);
      glLightfv(GL_LIGHT0,GL_AMBIENT ,Ambient);
      glLightfv(GL_LIGHT0,GL_DIFFUSE ,Diffuse);
      glLightfv(GL_LIGHT0,GL_POSITION,Position);
   }
   else
      glDisable(GL_LIGHTING);
}

/*
 *  OpenGL (GLUT) calls this routine to display the scene
 */
void display()
{
   //  Length of axes
   const double len=2.0;
   //  Eye position
   double Ex = -2*dim*Sin(th)*Cos(ph);
   double Ey = +2*dim        *Sin(ph);
   double Ez = +2*dim*Cos(th)*Cos(ph);

   //  Erase the window and the depth buffer
   glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
   //  Enable Z-buffering in OpenGL
   glEnable(GL_DEPTH_TEST);

   //  Set perspective
   glLoadIdentity();
   gluLookAt(Ex,Ey,Ez , 0,0,0 , 0,Cos(ph),0);

   //  OpenGL should generate normals
   glEnable(GL_AUTO_NORMAL);
   glEnable(GL_NORMALIZE);

   //  Draw teapot
   switch (mode)
   {
      //  Wireframe
      case 0:
         Teapot(GL_LINE,1,0 , 1,1,0);
         break;
      //  Wireframe with hidden line removal
      case 1:
         glEnable(GL_POLYGON_OFFSET_FILL);
         glPolygonOffset(1,1);
         Teapot(GL_FILL,1,0 , 0,0,0);
         glDisable(GL_POLYGON_OFFSET_FILL);
         Teapot(GL_LINE,1,0 , 1,1,0);
         break;
      //  Solid
      case 2:
         Teapot(GL_FILL,1,0 , 1,1,0);
         break;
      //  Solid with lighting
      case 3:
         Light(1);
         Teapot(GL_FILL,1,0 , 1,1,0);
         Light(0);
         break;
      //  Textured
      case 4:
         glEnable(GL_TEXTURE_2D);
         Teapot(GL_FILL,1,0 , 1,1,1);
         glDisable(GL_TEXTURE_2D);
         break;
      //  Textured with lighting
      case 5:
         Light(1);
         glEnable(GL_TEXTURE_2D);
         Teapot(GL_FILL,1,0 , 1,1,1);
         glDisable(GL_TEXTURE_2D);
         Light(0);
         break;
      //  Solid with lighting - manual evaluation
      case 6:
         Light(1);
         Teapot(GL_FILL,0,0 , 1,1,0);
         Light(0);
         break;
      //  Solid with lighting and normals - manual evaluation
      case 7:
         Light(1);
         Teapot(GL_FILL,0,1 , 1,1,0);
         Light(0);
         break;
      //  Textured with lighting - manual evaluation
      case 8:
         Light(1);
         glEnable(GL_TEXTURE_2D);
         Teapot(GL_FILL,0,0 , 1,1,1);
         glDisable(GL_TEXTURE_2D);
         Light(0);
         break;
      default:
         break;
   }

   //  Draw axes (white)
   glColor3f(1,1,1);
   if (axes)
   {
      glBegin(GL_LINES);
      glVertex3d(0.0,0.0,0.0);
      glVertex3d(len,0.0,0.0);
      glVertex3d(0.0,0.0,0.0);
      glVertex3d(0.0,len,0.0);
      glVertex3d(0.0,0.0,0.0);
      glVertex3d(0.0,0.0,len);
      glEnd();
      //  Label axes
      glRasterPos3d(len,0.0,0.0);
      Print("X");
      glRasterPos3d(0.0,len,0.0);
      Print("Y");
      glRasterPos3d(0.0,0.0,len);
      Print("Z");
   }

   //  Display parameters
   glWindowPos2i(5,5);
   Print("Angle=%d,%d  Dim=%.1f Slices=%d Part=%s Mode=%s",th,ph,dim,n,part[obj],text[mode]);

   //  Render the scene and make it visible
   ErrCheck("display");
   glFlush();
   glutSwapBuffers();
}

/*
 *  GLUT calls this routine when an arrow key is pressed
 */
void special(int key,int x,int y)
{
   //  Right arrow key - increase angle by 5 degrees
   if (key == GLUT_KEY_RIGHT)
      th += 5;
   //  Left arrow key - decrease angle by 5 degrees
   else if (key == GLUT_KEY_LEFT)
      th -= 5;
   //  Up arrow key - increase elevation by 5 degrees
   else if (key == GLUT_KEY_UP)
      ph += 5;
   //  Down arrow key - decrease elevation by 5 degrees
   else if (key == GLUT_KEY_DOWN)
      ph -= 5;
   //  PageUp key - increase dim
   else if (key == GLUT_KEY_PAGE_DOWN)
      dim += 0.1;
   //  PageDown key - decrease dim
   else if (key == GLUT_KEY_PAGE_UP && dim>1)
      dim -= 0.1;
   //  Keep angles to +/-360 degrees
   th %= 360;
   ph %= 360;
   //  Update projection
   Project(60,asp,dim);
   //  Tell GLUT it is necessary to redisplay the scene
   glutPostRedisplay();
}

/*
 *  GLUT calls this routine when a key is pressed
 */
void key(unsigned char ch,int x,int y)
{
   //  Exit on ESC
   if (ch == 27)
      exit(0);
   //  Reset view angle
   else if (ch == '0')
      th = ph = 0;
   //  Cycle number of slices
   else if (ch == '-' && n>1)
      n--;
   else if (ch == '+' && n<MAXN)
      n++;
   //  Cycle through displays
   else if (ch == 'o')
      obj = (obj+1)%(OBJ+1);
   else if (ch == 'O')
      obj = (obj+OBJ)%(OBJ+1);
   //  Toggle control points
   else if (ch == 'c' || ch == 'C')
      ctrl = 1-ctrl;
   //  Toggle reflection
   else if (ch == 'r' || ch == 'R')
      ref = 1-ref;
   //  Toggle axes
   else if (ch == 'a' || ch == 'A')
      axes = 1-axes;
   //  Toggle display modes
   else if (ch == 'm')
      mode = (mode+1)%MODE;
   else if (ch == 'M')
      mode = (mode+MODE-1)%MODE;
   //  Toggle control points
   else if (ch == '[')
      zh -= 5;
   else if (ch == ']')
      zh += 5;
   //  Toggle lid stretch
   else if (ch == 's')
   {
      int k;
      float s = size ? 1/1.12 : 1.12;
      size = 1-size;
      for (k=110;k<118;k++)
      {
         data[k].x *= s;
         data[k].y *= s;
      }
   }
   zh %= 360;
   //  Reproject
   Project(60,asp,dim);
   //  Tell GLUT it is necessary to redisplay the scene
   glutPostRedisplay();
}

/*
 *  GLUT calls this routine when the window is resized
 */
void reshape(int width,int height)
{
   //  Ratio of the width to the height of the window
   asp = (height>0) ? (double)width/height : 1;
   //  Set the viewport to the entire window
   glViewport(0,0, width,height);
   //  Set projection
   Project(60,asp,dim);
}

/*
 *  Start up GLUT and tell it what to do
 */
int main(int argc,char* argv[])
{
   //  Initialize GLUT
   glutInit(&argc,argv);
   //  Request double buffered, true color window with Z buffering at 700x500
   glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
   glutInitWindowSize(700,500);
   glutCreateWindow("Utah Teapot");
   //  Set callbacks
   glutDisplayFunc(display);
   glutReshapeFunc(reshape);
   glutSpecialFunc(special);
   glutKeyboardFunc(key);
   //  Load Texture
   LoadTexBMP("stop.bmp");
   //  Pass control to GLUT so it can interact with the user
   ErrCheck("init");
   glutMainLoop();
   return 0;
}
