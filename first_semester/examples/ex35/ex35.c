/*
 *  Shadow Volumes
 *
 *  This code is extremely inefficient in an attempt to make it easier to read
 *  It also has multiple ways of doing things - real code would do only one.
 *
 *  Use Transform to get light position in local coordinate system
 *
 *  Key bindings:
 *  m/M        Cycle through shadow volume steps (mode)
 *  o/O        Cycle through objects
 *  +/-        Change light elevation
 *  []         Change light position
 *  s/S        Start/stop light movement
 *  l/L        Toggle teapot lid stretch
 *  <>         Decrease/increase number of slices in objects
 *  b/B        Toggle room box
 *  a          Toggle axes
 *  arrows     Change view angle
 *  PgDn/PgUp  Zoom in and out
 *  0          Reset view angle
 *  ESC        Exit
 */
#include "CSCIx229.h"
typedef struct {float x,y,z;} Point;
//  Global variables
int    mode=4;    // Display mode
int    obj=15;    // Display objects (bitmap)
int    move=1;    // Light movement
int    axes=1;    // Display axes
int    box=0;     // Display enclosing box
int    n=8;       // Number of slices
int    th=-30;    // Azimuth of view angle
int    ph=+30;    // Elevation of view angle
int    inf=1;     // Infinity
int    tex2d[3];  // Textures (names)
int    size=0;    // Lid stretch
int    depth=0;   // Stencil depth
double asp=1;     // Aspect ratio
double dim=3;     // Size of world
int    zh=0;      // Light azimuth
float  Ylight=2;  // Elevation of light
int    light;     // Light mode: true=draw polygon, false=draw shadow volume
float  Lpos[4];   // Light position
//  Globals set by Transform
Point  Lp;        // Light position in local coordinate system
Point  Nc,Ec;     // Far or near clipping plane in local coordinate system
//  Mode text
char* text[] = {"Shadowed Object","Front Shadows","Back Shadows","Lit Object","Z-pass","Z-fail"};

#define MAXN 64    // Maximum number of slices (n) and points in a polygon

/*
 *  Define Utah teapot using Bezier patches
 *
 *  Rim, body, lid, and bottom data must be reflected in x and y;
 *  Handle and spout reflected across the y axis only.
 */
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
   { 0.2   ,  0     , 2.7    }, { 0.2   , -0.112 , 2.7    }, { 0.112 , -0.2   , 2.7    }, { 0     , -0.2   , 2.7    },
   { 1.3375,  0     , 2.53125}, { 1.3375, -0.749 , 2.53125}, { 0.749 , -1.3375, 2.53125}, { 0     , -1.3375, 2.53125},
   { 1.4375,  0     , 2.53125}, { 1.4375, -0.805 , 2.53125}, { 0.805 , -1.4375, 2.53125}, { 0     , -1.4375, 2.53125},
   { 1.5   ,  0     , 2.4    }, { 1.5   , -0.84  , 2.4    }, { 0.84  , -1.5   , 2.4    }, { 0     , -1.5   , 2.4    },
   { 1.75  ,  0     , 1.875  }, { 1.75  , -0.98  , 1.875  }, { 0.98  , -1.75  , 1.875  }, { 0     , -1.75  , 1.875  },
   { 2     ,  0     , 1.35   }, { 2     , -1.12  , 1.35   }, { 1.12  , -2     , 1.35   }, { 0     , -2     , 1.35   },
   { 2     ,  0     , 0.9    }, { 2     , -1.12  , 0.9    }, { 1.12  , -2     , 0.9    }, { 0     , -2     , 0.9    },
   { -2    ,  0     , 0.9    }, { 2     ,  0     , 0.45   }, { 2     , -1.12  , 0.45   }, { 1.12  , -2     , 0.45   },
   { 0     , -2     , 0.45   }, { 1.5   ,  0     , 0.225  }, { 1.5   , -0.84  , 0.225  }, { 0.84  , -1.5   , 0.225  },
   { 0     , -1.5   , 0.225  }, { 1.5   ,  0     , 0.15   }, { 1.5   , -0.84  , 0.15   }, { 0.84  , -1.5   , 0.15   },
   { 0     , -1.5   , 0.15   }, {-1.6   ,  0     , 2.025  }, {-1.6   , -0.3   , 2.025  }, {-1.5   , -0.3   , 2.25   },
   {-1.5   ,  0     , 2.25   }, {-2.3   ,  0     , 2.025  }, {-2.3   , -0.3   , 2.025  }, {-2.5   , -0.3   , 2.25   },
   {-2.5   ,  0     , 2.25   }, {-2.7   ,  0     , 2.025  }, {-2.7   , -0.3   , 2.025  }, {-3     , -0.3   , 2.25   },
   {-3     ,  0     , 2.25   }, {-2.7   ,  0     , 1.8    }, {-2.7   , -0.3   , 1.8    }, {-3     , -0.3   , 1.8    },
   {-3     ,  0     , 1.8    }, {-2.7   ,  0     , 1.575  }, {-2.7   , -0.3   , 1.575  }, {-3     , -0.3   , 1.35   },
   {-3     ,  0     , 1.35   }, {-2.5   ,  0     , 1.125  }, {-2.5   , -0.3   , 1.125  }, {-2.65  , -0.3   , 0.9375 },
   {-2.65  ,  0     , 0.9375 }, {-2     , -0.3   , 0.9    }, {-1.9   , -0.3   , 0.6    }, {-1.9   ,  0     , 0.6    },
   { 1.7   ,  0     , 1.425  }, { 1.7   , -0.66  , 1.425  }, { 1.7   , -0.66  , 0.6    }, { 1.7   ,  0     , 0.6    },
   { 2.6   ,  0     , 1.425  }, { 2.6   , -0.66  , 1.425  }, { 3.1   , -0.66  , 0.825  }, { 3.1   ,  0     , 0.825  },
   { 2.3   ,  0     , 2.1    }, { 2.3   , -0.25  , 2.1    }, { 2.4   , -0.25  , 2.025  }, { 2.4   ,  0     , 2.025  },
   { 2.7   ,  0     , 2.4    }, { 2.7   , -0.25  , 2.4    }, { 3.3   , -0.25  , 2.4    }, { 3.3   ,  0     , 2.4    },
   { 2.8   ,  0     , 2.475  }, { 2.8   , -0.25  , 2.475  }, { 3.525 , -0.25  , 2.49375}, { 3.525 ,  0     , 2.49375},
   { 2.9   ,  0     , 2.475  }, { 2.9   , -0.15  , 2.475  }, { 3.45  , -0.15  , 2.5125 }, { 3.45  ,  0     , 2.5125 },
   { 2.8   ,  0     , 2.4    }, { 2.8   , -0.15  , 2.4    }, { 3.2   , -0.15  , 2.4    }, { 3.2   ,  0     , 2.4    },
   { 0     ,  0     , 3.15   }, { 0.8   ,  0     , 3.15   }, { 0.8   , -0.45  , 3.15   }, { 0.45  , -0.8   , 3.15   },
   { 0     , -0.8   , 3.15   }, { 0     ,  0     , 2.85   }, { 1.4   ,  0     , 2.4    }, { 1.4   , -0.784 , 2.4    },
   { 0.784 , -1.4   , 2.4    }, { 0     , -1.4   , 2.4    }, { 0.4   ,  0     , 2.55   }, { 0.4   , -0.224 , 2.55   },
   { 0.224 , -0.4   , 2.55   }, { 0     , -0.4   , 2.55   }, { 1.3   ,  0     , 2.55   }, { 1.3   , -0.728 , 2.55   },
   { 0.728 , -1.3   , 2.55   }, { 0     , -1.3   , 2.55   }, { 1.3   ,  0     , 2.4    }, { 1.3   , -0.728 , 2.4    },
   { 0.728 , -1.3   , 2.4    }, { 0     , -1.3   , 2.4    }, { 0     ,  0     , 0      }, { 1.425 , -0.798 , 0      },
   { 1.5   ,  0     , 0.075  }, { 1.425 ,  0     , 0      }, { 0.798 , -1.425 , 0      }, { 0     , -1.5   , 0.075  },
   { 0     , -1.425 , 0      }, { 1.5   , -0.84  , 0.075  }, { 0.84  , -1.5   , 0.075  },
   };

/*
 *  Set color with shadows
 */
void Color(float R,float G,float B)
{
   //  Use black to draw shadowed objects for demonstration
   //  purposes in mode=3 (lit parts of objects only)
   if (light<0 && mode==3)
      glColor3f(0,0,0);
   //  Shaded color is 1/2 of lit color
   else if (light<0)
      glColor3f(0.5*R,0.5*G,0.5*B);
   //  Lit color
   else
      glColor3f(R,G,B);
}

/*
 *  Calculate shadow location
 */
Point Shadow(Point P)
{
   double lambda;
   Point  S;

   //  Fixed lambda
   if (inf)
      lambda = 1024;
   //  Calculate lambda for clipping plane
   else
   {
      // lambda = (E-L).N / (P-L).N = (E.N - L.N) / (P.N - L.N)
      double LN = Lp.x*Nc.x+Lp.y*Nc.y + Lp.z*Nc.z;
      double PLN = P.x*Nc.x+P.y*Nc.y+P.z*Nc.z - LN;
      lambda = (fabs(PLN)>1e-10) ? (Ec.x*Nc.x+Ec.y*Nc.y+Ec.z*Nc.z - LN)/PLN : 1024;
      //  If lambda<0, then the plane is behind the light
      //  If lambda [0,1] the plane is between the light and the object
      //  So make lambda big if this is true
      if (lambda<=1) lambda = 1024;
   }

   //  Calculate shadow location
   S.x = lambda*(P.x-Lp.x) + Lp.x;
   S.y = lambda*(P.y-Lp.y) + Lp.y;
   S.z = lambda*(P.z-Lp.z) + Lp.z;
   return S;
}

/*
 *  Draw polygon or shadow volume
 *    P[] array of vertexes making up the polygon
 *    N[] array of normals (not used with shadows)
 *    T[] array of texture coordinates (not used with shadows)
 *    n   number of vertexes
 *  Killer fact: the order of points MUST be CCW
 */
void DrawPolyShadow(Point P[],Point N[],Point T[],int n)
{
   int k;
   //  Draw polygon with normals and textures
   if (light)
   {
      glBegin(GL_POLYGON);
      for (k=0;k<n;k++)
      {
         glNormal3f(N[k].x,N[k].y,N[k].z);
         glTexCoord2f(T[k].x,T[k].y);
         glVertex3f(P[k].x,P[k].y,P[k].z);
      }
      glEnd();
   }
   //  Draw shadow volume
   else
   {
      //  Check if polygon is visible
      int vis = 0;
      for (k=0;k<n;k++)
         vis = vis | (N[k].x*(Lp.x-P[k].x) + N[k].y*(Lp.y-P[k].y) + N[k].z*(Lp.z-P[k].z) >= 0);
      //  Draw shadow volume only for those polygons facing the light
      if (vis)
      {
         //  Shadow coordinates (at infinity)
         Point S[MAXN];
         if (n>MAXN) Fatal("Too many points in polygon %d\n",n);
         //  Project shadow
         for (k=0;k<n;k++)
            S[k] = Shadow(P[k]);
         //  Front face
         glBegin(GL_POLYGON);
         for (k=0;k<n;k++)
            glVertex3f(P[k].x,P[k].y,P[k].z);
         glEnd();
         //  Back face
         glBegin(GL_POLYGON);
         for (k=n-1;k>=0;k--)
            glVertex3f(S[k].x,S[k].y,S[k].z);
         glEnd();
         //  Sides
         glBegin(GL_QUAD_STRIP);
         for (k=0;k<=n;k++)
         {
            glVertex3f(P[k%n].x,P[k%n].y,P[k%n].z);
            glVertex3f(S[k%n].x,S[k%n].y,S[k%n].z);
         }
         glEnd();
      }
   }
}

/*
 *  Evaluate 2D Bezier surface
 */
#define Bezier(x)  V*V*V*(U*U*U*p[0][0].x + 3*U*U*u*p[0][1].x + 3*U*u*u*p[0][2].x + u*u*u*p[0][3].x) \
               + 3*V*V*v*(U*U*U*p[1][0].x + 3*U*U*u*p[1][1].x + 3*U*u*u*p[1][2].x + u*u*u*p[1][3].x) \
               + 3*V*v*v*(U*U*U*p[2][0].x + 3*U*U*u*p[2][1].x + 3*U*u*u*p[2][2].x + u*u*u*p[2][3].x) \
               +   v*v*v*(U*U*U*p[3][0].x + 3*U*U*u*p[3][1].x + 3*U*u*u*p[3][2].x + u*u*u*p[3][3].x)
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

/*
 *  Evaluate 2D Bezier normal
 */
#define ddu(x)  -U*U*(V*V*V*p[0][0].x + 3*V*V*v*p[1][0].x + 3*V*v*v*p[2][0].x + v*v*v*p[3][0].x) \
         + (1-3*u)*U*(V*V*V*p[0][1].x + 3*V*V*v*p[1][1].x + 3*V*v*v*p[2][1].x + v*v*v*p[3][1].x) \
         + u*(2-3*u)*(V*V*V*p[0][2].x + 3*V*V*v*p[1][2].x + 3*V*v*v*p[2][2].x + v*v*v*p[3][2].x) \
         +       u*u*(V*V*V*p[0][3].x + 3*V*V*v*p[1][3].x + 3*V*v*v*p[2][3].x + v*v*v*p[3][3].x)
#define ddv(x)  -V*V*(U*U*U*p[0][0].x + 3*U*U*u*p[0][1].x + 3*U*u*u*p[0][2].x + u*u*u*p[0][3].x) \
         + (1-3*v)*V*(U*U*U*p[1][0].x + 3*U*U*u*p[1][1].x + 3*U*u*u*p[1][2].x + u*u*u*p[1][3].x) \
         + v*(2-3*v)*(U*U*U*p[2][0].x + 3*U*U*u*p[2][1].x + 3*U*u*u*p[2][2].x + u*u*u*p[2][3].x) \
         +       v*v*(U*U*U*p[3][0].x + 3*U*U*u*p[3][1].x + 3*U*u*u*p[3][2].x + u*u*u*p[3][3].x)
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
 *  Perform Crout LU decomposition of M in place
 *     M  4x4 matrix
 *     I  Pivot index
 *  Calls Fatal if the matrix is singular
 */
#define M(i,j) (M[4*j+i])
void Crout(double M[16],int I[4])
{
   int i,j,k;  // Counters

   //  Initialize index
   for (j=0;j<4;j++)
      I[j] = j;

   //  Pivot matrix to maximize diagonal
   for (j=0;j<3;j++)
   {
      //  Find largest absolute value in this column (J)
      int J=j;
      for (k=j+1;k<4;k++)
         if (fabs(M(k,j)) > fabs(M(J,j))) J = k;
      //  Swap rows if required
      if (j!=J)
      {
         k=I[j]; I[j]=I[J]; I[J]=k;
         for (k=0;k<4;k++)
         {
            double t=M(j,k); M(j,k)=M(J,k); M(J,k)=t;
         }
      }
   }

   //  Perform Crout LU decomposition
   for (j=0;j<4;j++)
   {
      //  Upper triangular matrix
      for (i=j;i<4;i++)
         for (k=0;k<j;k++)
            M(j,i) -= M(k,i)*M(j,k);
      if (fabs(M(j,j))<1e-10) Fatal("Singular transformation matrix\n");
      //  Lower triangular matrix
      for (i=j+1;i<4;i++)
      {
         for (k=0;k<j;k++)
            M(i,j) -= M(k,j)*M(i,k);
         M(i,j) /= M(j,j);
      }
   }
}

/*
 *  Backsolve LU decomposition
 *     M  4x4 matrix
 *     I  Pivot index
 *     Bx,By,Bz,Bw is RHS
 *     Returns renormalized Point
 */
Point Backsolve(double M[16],int I[4],double Bx,double By,double Bz,double Bw)
{
   int    i,j;                  //  Counters
   double x[4];                 //  Solution vector
   Point  X;                    //  Solution Point
   double b[4] = {Bx,By,Bz,Bw}; //  RHS

   //  Backsolve
   for (i=0;i<4;i++)
   {
      x[i] = b[I[i]];
      for (j=0;j<i;j++)
         x[i] -= M(i,j)*x[j];
   }
   for (i=3;i>=0;i--)
   {
      for (j=i+1;j<4;j++)
         x[i] -= M(i,j)*x[j];
      x[i] /= M(i,i);
   }

   //  Renormalize
   if (fabs(x[3])<1e-10) Fatal("Light position W is zero\n");
   X.x = x[0]/x[3];
   X.y = x[1]/x[3];
   X.z = x[2]/x[3];
   return X;
}

/*
 *  Set up transform
 *     Push and apply transformation
 *     Calculate light position in local coordinate system (Lp)
 *     Calculate clipping plane in local coordinate system (Ec,Nc)
 *  Global variables set: Lp,Nc,Ec
 *  Killer fact 1:  You MUST call this to set transforms for objects
 *  Killer fact 2:  You MUST call glPopMatrix to undo the transform
 */
void Transform(float x0,float y0,float z0,
               float Sx,float Sy,float Sz,
               float th,float ph)
{
   double M[16]; // Transformation matrix
   int    I[4];  // Pivot
   double Z;     // Location of clip plane

   //  Save current matrix
   glPushMatrix();

   //  Build transformation matrix and copy into M
   glPushMatrix();
   glLoadIdentity();
   glTranslated(x0,y0,z0);
   glRotated(ph,1,0,0);
   glRotated(th,0,1,0);
   glScaled(Sx,Sy,Sz);
   glGetDoublev(GL_MODELVIEW_MATRIX,M);
   glPopMatrix();

   //  Apply M to existing transformations
   glMultMatrixd(M);

   /*
    *  Determine light position in this coordinate system
    */
   Crout(M,I);
   Lp = Backsolve(M,I,Lpos[0],Lpos[1],Lpos[2],Lpos[3]);

   /*
    *  Determine clipping plane E & N
    *  Use the entire MODELVIEW matrix here
    */
   glGetDoublev(GL_MODELVIEW_MATRIX,M);
   //  Normal is down the Z axis (0,0,1) since +/- doesn't matter here
   //  The normal matrix is the inverse of the transpose of M
   Nc.x = M(2,0);
   Nc.y = M(2,1);
   Nc.z = M(2,2);
   //  Far  clipping plane for Z-fail should be just less than 8*dim
   //  Near clipping plane for Z-pass should be just more than dim/8
   Crout(M,I);
   Z = (mode==5) ? -7.9*dim : -0.13*dim;
   Ec = Backsolve(M,I,0,0,Z,1);
}

/*
 *  Draw a cube
 */
static void Cube(float x,float y,float z , float th,float ph , float D)
{
   //  Vertexes
   Point P1[] = { {-1,-1,+1} , {+1,-1,+1} , {+1,+1,+1} , {-1,+1,+1} }; //  Front
   Point P2[] = { {+1,-1,-1} , {-1,-1,-1} , {-1,+1,-1} , {+1,+1,-1} }; //  Back
   Point P3[] = { {+1,-1,+1} , {+1,-1,-1} , {+1,+1,-1} , {+1,+1,+1} }; //  Right
   Point P4[] = { {-1,-1,-1} , {-1,-1,+1} , {-1,+1,+1} , {-1,+1,-1} }; //  Left
   Point P5[] = { {-1,+1,+1} , {+1,+1,+1} , {+1,+1,-1} , {-1,+1,-1} }; //  Top
   Point P6[] = { {-1,-1,-1} , {+1,-1,-1} , {+1,-1,+1} , {-1,-1,+1} }; //  Bottom
   //  Normals
   Point N1[] = { { 0, 0,+1} , { 0, 0,+1} , { 0, 0,+1} , { 0, 0,+1} }; //  Front
   Point N2[] = { { 0, 0,-1} , { 0, 0,-1} , { 0, 0,-1} , { 0, 0,-1} }; //  Back
   Point N3[] = { {+1, 0, 0} , {+1, 0, 0} , {+1, 0, 0} , {+1, 0, 0} }; //  Right
   Point N4[] = { {-1, 0, 0} , {-1, 0, 0} , {-1, 0, 0} , {-1, 0, 0} }; //  Left
   Point N5[] = { { 0,+1, 0} , { 0,+1, 0} , { 0,+1, 0} , { 0,+1, 0} }; //  Top
   Point N6[] = { { 0,-1, 0} , { 0,-1, 0} , { 0,-1, 0} , { 0,-1, 0} }; //  Bottom
   //  Textures
   Point T[] = { {0,0,0} , {1,0,0} , {1,1,0} , {0,1,0} };

   Transform(x,y,z,D,D,D,th,ph);
   Color(1,0,1);
   DrawPolyShadow(P1,N1,T,4); //  Front
   DrawPolyShadow(P2,N2,T,4); //  Back
   DrawPolyShadow(P3,N3,T,4); //  Right
   DrawPolyShadow(P4,N4,T,4); //  Left
   DrawPolyShadow(P5,N5,T,4); //  Top
   DrawPolyShadow(P6,N6,T,4); //  Bottom
   glPopMatrix();
}

/*
 *  Draw a cylinder
 */
static void Cylinder(float x,float y,float z , float th,float ph , float R,float H)
{
   int i,j;   // Counters
   int N=4*n; // Number of slices

   Transform(x,y,z,R,R,H,th,ph);
   Color(0,1,1);
   //  Two end caps (fan of triangles)
   for (j=-1;j<=1;j+=2)
      for (i=0;i<N;i++)
      {
         float th0 = j* i   *360.0/N;
         float th1 = j*(i+1)*360.0/N;
         Point P[3] = { {0,0,j} , {Cos(th0),Sin(th0),j} , {Cos(th1),Sin(th1),j} };
         Point N[3] = { {0,0,j} , {       0,       0,j} , {       0,       0,j} };
         Point T[3] = { {0,0,0} , {Cos(th0),Sin(th0),0} , {Cos(th1),Sin(th1),0} };
         DrawPolyShadow(P,N,T,3);
      }
   //  Cylinder Body (strip of quads)
   for (i=0;i<N;i++)
   {
      float th0 =  i   *360.0/N;
      float th1 = (i+1)*360.0/N;
      Point P[4] = { {Cos(th0),Sin(th0),+1} , {Cos(th0),Sin(th0),-1} , {Cos(th1),Sin(th1),-1} , {Cos(th1),Sin(th1),+1} };
      Point N[4] = { {Cos(th0),Sin(th0), 0} , {Cos(th0),Sin(th0), 0} , {Cos(th1),Sin(th1), 0} , {Cos(th1),Sin(th1), 0} };
      Point T[4] = { {       0,th0/90.0, 0} , {       2,th0/90.0, 0} , {       2,th1/90.0, 0} , {       0,th1/90.0, 0} };
      DrawPolyShadow(P,N,T,4);
   }

   glPopMatrix();
}

/*
 *  Draw torus
 */
static void Torus(float x,float y,float z , float th,float ph , float S,float r)
{
   int i,j;   // Counters
   int N=4*n; // Number of slices

   Transform(x,y,z,S,S,S,th,ph);
   Color(1,1,0);
   //  Loop along ring
   for (i=0;i<N;i++)
   {
      float th0 =  i   *360.0/N;
      float th1 = (i+1)*360.0/N;
      //  Loop around ring
      for (j=0;j<N;j++)
      {
         float ph0 =  j   *360.0/N;
         float ph1 = (j+1)*360.0/N;
         Point P[4] = { {Cos(th1)*(1+r*Cos(ph0)),-Sin(th1)*(1+r*Cos(ph0)),r*Sin(ph0)} ,
                        {Cos(th0)*(1+r*Cos(ph0)),-Sin(th0)*(1+r*Cos(ph0)),r*Sin(ph0)} ,
                        {Cos(th0)*(1+r*Cos(ph1)),-Sin(th0)*(1+r*Cos(ph1)),r*Sin(ph1)} ,
                        {Cos(th1)*(1+r*Cos(ph1)),-Sin(th1)*(1+r*Cos(ph1)),r*Sin(ph1)} };
         Point N[4] = { {Cos(th1)*Cos(ph0) , -Sin(th1)*Cos(ph0) , Sin(ph0)} ,
                        {Cos(th0)*Cos(ph0) , -Sin(th0)*Cos(ph0) , Sin(ph0)} ,
                        {Cos(th0)*Cos(ph1) , -Sin(th0)*Cos(ph1) , Sin(ph1)} ,
                        {Cos(th1)*Cos(ph1) , -Sin(th1)*Cos(ph1) , Sin(ph1)} };
         Point T[4] = { {th1/30.0 , ph0/180.0 , 0} ,
                        {th0/30.0 , ph0/180.0 , 0} ,
                        {th0/30.0 , ph1/180.0 , 0} ,
                        {th1/30.0 , ph1/180.0 , 0} };
         DrawPolyShadow(P,N,T,4);
      }
   }
   glPopMatrix();
}

/*
 *  Draw patch
 *
 *  patch = set of point numbers
 *  Sx = x reflection (+/-1)
 *  Sy = y reflection (+/-1)
 *  Sz = z reflection (+/-1)
 */
static void Patch(int patch[4][4],float Sx,float Sy,float Sz)
{
   int   i,j,k;
   Point p[4][4],P[MAXN+1][MAXN+1],N[MAXN+1][MAXN+1],T[MAXN+1][MAXN+1];

   //  Copy data with reflection
   for (k=0;k<4;k++)
   {
      int K = Sx*Sy*Sz<0 ? 3-k : k;
      for (j=0;j<4;j++)
      {
         int l = patch[j][K];
         p[j][k].x = +Sx*data[l].x;
         p[j][k].z = -Sy*data[l].y;
         p[j][k].y = +Sz*data[l].z;
      }
   }

   //  Evaluate vertexes
   for (i=0;i<=n;i++)
      for (j=0;j<=n;j++)
      {
         float u = (float)i/n;
         float v = (float)j/n;
         //  Vertex coordinates
         P[i][j] = Bezier2D(p,u,v);
         //  Normal
         N[i][j] = Normal2D(p,u,v);
         //  Texture coordinates
         T[i][j].x = 1-u;
         T[i][j].y = 1-v;
         T[i][j].z = 0;
      }

   //  Draw quads
   for (i=0;i<n;i++)
      for (j=0;j<n;j++)
      {
         Point p[4] = {P[i][j],P[i+1][j],P[i+1][j+1],P[i][j+1]};
         Point n[4] = {N[i][j],N[i+1][j],N[i+1][j+1],N[i][j+1]};
         Point t[4] = {T[i][j],T[i+1][j],T[i+1][j+1],T[i][j+1]};
         DrawPolyShadow(p,n,t,4);
      }
}

/*
 *  Draw teapot
 */
static void Teapot(float x,float y,float z , float th,float ph , float S)
{
   int i;

   Transform(x,y,z,S,S,S,th,ph);
   Color(0,1,0);
   //  Draw patches
   for (i=0;i<10;i++)
   {
      //  Draw patch reflected to 4 quadrants
      if (i<6)
      {
         Patch(teapot[i],+1,+1,1);
         Patch(teapot[i],+1,-1,1);
         Patch(teapot[i],-1,+1,1);
         Patch(teapot[i],-1,-1,1);
      }
      //  Draw patch reflected to 2 hemispheres
      else
      {
         Patch(teapot[i],+1,+1,1);
         Patch(teapot[i],+1,-1,1);
      }
   }
   glPopMatrix();
}

/*
 *  Draw a wall
 */
static void Wall(float x,float y,float z, float th,float ph , float Sx,float Sy,float Sz , float St)
{
   int   i,j;
   float s=1.0/n;
   float t=0.5*St/n;
   Transform(x,y,z,Sx,Sy,Sz,th,ph);

   glNormal3f(0,0,1);
   for (j=-n;j<n;j++)
   {
      glBegin(GL_QUAD_STRIP);
      for (i=-n;i<=n;i++)
      {
         glTexCoord2f((i+n)*t,(j  +n)*t); glVertex3f(i*s,    j*s,-1);
         glTexCoord2f((i+n)*t,(j+1+n)*t); glVertex3f(i*s,(j+1)*s,-1);
      }
      glEnd();
   }

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
      glLightfv(GL_LIGHT0,GL_POSITION,Lpos);
   }
   else
      glDisable(GL_LIGHTING);
}

/*
 *  Draw scene
 *    light>0  lit colors
 *    light<0  shaded colors
 *    light=0  shadow volumes
 */
void Scene(int Light)
{
   int k;  // Counters used to draw floor
 
   //  Set global light switch (used in DrawPolyShadow)
   light = Light;

   //  Texture (pi.bmp)
   glBindTexture(GL_TEXTURE_2D,tex2d[2]);
   //  Enable textures if not drawing shadow volumes
   if (light) glEnable(GL_TEXTURE_2D);

   //  Draw objects         x    y   z          th,ph    dims   
   if (obj&0x01)     Cube(-0.8,+0.8,0.0 , -0.25*zh, 0  , 0.3    );
   if (obj&0x02) Cylinder(+0.8,+0.5,0.0 ,   0.5*zh,zh  , 0.2,0.5);
   if (obj&0x04)    Torus(+0.5,-0.8,0.0 ,        0,zh  , 0.5,0.2);
   if (obj&0x08)   Teapot(-0.5,-0.5,0.0 ,     2*zh, 0  , 0.25   );

   //  Disable textures
   if (light) glDisable(GL_TEXTURE_2D);

   //  The floor, ceiling and walls don't cast a shadow, so bail here
   if (!light) return;

   //  Always enable textures
   glEnable(GL_TEXTURE_2D);
   Color(1,1,1);

   //  Water texture for floor and ceiling
   glBindTexture(GL_TEXTURE_2D,tex2d[0]);
   for (k=-1;k<=box;k+=2)
      Wall(0,0,0, 0,90*k , 8,8,box?6:2 , 4);
   //  Crate texture for walls
   glBindTexture(GL_TEXTURE_2D,tex2d[1]);
   for (k=0;k<4*box;k++)
      Wall(0,0,0, 90*k,0 , 8,box?6:2,8 , 1);

   //  Disable textures
   glDisable(GL_TEXTURE_2D);
}

/*
 *  Draw scene using shadow volumes
 */
static void DrawSceneWithShadows()
{
   //  PASS 1:
   //  Draw whole scene as shadowed
   //  Lighting is still off
   //  The color should be what the object looks like when in the shadow
   //  This sets the object depths in the Z-buffer
   Scene(-1);

   //  Make color buffer and Z buffer read-only
   glColorMask(0,0,0,0);
   glDepthMask(0);
   //  Enable stencil
   glEnable(GL_STENCIL_TEST);
   //  Always draw regardless of the stencil value
   glStencilFunc(GL_ALWAYS,0,0xFFFFFFFF);

   //  Enable face culling
   glEnable(GL_CULL_FACE);

   //  This switch is for demonstration purposes only
   //  Normally you would pick just one method (mode=4 or mode=5)
   //  Modes 0 to 3 are for demonstration purposes
   //  which do not result in correctly rendered shadows
   switch (mode)
   {
      //  Shadowed objects only - do nothing
      case 0:
         break;
      //  Front shadows
      //  This is for demonstation purposes only
      case 1:
         glFrontFace(GL_CCW);
         glStencilOp(GL_KEEP,GL_KEEP,GL_INCR);
         Scene(0);
         break;
      //  Back shadows
      //  This is for demonstation purposes only
      case 2:
         glFrontFace(GL_CW);
         glStencilOp(GL_KEEP,GL_KEEP,GL_INCR);
         Scene(0);
         break;
      //  Lit parts of objects only (do Z-pass to help find this)
      case 3:
      //  Z-pass variation
      //  Count from the eye to the object
      case 4:
         //  PASS 2:
         //  Draw only the front faces of the shadow volume
         //  Increment the stencil value on Z pass
         //  Depth and color buffers unchanged
         glFrontFace(GL_CCW);
         glStencilOp(GL_KEEP,GL_KEEP,GL_INCR);
         Scene(0);
         //  PASS 3:
         //  Draw only the back faces of the shadow volume
         //  Decrement the stencil value on Z pass
         //  Depth and color buffers unchanged
         glFrontFace(GL_CW);
         glStencilOp(GL_KEEP,GL_KEEP,GL_DECR);
         Scene(0);
         break;
      //  Z-fail variation
      //  Count from the object to infinity
      case 5:
         //  PASS 2:
         //  Draw only the back faces of the shadow volume
         //  Increment the stencil value on Z fail
         //  Depth and color buffers unchanged
         glFrontFace(GL_CW);
         glStencilOp(GL_KEEP,GL_INCR,GL_KEEP);
         Scene(0);
         //  PASS 3:
         //  Draw only the front faces of the shadow volume
         //  Decrement the stencil value on Z fail
         //  Depth and color buffers unchanged
         glFrontFace(GL_CCW);
         glStencilOp(GL_KEEP,GL_DECR,GL_KEEP);
         Scene(0);
         break;
      default:
         break;
   }

   //  Disable face culling
   glDisable(GL_CULL_FACE);
   //  Make color mask and depth buffer read-write
   glColorMask(1,1,1,1);
   //  Update the color only where the stencil value is 0
   //  Do not change the stencil
   glStencilFunc(GL_EQUAL,0,0xFFFFFFFF);
   glStencilOp(GL_KEEP,GL_KEEP,GL_KEEP);
   //  Redraw only if depth matches front object
   glDepthFunc(GL_EQUAL);

   //  PASS 4:
   //  Enable lighting
   //  Render the parts of objects not in shadows
   //  (The mode test is for demonstrating unlit objects only)
   Light(1);
   if (mode) Scene(1);

   //  Undo changes (no stencil test, draw if closer and update Z-buffer)
   glDisable(GL_STENCIL_TEST);
   glDepthFunc(GL_LESS); 
   glDepthMask(1);
   //  Disable lighting
   Light(0);
}

/*
 *  OpenGL (GLUT) calls this routine to display the scene
 */
void display()
{
   //  Length of axes
   const double len=2.0;
   //  Eye position
   float Ex = -2*dim*Sin(th)*Cos(ph);
   float Ey = +2*dim        *Sin(ph);
   float Ez = +2*dim*Cos(th)*Cos(ph);
   //  Light position
   Lpos[0] = 2*Cos(zh);
   Lpos[1] = Ylight;
   Lpos[2] = 2*Sin(zh);
   Lpos[3] = 1;

   //  Erase the window and the depth and stencil buffers
   glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);
   //  Enable Z-buffering in OpenGL
   glEnable(GL_DEPTH_TEST);

   //  Set perspective
   glLoadIdentity();
   gluLookAt(Ex,Ey,Ez , 0,0,0 , 0,Cos(ph),0);

   //  Draw light position as sphere (still no lighting here)
   glColor3f(1,1,1);
   glPushMatrix();
   glTranslated(Lpos[0],Lpos[1],Lpos[2]);
   glutSolidSphere(0.03,10,10);
   glPopMatrix();

   //  Draw the scene with shadows
   DrawSceneWithShadows();

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
   Print("Ylight=%.1f Angle=%d,%d,%d  Dim=%.1f Slices=%d Mode=%s Inf=%s Stencil=%d",
     Ylight,th,ph,zh,dim,n,text[mode],inf?"Fixed":"Calculated",depth);

   //  Render the scene and make it visible
   ErrCheck("display");
   glFlush();
   glutSwapBuffers();
}

/*
 *  GLUT calls this routine when the window is resized
 */
void idle()
{
   //  Elapsed time in seconds
   double t = glutGet(GLUT_ELAPSED_TIME)/1000.0;
   zh = fmod(90*t,1440.0);
   //  Tell GLUT it is necessary to redisplay the scene
   glutPostRedisplay();
}

/*
 *  GLUT calls this routine when an arrow key is pressed
 */
void special(int key,int x,int y)
{
   //  Right arrow key - increase angle by 5 degrees
   if (key == GLUT_KEY_RIGHT)
      th += 1;
   //  Left arrow key - decrease angle by 5 degrees
   else if (key == GLUT_KEY_LEFT)
      th -= 1;
   //  Up arrow key - increase elevation by 5 degrees
   else if (key == GLUT_KEY_UP)
      ph += 1;
   //  Down arrow key - decrease elevation by 5 degrees
   else if (key == GLUT_KEY_DOWN)
      ph -= 1;
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
   //  Toggle axes
   else if (ch == 'a' || ch == 'A')
      axes = 1-axes;
   //  Toggle display modes
   else if (ch == 'm')
      mode = (mode+1)%6;
   else if (ch == 'M')
      mode = (mode+5)%6;
   //  Toggle light movement
   else if (ch == 's' || ch == 'S')
      move = 1-move;
   //  Toggle box
   else if (ch == 'b' || ch == 'B')
      box = 1-box;
   //  Toggle infinity calculation
   else if (ch == 'i' || ch == 'I')
      inf = 1-inf;
   //  Toggle objects
   else if (ch == 'o')
      obj = (obj+1)%16;
   else if (ch == 'O')
      obj = (obj+15)%16;
   //  Light elevation
   else if (ch=='-')
      Ylight -= 0.1;
   else if (ch=='+')
      Ylight += 0.1;
   //  Light azimuth
   else if (ch=='[')
      zh -= 1;
   else if (ch==']')
      zh += 1;
   //  Number of patches
   else if (ch=='<' && n>1)
      n--;
   else if (ch=='>' && n<MAXN)
      n++;
   //  Toggle lid stretch
   else if (ch == 'l')
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
   //  Set idle function
   glutIdleFunc(move?idle:NULL);
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
   //  Request double buffered, true color window with Z buffering & stencil at 600x600
   glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE | GLUT_STENCIL);
   glutInitWindowSize(600,600);
   glutCreateWindow("Shadow Volumes");
   //  Set callbacks
   glutDisplayFunc(display);
   glutReshapeFunc(reshape);
   glutSpecialFunc(special);
   glutKeyboardFunc(key);
   glutIdleFunc(move?idle:NULL);
   //  Check stencil depth
   glGetIntegerv(GL_STENCIL_BITS,&depth);
   if (depth<=0) Fatal("No stencil buffer\n");
   //  Load textures
   tex2d[0] = LoadTexBMP("water.bmp");
   tex2d[1] = LoadTexBMP("crate.bmp");
   tex2d[2] = LoadTexBMP("pi.bmp");
   //  Pass control to GLUT so it can interact with the user
   ErrCheck("init");
   glutMainLoop();
   return 0;
}
