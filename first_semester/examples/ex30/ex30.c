/*
 *  Polygons
 *
 *  Demontrates drawing non-convex polygons.
 *
 *  Key bindings:
 *  +/-    Cycle through polygon modes
 *  ESC    Exit
 */
#include "CSCIx229.h"
typedef struct
{
   double x,y,z;
   float  r,g,b;
} Vertex;
int mode=0;        //  Polygon mode
double XYZ[10][6]; //  Large pentagon
double xyz[5][6];  //  Small pentagon
Vertex V[10];      //  Large pentagon
Vertex v[5];       //  Small pentagon
#define LEN 256    //  Max working count
int    N;          //  Work vertex count
Vertex W[LEN];     //  Work vertexes
//  Modes
#define MODES 16
char* text[] = {"Outline","Filled Outline","Star Outline","Filled Star","Triangles and Pentagon","Quads",
                "Tesselation Outline (Pos)","Tesselation Fill (Pos)",
                "Tesselation Outline (Odd)","Tesselation Fill (Odd)",
                "Tesselation Outline with Hole (Pos)","Tesselation Fill with Hole (Pos)",
                "Tesselation Outline with Hole (Odd)","Tesselation Fill with Hole (Odd)",
                "Tesselation Outline with Hole (Odd)","Tesselation Fill with Hole (Odd)"};

/*
 *  Report Tessalation errors
 */
void TessError(GLenum err)
{
   Fatal("Tessellation Error: %s\n",gluErrorString(err));
}

/*
 *  Generate vertex that combines vertices
 */
void TessCombine(double coords[3],Vertex* V[4],float w[4],Vertex** result)
{
   if (N>=LEN) Fatal("Out of working vertexes [%d]\n",N);
   //  Coordinates
   W[N].x = coords[0];
   W[N].y = coords[1];
   W[N].z = coords[2];
   //  Weight colors
   W[N].r = w[0]*V[0]->r + w[1]*V[1]->r + w[2]*V[2]->r + w[3]*V[3]->r;
   W[N].g = w[0]*V[0]->g + w[1]*V[1]->g + w[2]*V[2]->g + w[3]*V[3]->g;
   W[N].b = w[0]*V[0]->b + w[1]*V[1]->b + w[2]*V[2]->b + w[3]*V[3]->b;
   //  Return results
   *result = W + N++;
}

/*
 *  Draw vertex
 */
void DrawVertex(Vertex* v)
{
   glColor3f(v->r,v->g,v->b);
   glVertex3d(v->x,v->y,v->z);
}

/*
 *  Tesselated star polygon
 */
void TesselatedStar(int star,int type,int hole,int rule)
{
   int k;
   //  Create new Tesselator
   GLUtesselator* tess = gluNewTess();
   //  Set polygon type (Line or Fill) and line width
   glPolygonMode(GL_FRONT_AND_BACK,type);
   if (type==GL_LINE) glLineWidth(3);
   //  Set winding rule
   gluTessProperty(tess,GLU_TESS_WINDING_RULE,rule);
   //  Set callbacks
   gluTessCallback(tess,GLU_TESS_BEGIN  ,(void*)glBegin);
   gluTessCallback(tess,GLU_TESS_END    ,(void*)glEnd);
   gluTessCallback(tess,GLU_TESS_VERTEX ,(void*)DrawVertex);
   gluTessCallback(tess,GLU_TESS_COMBINE,(void*)TessCombine);
   gluTessCallback(tess,GLU_TESS_ERROR  ,(void*)TessError);
   //  Start polygon
   N = 0;
   gluTessBeginPolygon(tess,NULL);
   //  Draw outside star
   if (star)
   {
      gluTessBeginContour(tess);
      for (k=0;k<5;k++)
         gluTessVertex(tess,XYZ[(2*k)%5],&V[(2*k)%5]);
      gluTessEndContour(tess);
   }
   else
   {
      //  Draw outside pentagon
      gluTessBeginContour(tess);
      for (k=0;k<5;k++)
         gluTessVertex(tess,XYZ[k],&V[k]);
      gluTessEndContour(tess);
      //  Draw inside pentagon
      gluTessBeginContour(tess);
      for (k=0;k<5;k++)
         gluTessVertex(tess,xyz[k],&v[k]);
      gluTessEndContour(tess);
   }
   //  Draw diamond
   if (hole)
   {
      gluTessBeginContour(tess);
      for (k=5;k<9;k++)
         gluTessVertex(tess,XYZ[k],&V[k]);
      gluTessEndContour(tess);
   }
   //  End of polygon
   gluTessEndPolygon(tess);
   //  Delete tessalator
   gluDeleteTess(tess);
   //  Set polygon mode back to fill
   if (type==GL_LINE) glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
}

/*
 *  OpenGL (GLUT) calls this routine to display the scene
 */
void display()
{
   int k;
   //  Erase the window and the depth buffer
   glClear(GL_COLOR_BUFFER_BIT);

   //  Draw star or pentagon
   glColor3f(1,1,0);
   switch (mode)
   {
      //  Outline
      case 0:
         break;
      //  Filled outline
      case 1:
         glBegin(GL_POLYGON);
         for (k=0;k<5;k++)
            DrawVertex(V+k);
         glEnd();
         break;
      //  Outline star
      case 2:
         glLineWidth(3);
         glBegin(GL_LINE_LOOP);
         for (k=0;k<5;k++)
            DrawVertex(V+(2*k)%5);
         glEnd();
         break;
      //  Filled star (INCORRECT)
      case 3:
         glBegin(GL_POLYGON);
         for (k=0;k<5;k++)
            DrawVertex(V+(2*k)%5);
         glEnd();
         break;
      //  Filled star (Triangles + pentagon)
      case 4:
         glBegin(GL_TRIANGLES);
         for (k=0;k<5;k++)
         {
            DrawVertex(v+(k+4)%5);
            DrawVertex(V+k);
            DrawVertex(v+k);
         }
         glEnd();
         glBegin(GL_POLYGON);
         for (k=0;k<5;k++)
            DrawVertex(v+k);
         glEnd();
         break;
      //  Filled star (Quads)
      case 5:
         glBegin(GL_QUADS);
         for (k=0;k<5;k++)
         {
            DrawVertex(V+9);
            DrawVertex(v+(k+4)%5);
            DrawVertex(V+k);
            DrawVertex(v+k);
         }
         glEnd();
         break;
      //  Outline star (tesselation with positive winding rule)
      case 6:
         TesselatedStar(1,GL_LINE,0,GLU_TESS_WINDING_POSITIVE);
         break;
      //  Fill star (tesselation with positive winding rule)
      case 7:
         TesselatedStar(1,GL_FILL,0,GLU_TESS_WINDING_POSITIVE);
         break;
      //  Outline star (tesselation with odd winding rule)
      case 8:
         TesselatedStar(1,GL_LINE,0,GLU_TESS_WINDING_ODD);
         break;
      //  Fill star (tesselation with odd winding rule)
      case 9:
         TesselatedStar(1,GL_FILL,0,GLU_TESS_WINDING_ODD);
         break;
      //  Outline star with hole (tesselation with positive winding rule)
      case 10:
         TesselatedStar(1,GL_LINE,1,GLU_TESS_WINDING_POSITIVE);
         break;
      //  Fill star with hole (tesselation with positive winding rule)
      case 11:
         TesselatedStar(1,GL_FILL,1,GLU_TESS_WINDING_POSITIVE);
         break;
      //  Outline star with hole (tesselation with odd winding rule)
      case 12:
         TesselatedStar(1,GL_LINE,1,GLU_TESS_WINDING_ODD);
         break;
      //  Fill star with hole (tesselation with odd winding rule)
      case 13:
         TesselatedStar(1,GL_FILL,1,GLU_TESS_WINDING_ODD);
         break;
      //  Outline pentagon with hole (tesselation with odd winding rule)
      case 14:
         TesselatedStar(0,GL_LINE,1,GLU_TESS_WINDING_ODD);
         break;
      //  Fill pentagon with hole (tesselation with odd winding rule)
      case 15:
         TesselatedStar(0,GL_FILL,1,GLU_TESS_WINDING_ODD);
         break;
      default:
         break;
   }

   //  Draw outline for large pentagon
   glLineWidth(1);
   glColor3f(1,0,0);
   glBegin(GL_LINE_LOOP);
   for (k=0;k<5;k++)
      glVertex3dv(XYZ[k]);
   glEnd();
   //  Draw outline for small pentagon
   glColor3f(0,0,1);
   glBegin(GL_LINE_LOOP);
   for (k=0;k<5;k++)
      glVertex3dv(xyz[k]);
   glEnd();
   //  Label verteces
   glColor3f(1,1,1);
   for (k=0;k<5;k++)
   {
      glRasterPos3dv(XYZ[k]);
      Print("%d",k);
      glRasterPos3dv(xyz[k]);
      Print("%d",k);
   }

   //  Display parameters
   glColor3f(1,1,1);
   glWindowPos2i(5,5);
   Print("Mode=%d [%s]",mode,text[mode]);
   //  Render the scene and make it visible
   ErrCheck("display");
   glFlush();
   glutSwapBuffers();
}

/*
 *  GLUT calls this routine when a key is pressed
 */
void key(unsigned char ch,int x,int y)
{
   //  Exit on ESC
   if (ch == 27)
      exit(0);
   //  Increase/decrease polygon mode
   else if (ch == '-' && mode>0)
      mode--;
   else if (ch == '+' && mode<MODES-1)
      mode++;
   //  Tell GLUT it is necessary to redisplay the scene
   glutPostRedisplay();
}

/*
 *  GLUT calls this routine when the window is resized
 */
void reshape(int width,int height)
{
   //  Size of world
   double dim=1.1;
   //  Ratio of the width to the height of the window
   double asp = (height>0) ? (double)width/height : 1;
   //  Set the viewport to the entire window
   glViewport(0,0, width,height);
   //  Set projection
   Project(0,asp,dim);
}

/*
 *  Start up GLUT and tell it what to do
 */
int main(int argc,char* argv[])
{
   int k;
   //  Colors
   double R[] = {1.0,1.0,0.0,0.0,0.0,1.0};
   double G[] = {0.0,1.0,1.0,1.0,0.0,0.0};
   double B[] = {0.0,0.0,0.0,1.0,1.0,1.0};
   //  Initialize GLUT
   glutInit(&argc,argv);
   //  Request double buffered, true color window with Z buffering at 600x600
   glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
   glutInitWindowSize(600,600);
   glutCreateWindow("Polygons");
   //  Set callbacks
   glutDisplayFunc(display);
   glutReshapeFunc(reshape);
   glutKeyboardFunc(key);
   //  Initialize star
   for (k=0;k<5;k++)
   {
      //  Large pentagon
      XYZ[k][0] = Sin(72*k);
      XYZ[k][1] = Cos(72*k);
      XYZ[k][2] = 0;
      V[k].x = XYZ[k][0];
      V[k].y = XYZ[k][1];
      V[k].z = XYZ[k][2];
      V[k].r = R[k];
      V[k].g = G[k];
      V[k].b = B[k];
      //  Small pentagon
      xyz[k][0] = 0.381967*Sin(72*k+36);
      xyz[k][1] = 0.381967*Cos(72*k+36);
      xyz[k][2] = 0;
      v[k].x = xyz[k][0];
      v[k].y = xyz[k][1];
      v[k].z = xyz[k][2];
      v[k].r = R[k];
      v[k].g = G[k];
      v[k].b = B[k];
   }
   //  Central diamond
   XYZ[5][0] = -0.2; XYZ[5][1] =    0; XYZ[5][2] = 0;
   XYZ[6][0] =    0; XYZ[6][1] = -0.2; XYZ[6][2] = 0;
   XYZ[7][0] = +0.2; XYZ[7][1] =    0; XYZ[7][2] = 0;
   XYZ[8][0] =    0; XYZ[8][1] = +0.2; XYZ[8][2] = 0;
   XYZ[9][0] =    0; XYZ[9][1] =    0; XYZ[9][2] = 0;
   for (k=5;k<10;k++)
   {
      V[k].x = XYZ[k][0];
      V[k].y = XYZ[k][1];
      V[k].z = XYZ[k][2];
      V[k].r = R[5];
      V[k].g = G[5];
      V[k].b = B[5];
   }
   //  Pass control to GLUT so it can interact with the user
   ErrCheck("init");
   glutMainLoop();
   return 0;
}
