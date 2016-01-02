/*
 *  FEM viewer
 *
 *  Demonstrates display lists using a viewer for a finite element model.
 *
 *  Key bindings:
 *  d/D        Toggle display lists on/off
 *  +/-        Cycle through compartments
 *  p/P        Toggle projection mode
 *  l/L        Toggle lighting
 *  k/K        Toggle marked cells
 *  o/O        Toggle offset
 *  k/K        Toggle marked cells
 *  a/A        Toggle axes
 *  z/Z        Decrease/increase Z exaggeration
 *  RightMouse Selection menu
 *  a          Toggle axes
 *  arrows     Change view angle
 *  PgDn/PgUp  Zoom in and out
 *  0          Reset view angle
 *  ESC        Exit
 */
#include "CSCIx229.h"

//  Node
typedef struct
{
   double x,y,z;  // Node location
}  node_t;
//  Element
typedef struct
{
   int n;         // Number of nodes
   int node[6];   // Nodes
   int mat;       // Material
   int mark;      // Mark
   double x,y,z;  // Center of element
}  elem_t;
//  Compartment
typedef struct
{
   char*   name;  // Compartment name
   int     list;  // Display list
   int     Nnode; // Number of nodes
   int     Nelem; // Number of elements
   node_t* node;  // List of nodes
   elem_t* elem;  // List of elements
   double  x,y,z; // Center of compartment
} comp_t;

//  Projection
int    list=1;   // Use display lists
int    mark=0;   // Show only marked elements
int    ortho=1;  // Projection type
int    offset=1; // Offset
int    light=1;  // Lighting
int    th=-20;   // Azimuth
int    ph=+40;   // Elevation
int    zh=30;    // Light azimuth
int    zmag=1;   // Vertical exaggeration
double fov=45;   // Field of View
double asp=1;    // Aspect ratio
double dim=1;    // Dimension
int    axes=1;   // Axes
double X0=0;     // X center
double Y0=0;     // Y center
double Z0=0;     // Z center

//  FEM data
int  mode=-1;    // Compartment to display
int  Ncomp=0;    // Number of compartments
comp_t* comp;    // List of compartments
int  all;        // Display list with all compartments

//  Materials
const int Nmat=7;
int   mat[] = {1,1,1,1,1,1,1};
char* material[] = {"Upper Valley Fill","Lower Valley Fill","Plutonic Rock","Upper Carbonate Rocks","Upper Aquitard","Lower Carbonate Rocks","Basement Rocks"};

/*
 *  Draw 3 node face
 */
static void DrawFace3(int k,int i,int i0,int i1,int i2)
{
   double x0 = comp[k].node[i0].x;  double y0 = comp[k].node[i0].y;  double z0 = comp[k].node[i0].z; // Node 0
   double x1 = comp[k].node[i1].x;  double y1 = comp[k].node[i1].y;  double z1 = comp[k].node[i1].z; // Node 1
   double x2 = comp[k].node[i2].x;  double y2 = comp[k].node[i2].y;  double z2 = comp[k].node[i2].z; // Node 2
   double x  = (x0+x1+x2)/3;        double y  = (y0+y1+y2)/3;        double z  = (z0+z1+z2)/3;       // Center
   // Difference vectors
   double dx0 = x1-x0; double dy0 = y1-y0; double dz0 = z1-z0;
   double dx1 = x2-x1; double dy1 = y2-y1; double dz1 = z2-z1;
   double dx = dy0*dz1 - dy1*dz0;
   double dy = dz0*dx1 - dz1*dx0;
   double dz = dx0*dy1 - dx1*dy0;
   double d = sqrt(dx*dx+dy*dy+dz*dz);
   if (d==0) return;
   // Make sure the vector is outward (positive dot product)
   if (dx*(x-comp[k].elem[i].x)+dy*(y-comp[k].elem[i].y)+dz*(z-comp[k].elem[i].z)<0)
   {
      dx = -dx;
      dy = -dy;
      dz = -dz;
   }
   // Draw triangle
   glBegin(GL_TRIANGLES);
   glNormal3d(dx/d,dy/d,dz/d);
   glVertex3d(x0,y0,z0);
   glVertex3d(x1,y1,z1);
   glVertex3d(x2,y2,z2);
   glEnd();
}

/*
 *  Draw 4 node face
 */
static void DrawFace4(int k,int i,int i0,int i1,int i2,int i3)
{
   double x0 = comp[k].node[i0].x;    double y0 = comp[k].node[i0].y;    double z0 = comp[k].node[i0].z; // Node 0
   double x1 = comp[k].node[i1].x;    double y1 = comp[k].node[i1].y;    double z1 = comp[k].node[i1].z; // Node 1
   double x2 = comp[k].node[i2].x;    double y2 = comp[k].node[i2].y;    double z2 = comp[k].node[i2].z; // Node 2
   double x3 = comp[k].node[i3].x;    double y3 = comp[k].node[i3].y;    double z3 = comp[k].node[i3].z; // Node 3
   double x  = (x0+x1+x2+x3)/4;       double y  = (y0+y1+y2+y3)/4;       double z  = (z0+z1+z2+z3)/4;    // Center
   // Difference vectors
   double dx0 = x1-x0; double dy0 = y1-y0; double dz0 = z1-z0;
   double dx1 = x2-x1; double dy1 = y2-y1; double dz1 = z2-z1;
   // Cross product
   double dx = dy0*dz1 - dy1*dz0;
   double dy = dz0*dx1 - dz1*dx0;
   double dz = dx0*dy1 - dx1*dy0;
   double d = sqrt(dx*dx+dy*dy+dz*dz);
   if (d==0) return;
   // Make sure the vector is outward (positive dot product)
   if (dx*(x-comp[k].elem[i].x)+dy*(y-comp[k].elem[i].y)+dz*(z-comp[k].elem[i].z)<0)
   {
      dx = -dx;
      dy = -dy;
      dz = -dz;
   }
   // Draw quadrangle
   glBegin(GL_QUADS);
   glNormal3d(dx/d,dy/d,dz/d);
   glVertex3d(x0,y0,z0);
   glVertex3d(x1,y1,z1);
   glVertex3d(x2,y2,z2);
   glVertex3d(x3,y3,z3);
   glEnd();
}

/*
 *  Draw compartment
 */
static void DrawCompartment(int k)
{
   int i;
   for (i=0;i<comp[k].Nelem;i++)
   {
      int i0 = comp[k].elem[i].node[0];
      int i1 = comp[k].elem[i].node[1];
      int i2 = comp[k].elem[i].node[2];
      int i3 = comp[k].elem[i].node[3];
      int i4 = comp[k].elem[i].node[4];
      int i5 = comp[k].elem[i].node[5];
      float RGB[][3] = {{1,1,1},{1,0,0},{1,0.5,0},{1,1,0},{0,1,0},{0,0,1},{0,1,1}};
      //  Skip if element is not marked for display
      if (mark && !comp[k].elem[i].mark) continue;
      //  Skip if material is not selected
      if (!mat[comp[k].elem[i].mat]) continue;
      //  Set color from material
      glColor3fv(RGB[comp[k].elem[i].mat]);
      //  Draw faceematerial is not selected
      switch (comp[k].elem[i].n)
      {
      case 4:
         DrawFace3(k,i,i0,i1,i2);
         DrawFace3(k,i,i0,i1,i3);
         DrawFace3(k,i,i0,i2,i3);
         DrawFace3(k,i,i1,i2,i3);
         break;
      case 5:
         DrawFace3(k,i,i0,i1,i2);
         DrawFace3(k,i,i0,i1,i3);
         DrawFace3(k,i,i0,i2,i3);
         DrawFace3(k,i,i1,i2,i3);
         DrawFace3(k,i,i0,i1,i4);
         DrawFace3(k,i,i0,i2,i4);
         DrawFace3(k,i,i1,i2,i4);
         break;
      case 6:
         DrawFace3(k,i,i0,i1,i2);
         DrawFace3(k,i,i3,i4,i5);
         DrawFace4(k,i,i0,i1,i4,i3);
         DrawFace4(k,i,i0,i2,i5,i3);
         DrawFace4(k,i,i1,i2,i5,i4);
         break;
      default:
         fprintf(stderr,"Invalid number of nodes %d\n",comp[k].elem[i].n);
         break;
      }
   }
}

/*
 *  Compile display list
 */
void compile(int delete)
{
   int k;
   //  Compile each compartment into a list
   for (k=0;k<Ncomp;k++)
   {
      if (delete) glDeleteLists(comp[k].list,1);
      comp[k].list = glGenLists(1);
      glNewList(comp[k].list,GL_COMPILE);
      DrawCompartment(k);
      glEndList();
   }
   //  Compile a list of all compartments
   if (delete) glDeleteLists(all,1);
   all = glGenLists(1);
   glNewList(all,GL_COMPILE);
   for (k=0;k<Ncomp;k++)
      glCallList(comp[k].list);
   glEndList();
}

/*
 *  OpenGL (GLUT) calls this routine to display the scene
 */
void display()
{
   int k;
   const double len=3e5;  //  Length of axes
   //  Erase the window and the depth buffer
   glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
   //  Enable Z-buffering in OpenGL
   glEnable(GL_DEPTH_TEST);
   //  Undo previous transformations
   glLoadIdentity();
   //  Orthogonal projection - set world rotation
   if (ortho)
   {
      glRotated(ph,1,0,0);
      glRotated(th,0,1,0);
      glRotated(-90,1,0,0);
   }
   //  Perspective - set eye position
   else
   {
      double Ex = -2*dim*Sin(th)*Cos(ph);
      double Ey = +2*dim        *Sin(ph);
      double Ez = +2*dim*Cos(th)*Cos(ph);
      gluLookAt(Ex,Ey,Ez , 0,0,0 , 0,Cos(ph),0);
      glRotated(-90,1,0,0);
   }

   //  Enable lighting
   if (light)
   {
      float Ambient[]   = {0.3,0.3,0.3,1.0};
      float Diffuse[]   = {0.5,0.5,0.5,1};
      //  Light direction
      float Position[]  = {Sin(zh),Cos(zh),0.33,0};
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

   //  Translate to origin & set vertical exaggeration
   glPushMatrix();
   if (mode<0 || offset)
      glTranslated(-X0,-Y0,-Z0);
   else
      glTranslated(-comp[mode].x,-comp[mode].y,-comp[mode].z);
   glScaled(1,1,zmag);

   //  Draw compartments from display lists
   if (list)
      glCallList(mode<0 ? all : comp[mode].list);
   //  Draw compartments from elements
   else if (mode<0)
      for (k=0;k<Ncomp;k++)
         DrawCompartment(k);
   else
      DrawCompartment(mode);
   glPopMatrix();

   //  Sanity check
   if (glGetError()) Fatal("ERROR: %s\n",gluErrorString(glGetError()));

   //  Draw axes
   glDisable(GL_LIGHTING);
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
   Print("Angle=%d,%d  Zmag=%d Comparment=%s Display=%s",th,ph,zmag,mode<0?"All":comp[mode].name,list?"List":"Immediate");
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
      dim *= 1.05;
   //  PageDown key - decrease dim
   else if (key == GLUT_KEY_PAGE_UP && dim>1)
      dim *= 0.95;
   //  Keep angles to +/-360 degrees
   th %= 360;
   ph %= 360;
   //  Update projection
   Project(ortho?0:fov,asp,dim);
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
   //  Toggle light
   else if (ch == 'l' || ch == 'L')
      light = 1-light;
   //  Toggle mark
   else if (ch == 'k' || ch == 'K')
   {
      mark = 1-mark;
      compile(1);
   }
   //  Toggle display lists
   else if (ch == 'd' || ch == 'D')
      list = 1-list;
   //  Toggle projection
   else if (ch == 'p' || ch == 'P')
      ortho = 1-ortho;
   //  Toggle offset
   else if (ch == 'o' || ch == 'O')
      offset = 1-offset;
   //  Vertical magnification
   else if (ch == 'z')
      zmag++;
   else if (ch == 'Z' && zmag>1)
      zmag--;
   //  Step through modes
   else if (ch == '+')
      mode = (mode+2)%(Ncomp+1)-1;
   else if (ch == '-')
      mode = (mode+Ncomp+1)%(Ncomp+1)-1;
   //  Reproject
   Project(ortho?0:fov,asp,dim);
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
   Project(ortho?0:fov,asp,dim);
}

/*
 *  Material menu callback
 */
void menu(int k)
{
   int i;
   if (k<0 || k>=Nmat)
     for (i=0;i<Nmat;i++)
        mat[i] = (k>0);
   else
     mat[k] = !mat[k];
   compile(1);
   glutPostRedisplay();
}

/*
 *  Load model from file
 */
void LoadModel()
{
   int k;
   FILE* f;
   double Xmin = +1e308;
   double Xmax = -1e308;
   double Ymin = +1e308;
   double Ymax = -1e308;
   double Zmin = +1e308;
   double Zmax = -1e308;

   //  Open BAS file to read number of compartments
   f = fopen("sv_model_ts.bas","r");
   if (!f) Fatal ("Cannot open sv_model_ts.bas\n");
   //  Skip two lines
   for (k=0;k<2;k++)
      while (fgetc(f)!='\n');
   //  Read number of compartments
   if (fscanf(f,"%d",&Ncomp)!=1) Fatal("Cannot read number of compartments\n");
   while (fgetc(f)!='\n');
   //  Allocate node structures
   comp = (comp_t*)malloc(Ncomp*sizeof(comp_t));
   if (!comp) Fatal("Cannot allocate memory for %d node array pointers\n",Ncomp);
   //  Read names
   for (k=0;k<Ncomp;k++)
   {
      int n;
      char buffer[8192];
      if (fscanf(f,"Compartment %s",buffer)!=1)  Fatal("Cannot read name compartment %d\n",k);
      while (fgetc(f)!='\n');
      n = strlen(buffer)+1;
      comp[k].name = malloc(n+1);
      strcpy(comp[k].name,buffer);
   }
   fclose(f);

   //  Open XYZ file to read node locations
   f = fopen("sv_model_ss.xyz","r");
   if (!f) Fatal ("Cannot open sv_model_ss.xyz\n");
   for (k=0;k<Ncomp;k++)
   {
      int    i;
      double Sx,Sy,Sz;
      //  Skip compartment name line
      while (fgetc(f)!='\n');
      //  Read number of nodes and scales
      if (fscanf(f,"%d %lf %lf %lf %d\n",&comp[k].Nnode,&Sx,&Sy,&Sz,&i)!=5) Fatal("Error reading header compartment %d\n",k);
      //  Allocate memory
      comp[k].node = (node_t*)malloc(comp[k].Nnode*sizeof(node_t));
      if (!comp[k].node) Fatal("Cannot allocate memory for %d nodes compartment %d\n",comp[k].Nnode,k);
      //  Read coordinates
      for (i=0;i<comp[k].Nnode;i++)
      {
         int j;
         double x,y,z;
         if (fscanf(f,"%d %lf %lf %lf\n",&j,&x,&y,&z)!=4) Fatal("Error reading XYZ node %d compartment %d\n",i,k);
         x *= Sx;
         y *= Sy;
         z *= Sz;
         comp[k].node[i].x = x;
         comp[k].node[i].y = y;
         comp[k].node[i].z = z;
         if (x<Xmin) Xmin = x; if (x>Xmax) Xmax = x;
         if (y<Ymin) Ymin = y; if (y>Ymax) Ymax = y;
         if (z<Zmin) Zmin = z; if (z>Zmax) Zmax = z;
      }
   }
   //  Calculate center and dim
   X0 = (Xmin+Xmax)/2;
   Y0 = (Ymin+Ymax)/2;
   Z0 = (Zmin+Zmax)/2;
   dim = (Xmax-Xmin + Ymax-Ymin + Zmax-Zmin)/3;
   fclose(f);

   //  Open ELE file to read node locations
   f = fopen("sv_model_ss.ele","r");
   if (!f) Fatal ("Cannot open sv_model_ss.ele\n");
   //  Skip initial line
   while (fgetc(f)!='\n');
   for (k=0;k<Ncomp;k++)
   {
      int i;
      int N=0;
      //  Skip compartment name line
      while (fgetc(f)!='\n');
      //  Read number of elements
      if (fscanf(f,"%d %d %d\n",&comp[k].Nelem,&i,&i)!=3) Fatal("Error reading element header compartment %d\n",k);
      //  Allocate memory
      comp[k].elem = (elem_t*)malloc(comp[k].Nelem*sizeof(elem_t));
      if (!comp[k].elem) Fatal("Cannot allocate memory for %d elements compartment %d\n",comp[k].Nelem,k);
      //  Read coordinates
      comp[k].x = comp[k].y = comp[k].z = 0;
      for (i=0;i<comp[k].Nelem;i++)
      {
         int j,l,m;
         int n=0;   // Number of nodes
         int nn[6]; // List of nodes
         if (fscanf(f,"%d %d %d %d %d %d %d %d %d\n",&j,nn+0,nn+1,nn+2,nn+3,nn+4,nn+5,&m,&l)!=9) Fatal("Error reading element %d compartment %d\n",i,k);
         comp[k].elem[i].mat = m-1;
         if (m<1 || m>Nmat) Fatal("Unknown material %d\n",m);
         comp[k].elem[i].mark = l;
         comp[k].elem[i].x = comp[k].elem[i].y = comp[k].elem[i].z = 0;
         for (j=0;j<6;j++)
         {
            int l = nn[j]-1;
            if (l<0) continue;
            comp[k].elem[i].node[n++] = nn[j]-1;
            comp[k].elem[i].x += comp[k].node[l].x;
            comp[k].elem[i].y += comp[k].node[l].y;
            comp[k].elem[i].z += comp[k].node[l].z;
            N++;
            comp[k].x += comp[k].node[l].x;
            comp[k].y += comp[k].node[l].y;
            comp[k].z += comp[k].node[l].z;
         }
         comp[k].elem[i].n = n;
         comp[k].elem[i].x /= n;
         comp[k].elem[i].y /= n;
         comp[k].elem[i].z /= n;
      }
      comp[k].x /= N;
      comp[k].y /= N;
      comp[k].z /= N;
   }
   fclose(f);

   //  Compile default display list
   compile(0);
   //  Set menu for switching materials
   glutCreateMenu(menu);
   for (k=0;k<Nmat;k++)
      glutAddMenuEntry(material[k],k);
   glutAddMenuEntry("None",-1);
   glutAddMenuEntry("All",Nmat);
   glutAttachMenu(GLUT_RIGHT_BUTTON);
}

/*
 *  Start up GLUT and tell it what to do
 */
int main(int argc,char* argv[])
{
   //  Initialize GLUT
   glutInit(&argc,argv);
   //  Request double buffered, true color window with Z buffering at 600x600
   glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
   glutInitWindowSize(600,600);
   glutCreateWindow("Durbin Spring/Snake Valley Model");
   //  Set callbacks
   glutDisplayFunc(display);
   glutReshapeFunc(reshape);
   glutSpecialFunc(special);
   glutKeyboardFunc(key);
   //  Load model
   LoadModel();
   //  Pass control to GLUT so it can interact with the user
   ErrCheck("init");
   glutMainLoop();
   return 0;
}
