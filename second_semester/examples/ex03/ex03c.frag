// Fragment shader for drawing the Mandelbrot set
// Adapted from Orange book

varying vec2  ModelPos;
varying float LightIntensity;

// Maximum number of iterations
const int   MaxIter = 100;
// Beyond this the sequence always diverges
const float MaxR2   = 4.0;

// Colors
const vec3  In   = vec3(0,0,0);
const vec3  Out1 = vec3(1,0,0);
const vec3  Out2 = vec3(0,1,0);

void main()
{
   //  Iterate z = z^2 + c;
   int   iter;
   vec2  z  = vec2(0,0);
   float r2 = 0.0;
   for (iter=0 ; iter<MaxIter && r2<MaxR2 ; iter++)
   {
       z  = vec2(z.x*z.x-z.y*z.y , 2.0*z.x*z.y) + ModelPos;
       r2 = z.x*z.x+z.y*z.y;
   }

   // Base the color on the number of iterations
   vec3 color = (r2<MaxR2) ? In : mix(Out1,Out2,float(iter)/float(MaxIter));
   gl_FragColor = vec4(color*LightIntensity , 1);
}
