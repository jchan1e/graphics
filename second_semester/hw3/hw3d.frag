//  Per Pixel Lighting shader

varying vec3 View;
varying vec3 Light;
varying vec3 Normal;
varying vec2 TexCoord;
uniform float time;


void main()
{
   //  N is the object normal
   vec3 N = normalize(Normal);
   //  L is the light vector
   vec3 L = normalize(Light);
   //  R is the reflected light vector R = 2(L.N)N - L
   vec3 R = reflect(-L,N);
   //  V is the view vector (eye vector)
   vec3 V = normalize(View);

   //  Diffuse light is cosine of light and normal vectors
   float Id = max(dot(L,N) , 0.0);
   //  Specular is cosine of reflected and view vectors
   float Is = (Id>0.0) ? pow(max(dot(R,V),0.0) , gl_FrontMaterial.shininess) : 0.0;

   //  Julia Set Iterations
   int i;
   vec2 c = vec2(-0.5*(sin(radians(time)) - sin(radians(3.0*time))) + 0.25, 0.5*(cos(radians(time)) - cos(radians(3.0*time))));
   //vec2 c = vec2(-0.75, 0.11);
   vec2 Z = TexCoord*vec2(8.0,4.0) - vec2(4.0, 2.0);
   Z.x = mod(Z.x+2.0, 4.0) - 2.0;
   float zz;
   for (i=0; i<100 && zz<4.0; i++)
   {
      Z = vec2(Z.x*Z.x - Z.y*Z.y, 2.0*Z.x*Z.y) + c;
      zz = Z.x*Z.x + Z.y*Z.y;
   }
   
   vec3 texturecolor;
   if (zz < 4.0)
      texturecolor = vec3(0.1, 0.1, 0.1);
   else
   {
      vec3 colorA = mix(vec3(0.0, 0.0,0.75), vec3(0.0, 2.0, 0.0), float(i)/100.0);
      vec3 colorB = mix(vec3(0.0, 2.0, 0.0), vec3(1.0, 0.0, 0.5), float(i)/100.0);
      texturecolor = mix(colorA, colorB, float(i)/100.0);
   }

   //  Sum color types
   gl_FragColor = vec4(texturecolor, 1.0)
                *(vec4(0.4, 0.4, 0.4, 1.0) //emission
                + gl_FrontLightProduct[0].ambient
                + Id*gl_FrontLightProduct[0].diffuse
                + Is*gl_FrontLightProduct[0].specular);
}
