//  Per Pixel Lighting shader

varying vec3 View;
varying vec3 Light;
varying vec3 Normal;
varying vec2 TexCoord;
varying vec4 Colour;
uniform float C;
uniform float D;

void main()
{
   //  Vertex location in modelview coordinates
   vec4 P = gl_ModelViewMatrix * gl_Vertex;
   //  Light position
   Light  = gl_LightSource[0].position.xyz - P.xyz;
   //  Normal
   Normal = gl_NormalMatrix * gl_Normal;
   //  Eye position
   View  = -P.xyz;
   //  Set vertex position
   gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
   //  Set texture coordinate
   TexCoord = gl_MultiTexCoord0.xy;

   int i;
   vec2 c = vec2(C, D);
   //vec2 c = vec2(-0.75, 0.11);
   vec2 Z = TexCoord*vec2(8.0,4.0) - vec2(4.0,2.0);
   Z.x = mod(Z.x+2.0, 4.0) - 2.0;
   float zz;
   for (i=0; i<100 && zz<4.0; i++)
   {
      Z = vec2(Z.x*Z.x - Z.y*Z.y, 2.0*Z.x*Z.y) + c;
      zz = Z.x*Z.x + Z.y*Z.y;
   }

   vec3 thisColor;
   if (zz < 4.0)
      thisColor = vec3(0.1,0.1,0.1);
   else
   {
      vec3 colorA = mix(vec3(0.0, 0.0,0.75), vec3(0.0, 2.0, 0.0), float(i)/100.0);
      vec3 colorB = mix(vec3(0.0, 2.0, 0.0), vec3(1.0, 0.0, 0.5), float(i)/100.0);
      thisColor = mix(colorA, colorB, float(i)/100.0);
   }
   Colour = vec4(thisColor, 1.0);
}
