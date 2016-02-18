#version 410 core

//in vec3 View;
//in vec3 Light;
//in vec3 Norm;
//in vec2 TexCoord;
//in vec4 Front_Color;
//uniform vec4 Ambient;
//uniform vec4 Diffuse;
//uniform vec4 Specular;
//uniform float shininess;
//
//uniform float C;
//uniform float D;


void main()
{
   //  N is the object normal
   //vec3 N = normalize(Norm);
   ////  L is the light vector
   //vec3 L = normalize(Light);
   ////  R is the reflected light vector R = 2(L.N)N - L
   //vec3 R = reflect(-L,N);
   ////  V is the view vector (eye vector)
   //vec3 V = normalize(View);

   ////  Diffuse light is cosine of light and normal vectors
   //float Id = max(dot(L,N) , 0.0);
   ////  Specular is cosine of reflected and view vectors
   //float Is = (Id>0.0) ? pow(max(dot(R,V),0.0) , shininess) : 0.0;

   ////  Julia Set Iterations
   //float i;
   //vec2 c = vec2(C, D);
   ////vec2 c = vec2(-0.75, 0.11);
   //vec2 Z = TexCoord*vec2(8.0,4.0) - vec2(4.0, 2.0);
   //Z.x = mod(Z.x+2.0, 4.0) - 2.0;
   //float zz;
   //for (i=0.0; i<100.0 && zz<4.0; i+=1.0)
   //{
   //   Z = vec2(Z.x*Z.x - Z.y*Z.y, 2.0*Z.x*Z.y) + c;
   //   zz = Z.x*Z.x + Z.y*Z.y;
   //}
   //
   //vec3 texturecolor;
   //if (zz < 4.0)
   //   texturecolor = vec3(0.1, 0.1, 0.1);
   //else
   //{
   //   vec3 colorA = mix(vec3(0.0, 0.0,0.75), vec3(0.0, 2.0, 0.0), i/100.0);
   //   vec3 colorB = mix(vec3(0.0, 2.0, 0.0), vec3(1.0, 0.0, 0.5), i/100.0);
   //   texturecolor = mix(colorA, colorB, i/100.0);
   //}

   ////  Sum color types
   //gl_FragColor = vec4(texturecolor, 1.0)
   //             *(vec4(0.2, 0.2, 0.2, 1.0) //emission
   //             +    Front_Color*Ambient
   //             + Id*Front_Color*Diffuse)
   //             + Is*Specular;
   gl_FragColor = (0.0,1.0,1.0);
}
