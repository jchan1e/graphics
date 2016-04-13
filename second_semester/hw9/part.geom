//
//  nBody geometry shader
//  Billboards point to quad
//

#version 400 compatibility
layout(points) in;
layout(triangle_strip,max_vertices=15) out;

void billboard(float x,float y)
{
   gl_FrontColor  = x == 0.0 && y == 0.0 ? gl_FrontColorIn[0] : vec4(1.0);
   vec2 delta = vec2(x,y);
   vec4 p = gl_PositionIn[0];
   p.x += dot(delta,gl_ModelViewMatrix[0].xy);
   p.y += dot(delta,gl_ModelViewMatrix[1].xy);
   p.z += dot(delta,gl_ModelViewMatrix[2].xy);
   gl_Position = gl_ModelViewProjectionMatrix*p;
   //  Emit new vertex
   EmitVertex();
}

void main()
{
   float size = 1920.0/50.0;

   billboard(-size,-size);
   billboard(+size,-size);
   billboard( 0.0 , 0.0 );
   billboard(+size,+size);
   billboard(-size,+size);
   billboard( 0.0 , 0.0 );
   billboard(-size,-size);

   EndPrimitive();
}
