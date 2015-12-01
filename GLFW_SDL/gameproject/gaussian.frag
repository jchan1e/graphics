// Blur (low-pass)
//   1 2 1
//   2 1 2   / 13
//   1 2 1

uniform float dX;
uniform float dY;
uniform sampler2D img;

vec4 sample(float dx,float dy)
{
   return texture2D(img,gl_TexCoord[0].st+vec2(dx,dy));
}

void main()
{
   float one = 1.0/13.0;
   float two = 2.0/13.0;
   vec4 thiscolor = one*sample(-dX,+dY) + two*sample(0.0,+dY) + one*sample(+dX,+dY)
                  + two*sample(-dX,0.0) + one*sample(0.0,0.0) + two*sample(+dX,0.0)
                  + one*sample(-dX,-dY) + two*sample(0.0,-dY) + one*sample(+dX,-dY);
   thiscolor[3] = max(thiscolor[0], thiscolor[1], thiscolor[2]);
   gl_FragColor = thiscolor;
}
