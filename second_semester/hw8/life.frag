uniform float dX;
uniform float dY;
uniform sampler2D img;
uniform sampler2D board;

vec3 sample(float dx, float dy)
{
   vec4 x = texture2D(board,gl_TexCoord[0].st + vec2(dx, dy));
   return min(floor(2.0*x.rgb), vec3(1.0));
}

void main()
{
   //Live neighbors in each color
   vec3 color = + sample(-dX,+dY) + sample(0.0,+dY) + sample(+dX,+dY)
                + sample(-dX,0.0)                   + sample(+dX,0.0)
                + sample(-dX,-dY) + sample(0.0,-dY) + sample(+dX,-dY);

   color.r = (color.r == 3.0 || color.r == 2.0 && sample(0.0,0.0).r == 1.0) ? 1.0 : 0.0;
   color.g = (color.g == 3.0 || color.g == 2.0 && sample(0.0,0.0).g == 1.0) ? 1.0 : 0.0;
   color.b = (color.b == 3.0 || color.b == 2.0 && sample(0.0,0.0).b == 1.0) ? 1.0 : 0.0;

   color += texture2D(img, gl_TexCoord[0].st).rgb;

   gl_FragColor = vec4(color, 1.0);
   //gl_FragColor = vec4(sample(0.0,0.0), 1.0);
}
