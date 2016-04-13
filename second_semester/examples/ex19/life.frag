// Conway's Game of Life
// B3/S23 Rule
//   Born if there are 3 neighbors
//   Survive if there are 2 or 3 neighbors

uniform float dX;
uniform float dY;
uniform sampler2D img;

//  Get cell value (stored in a)
float cell(float dx,float dy)
{
   return texture2D(img,gl_TexCoord[0].st+vec2(dx,dy)).a;
}

//  Evaluate cell
void main()
{
   //  Number of live neighbors
   float Nnb = cell(-dX,+dY) + cell(0.0,+dY) + cell(+dX,+dY)
              +cell(-dX,0.0) +               + cell(+dX,0.0)
              +cell(-dX,-dY) + cell(0.0,-dY) + cell(+dX,-dY);
   //  Decide if the cell is alive on the next cycle
   float live = (Nnb==3.0 || cell(0.0,0.0)==1.0 && Nnb==2.0) ? 1.0 : 0.0;
   //  Set the color to red if live, black if not
   gl_FragColor = vec4(live,0,0,live);
}
