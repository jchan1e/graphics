//  Set the fragment color

uniform vec3 dim;

void main()
{
   gl_FragColor = vec4(gl_FragCoord.xyz/dim,1);
}
