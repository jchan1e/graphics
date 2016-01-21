//  Basic fragment shader

uniform float time;
varying vec4 foo;

void main()
{
   gl_FragColor = foo+vec4(sin(time),cos(time),0.0,0.0);
}
