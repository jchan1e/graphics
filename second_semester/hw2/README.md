## HW2

### Procedural Texture
Dynamic Julia set animation. f(z) = z^2 + c  
The (x,y) texture coordinates are mapped to the real and imaginary components of the starting value of z. the real and imaginary components of c are varied in time by a pair of sin functions that trace a cardioid path around the edge of the Mandelbrot set main bulb. The program displays all the Julia sets associated with all the values of c along the way.  
I recommend slowing down the animation significantly when the animation is 40-60% through its cycle. A lot happen very quickly during that time.

### Controls

* arrow keys: rotate the camera
* z/x - zoom in and out
* space bar - pause animation
* ,/. - slow down/speed up animation
* n - toggle perspective/ortho projection modes
* m - toggle shaders

