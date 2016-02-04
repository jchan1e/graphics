## HW3

### Procedural Texture
Performance experiment using the dynamic Julia set animation from last week: f(z) = z^2 + c  

### Controls

to run the program use
```
./hw3 <mode>
```
where mode is in [a-e]  
  
  
* arrow keys: rotate the camera
* z/x - zoom in and out
* space bar - pause animation
* ,/. - slow down/speed up animation
* n - toggle perspective/ortho projection modes

### Results
Average Framerate tested on macbook pro (GT 750M) and Linux desktop (GTX 960) by running the program for one full cycle of the Julia set parameters and counting the total number of frames rendered, then dividing by the running time of the program. (initialization time is not counted)  
framerates: (macbook, desktop)  

* a - control; no change from last week's homework
  * (450, )
* b - evaluate Julia set using while loop instead of for loop
  * (, )
* c - remove secondary condition from for loop; use fixed number of iterations so the compiler can unroll it
  * (, )
* d - calculate Julia parameters in shader rather than passing from CPU
  * (, )
* e - evaluate Julia set in vertex shader instead of fragment shader
  * (, ) (Julia set correctness broken, resolution reduced to vertex level)
* f - loop using floats, remove all integers
  * (, )
* g - combine fastest results from all previous tests
  * (, )
