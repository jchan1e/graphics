## HW3

### Procedural Texture
Performance experiment using the dynamic Julia set animation from last week: f(z) = z^2 + c  

### Controls

to run the program use
```
./hw3 <mode>
```
where mode is in [a-g]  
  
  
* arrow keys: rotate the camera
* z/x - zoom in and out
* space bar - pause animation
* ,/. - slow down/speed up animation
* n - toggle perspective/ortho projection modes

### Results
Average Framerate tested on macbook pro (GT 750M) and Linux desktop (GTX 960) by running the program for one full cycle of the Julia set parameters and averaging the total frames over the total time.
framerates: (macbook, desktop)  

* a - control; no change from last week's homework
  * (436, 1673)
* b - evaluate Julia set using while loop instead of for loop
  * (427, 1671)
* c - remove secondary condition from for loop; use fixed number of iterations so the compiler can unroll it
  * (219, 565) (my guess is since I still had to put the secondary condition in anyway ,the compiler wasn't able to optimize it properly with the for loop)
* d - calculate Julia parameters in shader rather than passing from CPU
  * (442, 1691)
* e - evaluate Julia set in vertex shader instead of fragment shader
  * (770, 3370) (Julia set correctness reduced to vertex level)
* f - loop using floats, remove all integers
  * (438, 1788)
* g - combine fastest results from all previous tests: use floats instead of ints
  * (440, 1779)
  
  
The only two changes that had any major effect on the framerate were removing a condition from the for loop and adding it in as an if statement (which cut down the laptop's framerate by half and the desktop's framerate by 2/3), and evaluating the Julia set calculations in the vertex shader rather than the fragment shader (which, frankly, made the whole thing look like shiße).  
Of all the other optimizations I tried, the only one that seemed to do much of anything at all was declaring ints as floats, and even that was a small enough difference to be natural statistical variation over 6 trials. Everything else seems to already be optimized by the Nvidia compiler and translated down to effectively the same assembly code.
