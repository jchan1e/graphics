# idkTD: a tower defense game

### Installation
Requirements  
* C++11
* SDL2
* make

```
make
```

### Controls
* esc - quit
* z/x - zoom in and out
* arrow keys - rotate around the field
* 0 - reset view angles
* q - enter turret placement mode
  * WASD - move turret location
  * ENTER - place turret
* n - spawn the next wave
* m - place demo turrets, spawn the next wave

### Fun Facts

This program uses SDL2 instead of Glut to create the OpenGL context and process user interaction events. The main loop is handled manually in the int main() function.  
  
Physics timesteps are separated from the rendering framerate by running 1 fixed-length animation step at a time, and using a global timing variable to control how often a physics step occurs. Each frame, the previous frame time is added to it, and if it is larger than the physics timestep, the physics timestep is subtracted from it and a physics step occurs until it's less than the physics timestep. This allows multiple render frames to pass without a physics step if the framerate is high, or multiple physics steps per frame if it is low.  
  
Each frame, the scene is rendered once (with a per-pixel lighting shader) and saved to two textures. One texture is run through a Gaussian blur filter multiple times to create a glow effect. The other is passed through an edge detection filter and given an alpha layer based on brightness/value, then alpha blended on top of the blurred texture to create the final image on the screen.
