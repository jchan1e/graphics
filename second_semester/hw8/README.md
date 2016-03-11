# HW8: Using Textures for Data Storage

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
* z/x - zoom
* arrow keys - rotate around the field
* 0 - reset view angles
* enter - spawn a random object with a life span of 1-4 seconds

### Fun Facts

Each frame, the scene is rendered once (with a per-pixel lighting shader) and saved to two textures. One texture is rendered through a Gaussian blur filter multiple times to create a glow effect. The other is passed through an edge detection filter and given an alpha component based on brightness/value, then alpha blended on top of the blurred texture to create the final image on the screen.  

For persistent data, I've saved the previous frame to a third texture. Every frame, the current image is combined with the previous frame, and a Conway's Game of Life shader is run on all three color components. This allows the game of life to progress between frames even when the relevant part of the image isn't being modified by the objects being drawn.  

The most interesting part of this for me is watching how the three color components interact in the game of life. Since the scene starts with only the white light source ball rotating around, all the life particles are white initially, but as soon as you add another colored object, for instance a red one, they'll interact with the white particles and split them into red and cyan patterns, which will have no further interactions with each other.
