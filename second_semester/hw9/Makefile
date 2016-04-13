#  MinGW
ifeq "$(OS)" "Windows_NT"
CFLG= -O3 -Wall
LIBS=-lglut32cu -lglu32 -lopengl32
CLEAN=del *.exe *.o *.a
else
#  OSX
ifeq "$(shell uname)" "Darwin"
CFLG=-g -O3 -Wall -Wno-deprecated-declarations $(shell sdl2-config --cflags)
LIBS=-framework OpenGL $(shell sdl2-config --libs)
#  Linux/Unix/Solaris
else
CFLG=-g -O3 -Wall $(shell sdl2-config --cflags) -DGL_GLEXT_PROTOTYPES
LIBS=-lGLU -lGL -lm $(shell sdl2-config --libs)
endif
#  OSX/Linux/Unix/Solaris
CLEAN=rm -rf play *.o *.a *.ptx *.cubin *.dSYM
endif

ARCH=-gencode arch=compute_52,code=sm_52

all:game

#  Compile
#.c.o:
#	gcc -std=c99 -c $(CFLG) $<
#.cpp.o:
#	g++ -c $(CFLG) $<
game.o:game.cu structures.h objects.h pixlight.vert pixlight.frag gaussian.frag blender.frag
	nvcc -ccbin g++ -m64 $(ARCH) -std=c++11 -c $< -Xcompiler "$(CFLG)"

structures.o:structures.cpp structures.h objects.h
	g++ -std=c++11 -c $(CFLG) $<

objects.o:objects.cpp objects.h
	g++ -c $(CFLG) $<
#	gcc -std=c99 -c $(CFLG) $<

#part.o:part.cu
#	nvcc -c $(ARCH) $< -o part_.o
#	nvcc $(ARCH) -dlink part_.o -o $@

#  link
game:game.o structures.o objects.o #part.o
	nvcc -ccbin g++ $(ARCH) -m64 -g -O3 -o play $^ -lGLU -lGL -lm -L/usr/local/lib -lSDL2 -lpthread

#  Clean
clean:
	$(CLEAN)
