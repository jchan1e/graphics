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
CLEAN=rm -rf play *.o *.a *.dSYM
endif

all:game

#  Compile
#.c.o:
#	gcc -std=c99 -c $(CFLG) $<
#.cpp.o:
#	g++ -c $(CFLG) $<
game.o:game.cpp structures.h objects.h pixlight.vert pixlight.frag gaussian.frag blender.frag
	g++ -std=c++11 -c $(CFLG) $<

structures.o:structures.cpp structures.h objects.h
	g++ -std=c++11 -c $(CFLG) $<

objects.o: objects.cpp objects.h
	g++ -c $(CFLG) $<
#	gcc -std=c99 -c $(CFLG) $<

#  link
game:game.o structures.o objects.o
	g++ -g -O3 -o play $^ $(LIBS)

#  Clean
clean:
	$(CLEAN)
