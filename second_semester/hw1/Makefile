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
CLEAN=rm -rf hw1 *.o *.a *.dSYM
#  Linux/Unix/Solaris
else
CFLG=-g -O3 -Wall $(shell sdl2-config --cflags) -DGL_GLEXT_PROTOTYPES
LIBS=-lGLU -lGL -lm $(shell sdl2-config --libs)
CLEAN=rm -rf hw1 *.o *.a *.dSYM
endif
endif

all:hw1

#  Compile
#.c.o:
#	gcc -std=c99 -c $(CFLG) $<
#.cpp.o:
#	g++ -c $(CFLG) $<
hw1.o:hw1.cpp objects.h hw1.vert hw1.frag pixlight.vert pixlight.frag
	g++ -std=c++11 -c $(CFLG) $<

objects.o: objects.cpp objects.h
	g++ -c $(CFLG) $<
#	gcc -std=c99 -c $(CFLG) $<

#  link
hw1:hw1.o objects.o
	g++ -g -O3 -o hw1 $^ $(LIBS)

#  Clean
clean:
	$(CLEAN)
