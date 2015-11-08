#  MinGW
ifeq "$(OS)" "Windows_NT"
CFLG=-O3 -Wall
LIBS=-lglut32cu -lglu32 -lopengl32
CLEAN=del *.exe *.o *.a
else
#  OSX
ifeq "$(shell uname)" "Darwin"
CFLG=-O3 -Wall -Wno-deprecated-declarations $(shell sdl2-config --cflags)
LIBS=-framework OpenGL $(shell sdl2-config --libs)
#  Linux/Unix/Solaris
else
CFLG=-O3 -Wall $(shell sdl2-config --cflags)
LIBS=-lGLU -lGL -lm $(shell sdl2-config --libs)
endif
#  OSX/Linux/Unix/Solaris
CLEAN=rm -f play *.o *.a
endif

all:game

#  Compile
#.c.o:
#	gcc -std=c99 -c $(CFLG) $<
#.cpp.o:
#	g++ -c $(CFLG) $<
game.o:game.cpp objects.o structures.o structures.h objects.h
	g++ -c $(CFLG) $<

structures.o:structures.cpp objects.o structures.h objects.h
	g++ -c $(CFLG) $<

objects.o: objects.c objects.h
	gcc -std=c99 -c $(CFLG) $<

#  link
game:game.o structures.o objects.o
	g++ -O3 -o play $^ $(LIBS)

#  Clean
clean:
	$(CLEAN)
