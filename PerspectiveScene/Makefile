#  MinGW
ifeq "$(OS)" "Windows_NT"
CFLG=-O3 -Wall
LIBS=-lglut32cu -lglu32 -lopengl32
CLEAN=del *.exe *.o *.a
else
#  OSX
ifeq "$(shell uname)" "Darwin"
CFLG=-O3 -Wall -Wno-deprecated-declarations
LIBS=-framework GLUT -framework OpenGL
#  Linux/Unix/Solaris
else
CFLG=-O3 -Wall
LIBS=-lglut -lGLU -lGL -lm
endif
#  OSX/Linux/Unix/Solaris
CLEAN=rm -f scene *.o *.a hw*
endif

all:scene

#  Compile
.c.o:
	gcc -std=c99  -c $(CFLG) $<
.cpp.o:
	g++ -c $(CFLG) $<

#  link
scene:scene.o 
	gcc -O3 -o hw4 $^   $(LIBS)

#  Clean
clean:
	$(CLEAN)
