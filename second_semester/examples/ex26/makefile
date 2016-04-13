# Example 26
EXE=ex26

# Main target
all: $(EXE)

#  MinGW
ifeq "$(OS)" "Windows_NT"
CFLG=-O3 -Wall -DUSEGLEW -fopenmp
LIBS=-lglew32 -lglut32cu -lglu32 -lopengl32
CLEAN=del *.exe *.o *.a
else
#  OSX
ifeq "$(shell uname)" "Darwin"
CFLG=-O3 -Wall -Wno-deprecated-declarations -fopenmp
LIBS=-framework GLUT -framework OpenGL
#  Linux/Unix/Solaris
else
CFLG=-O3 -Wall -fopenmp
LIBS=-lglut -lGLU -lGL -lm
endif
#  OSX/Linux/Unix/Solaris
CLEAN=rm -f $(EXE) *.o *.a
endif

# Dependencies
InitCL.o: InitCL.cpp CSCIx239.h InitCL.h
ex26.o: ex26.cpp CSCIx239.h InitCL.h
fatal.o: fatal.c CSCIx239.h
loadtexbmp.o: loadtexbmp.c CSCIx239.h
print.o: print.c CSCIx239.h
project.o: project.c CSCIx239.h
errcheck.o: errcheck.c CSCIx239.h
object.o: object.c CSCIx239.h
fps.o: fps.c CSCIx239.h
elapsed.o: elapsed.c CSCIx239.h
shader.o: shader.c CSCIx239.h
noise.o: noise.c CSCIx239.h

#  Create archive
CSCIx239.a:fatal.o loadtexbmp.o print.o project.o errcheck.o object.o fps.o elapsed.o shader.o noise.o
	ar -rcs $@ $^

# Compile rules
.c.o:
	gcc -c $(CFLG) $<
.cpp.o:
	g++ -c $(CFLG) $<

#  Link
ex26:ex26.o InitCL.o CSCIx239.a
	g++ -fopenmp -O3 -o $@ $^  -lOpenCL  $(LIBS)

#  Clean
clean:
	$(CLEAN)
