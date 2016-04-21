CFLG=-g -O3 -Wall
SDLFLG=$(CFLG) $(shell sdl2-config --cflags) -DGL_GLEXT_PROTOTYPES
LIBS=-lGLU -lGL -lm $(shell sdl2-config --libs)
CLEAN=rm -f *run* Rnn *.o *.a
ARCH=-gencode arch=compute_52,code=sm_52

all:run run_cpu play

NN_CPU.o:NN_CPU.c reader_.c
	gcc -std=c99 -c $(CFLG) $<
NN.o:NN.cu reader_.c
	nvcc -ccbin gcc -m64 $(ARCH) -c -g -G -O3 $<
game.o:game.cpp
	g++ -c $(SDLFLG) $<

#  link
run_cpu:NN_CPU.o
	gcc -std=c99 -g -O3 -o $@ $^ -lm
run:NN.o
	nvcc -ccbin gcc -m64 $(ARCH) -g -G -O3 -o $@ $^ -lm
play:game.o
	g++ -g -O3 -o $@ $^ $(LIBS)

#  Clean
clean:
	$(CLEAN)
