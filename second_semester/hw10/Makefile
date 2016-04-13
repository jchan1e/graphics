CFLG=-g -O3 -Wall
LIBS=-lsndfile -lm
CLEAN=rm -f *run* Rnn *.o *.a
ARCH=-gencode arch=compute_52,code=sm_52

all:run run_cpu

NN_CPU.o:NN_CPU.c reader_.c
	gcc -std=c99 -c $(CFLG) $<
NN.o:NN.cu reader_.c
	nvcc -ccbin gcc -m64 $(ARCH) -c -g -G -O3 $<

#  link
run_cpu:NN_CPU.o
	gcc -std=c99 -g -O3 -o $@ $^ $(LIBS)
run:NN.o
	nvcc -ccbin gcc -m64 $(ARCH) -g -G -O3 -o $@ $^ $(LIBS)

#  Clean
clean:
	$(CLEAN)
