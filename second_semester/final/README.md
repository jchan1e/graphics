## HW10: Neural Networks

### Build
to build
```
make
```

to build just the CPU implementation
```
make run_cpu
```

to build just the GPU implementation
```
make srun
```

### Run
to run the CPU implementation
```
./run_cpu trainingimages traininglabels testimages testlabels num_iterations alpha
```
for example, with the data in this folder, a learning rate (alpha) of 0.01, and 32 training iterations
```
./run_cpu train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte 32 0.01
```

to run the GPU implementation
```
./run trainingimages traininglabels testimages testlabels num_iterations alpha
```
```
./run train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte 32 0.01
```

### Description
This assignment implements a simple neural network with three layers: Input, Hidden state, and Output. The data comes from the MNIST fdataset, which contains 70000 samples of handwritten digits 0-9, all as 28x28 greyscale images with labels. These are unrolled into 784-length vectors and fed to the neural network en masse. The neural network propagates the information forward through the net using weight matrices that control the flow of information between layers. The net then evaluates its error based on how close its output vectors came to the data's labels, then propagates corrections back through the weight matrices. At the end of the program, the network tests its fully trained weight matrices by evaluating a distinct test set of data, and again measuring the differences between its predictions and the labels in the data.  

Unfortunately I didn't have enough time to make this program visually interesting at all yet, so the only outputs it gives are training time and testing accuracy.
