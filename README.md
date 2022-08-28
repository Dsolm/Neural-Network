# A simple c++ neural network.
## Building
In order to build this you'll need [cmake](https://cmake.org/). 

Just run:

```
cmake .
make
```

Run the resulting binary with:

```
./neuralnetwork train
```

You can also use a pre-trained network if you run it with:

```
./neuralnetwork file
```

The program will write the network data to output.json automatically at the end of every execution. You can also set the output file with --output file.
