#!/usr/bin/env bash
#original/with first 3 layers optimized/with one more constrain
python my_analyzer.py mnist_nets/mnist_relu_6_50.txt 0.01
#python my_analyzer.py mnist_nets/mnist_relu_6_100.txt 0.01
#python my_analyzer.py mnist_nets/mnist_relu_6_200.txt 0.01
#python my_analyzer.py mnist_nets/mnist_relu_9_200.txt 0.01
#python my_analyzer.py mnist_nets/mnist_relu_4_1024.txt 0.001