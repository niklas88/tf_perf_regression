Performance Regression TF 1.6 Reproducer
=======================================

Dependencies
------------
This reproducer is designed to use nvidia-docker with the official tensorflow
images. Thus the only dependency is a setup of nvidia-docker.

Getting the data
----------------
Assuming you have changed to the root of this repository, download the
necessary data as follows:

    curl https://daphne.informatik.uni-freiburg.de/downloads/tf_regression_reproducer/reproducer_data.tar | tar -C input -xvf -

Running
-------
To reproduce the performance regression in Tensorflow 1.6 run
the code on TF 1.5 like so

    nvidia-docker build -t reltest 
    nvidia-docker run -it --rm -v $(pwd)/input/:/app/input/ -v $(pwd)/models/:/app/models/ reltest

Each epoch should take about 1-2 seconds. Now edit the Dockerfile and in the
`FROM` line change `1.5.0` to `1.6.0`. and run the above commands again. 

