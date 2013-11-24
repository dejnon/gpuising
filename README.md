GPU 1D Ising simulation
========================

Try to put the repo inside CUDA samples folder like:
```
nvcc gpuising.cu -arch=sm_20
```
Run with:
```
./a.out
```


You can also try to run test file with
```
nvcc curand.cu -arch=sm_20
```
Run with:
```
./a.out
```
Which should produce a list of random numbers.
Note that your GPGPU should be cuRAND compatible and accept pritf's from kernel code.
