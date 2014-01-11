#!/usr/bin/python
import os
import numpy as np
import subprocess

PARAMS = np.linspace(1, 40, 40)
MCS_RANGE = [1000, 10000]

def macros(mcs, params):
  return """#define MAX_MCS     %d
#define W0_START    0.0 
#define W0_END      1.0 
#define W0_SIZE     %d  
#define MIU_START   0.0 
#define MIU_END     1.0 
#define MIU_SIZE    %d  
#define SIGMA_START 0.0 
#define SIGMA_END   1.0 
#define SIGMA_SIZE  %d""" % (mcs, params,params,params)

cuda_files = []
for files in os.listdir("./"):
    if files.endswith(".cu") and files is not "main.cu":
        cuda_files.append(files)

for file in cuda_files:
  for param in PARAMS:
    for mcs in MCS_RANGE:
      contents = open(file).read()
      contents = contents.replace("<<<INSERT>>>", macros(mcs, param))
      output = open("main.cu", "w")
      output.write(contents)
      output.close()
      output = open(file+"_benchmark", "a")
      output.write("MCS: "+str(mcs)+" PARAMS: "+str(param)+"\n")
      output.close()      
      os.system("nvcc -arch=sm_20 main.cu && time ./a.out >> %s" % file+"_benchmark")
      os.system("nvcc -arch=sm_20 main.cu && time ./a.out >> %s" % file+"_benchmark")
      os.system("nvcc -arch=sm_20 main.cu && time ./a.out >> %s" % file+"_benchmark")
      os.system("nvcc -arch=sm_20 main.cu && time ./a.out >> %s" % file+"_benchmark")
      os.system("nvcc -arch=sm_20 main.cu && time ./a.out >> %s" % file+"_benchmark")
