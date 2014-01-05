import os
import argparse
import numpy as np
import subprocess
import csv
import sys
import itertools
from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import glob

miu   = 0
sigma = 1
w0    = 2 
rho   = 3
mcs   = 4

grid_size = 60
maxt = 100000

# filename = "_w0-all_cmean-all_csigma-all_maxt-10-5_l-60_w0grain-10_cgrain-10"

filename = sys.argv[1]
output_folder = sys.argv[2]

def output_file(w0, cmean, csigma, maxt, l):
  return 'w0-%.4f_cmean-%.4f_csigma-%.4f_maxt-%d_l-%d' % (float(w0), float(cmean), float(csigma), int(maxt), l)

with open(filename) as csv_file:
  file = csv.reader(csv_file, delimiter=' ', quotechar='"')
  for row in file:
    with open(output_folder + output_file(row[w0], row[miu], row[sigma], maxt, grid_size), "a") as myfile:
      # t-2 mcs-3 magnetization-4 bonddens-5
      myfile.write( "0  0  %d %d %.6f 0\n" % (int(row[mcs]), int(row[mcs]), float(row[rho])) )

