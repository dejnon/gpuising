import os
import argparse
import numpy as np
import subprocess
import csv
import itertools
from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import glob

# catalogue = "_w0-0.1.51_cmean-all_csigma-all_maxt-10-7_l-100_w0grain-1_cgrain-30"
catalogue = "results"

x_name = "cmean"
y_name = "csigma"
z_name = "BondDensity"
col_no = 4 # t-2 mcs-3 magnetization-4 bonddens-5
# W0_RANGE = np.linspace(0, 1.0, 10)
W0_RANGE = [0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000, 1.0000]
w0 = W0_RANGE[int(sys.argv[1])]
# w0 = W0_RANGE[0]

title = "Triangular "+z_name+" W0:%.2f Averages:10 Grain:20x20 MaxT:10^6" % w0

# x = CM_RANGE = np.linspace(0.0, 1.0, 11)
# y = CS_RANGE = np.linspace(0.0, 1.0, 11)

# 50 x 50
# x = CS_RANGE = [0.0200, 0.0400, 0.0600, 0.0800, 0.1000, 0.1200, 0.1400, 0.1600, 0.1800, 0.2000, 0.2200, 0.2400, 0.2600, 0.2800, 0.3000, 0.3200, 0.3400, 0.3600, 0.3800, 0.4000, 0.4200, 0.4400, 0.4600, 0.4800, 0.5000, 0.5200, 0.5400, 0.5600, 0.5800, 0.6000, 0.6200, 0.6400, 0.6600, 0.6800, 0.7000, 0.7200, 0.7400, 0.7600, 0.7800, 0.8000, 0.8200, 0.8400, 0.8600, 0.8800, 0.9000, 0.9200, 0.9400, 0.9600, 0.9800]
# y = CM_RANGE = [0.0200, 0.0400, 0.0600, 0.0800, 0.1000, 0.1200, 0.1400, 0.1600, 0.1800, 0.2000, 0.2200, 0.2400, 0.2600, 0.2800, 0.3000, 0.3200, 0.3400, 0.3600, 0.3800, 0.4000, 0.4200, 0.4400, 0.4600, 0.4800, 0.5000, 0.5200, 0.5400, 0.5600, 0.5800, 0.6000, 0.6200, 0.6400, 0.6600, 0.6800, 0.7000, 0.7200, 0.7400, 0.7600, 0.7800, 0.8000, 0.8200, 0.8400, 0.8600, 0.8800, 0.9000, 0.9200, 0.9400, 0.9600, 0.9800]

# 20 x 20
x = CS_RANGE = [0.0000, 0.0500, 0.1000, 0.1500, 0.2000, 0.2500, 0.3000, 0.3500, 0.4000, 0.4500, 0.5000, 0.5500, 0.6000, 0.6500, 0.7000, 0.7500, 0.8000, 0.8500, 0.9000, 0.9500]
y = CM_RANGE = [0.0000, 0.0500, 0.1000, 0.1500, 0.2000, 0.2500, 0.3000, 0.3500, 0.4000, 0.4500, 0.5000, 0.5500, 0.6000, 0.6500, 0.7000, 0.7500, 0.8000, 0.8500, 0.9000, 0.9500]

def fun(x,y):
  filename = [
    glob.glob(catalogue+'/w0-%.4f_%s-%.4f_%s-%.4f_maxt-100000_l-60' % (w0, x_name, x, y_name, y)),
    glob.glob(catalogue+'/w0-%.4f_%s-%.4f_%s-%.4f_maxt-100000_l-60' % (w0, y_name, y, x_name, x))
  ]
  if filename == [[],[]]:
    print str(x)+" "+str(y)
    print catalogue+'/w0-%.4f_%s-%.4f_%s-%.4f_maxt-100000_l-60' % (w0, x_name, x, y_name, y)
    print x
    raise Exception('No file foud matching criteria')
  filename = filename[0] if filename[0] else filename[1]

  with open(filename[0]) as csv_file:
    reader = csv.reader(csv_file, delimiter=' ', quotechar='"')
    column = []
    for row in reader:
      row = list(filter(("").__ne__, row)) # dubble spaces... ehhh...
      column.append(float(row[col_no]))
    average = sum(column) / len(column)
  return average

X, Y = np.meshgrid(x, y)
zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

fig = plt.figure(figsize=(8, 6), dpi=100)
fig.text(.4, .95, title)
# bigfigure (comment any other)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(20, -130)
p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
cb = fig.colorbar(p, shrink=0.5)
ax.set_zlim3d(0, 1)
ax.set_xlabel(x_name)
ax.set_ylabel(y_name)
ax.set_zlabel(z_name)


# # original
# ax = fig.add_subplot(221, projection='3d')
# ax.view_init(30, 180)
# ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=0)
# ax.set_xlabel(x_name)
# ax.set_ylabel(y_name)
# ax.set_zlabel(z_name)

# # tails colored
# ax = fig.add_subplot(222, projection='3d')
# ax.view_init(20, -130)
# p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# cb = fig.colorbar(p, shrink=0.5)
# ax.set_xlabel(x_name)
# ax.set_ylabel(y_name)
# ax.set_zlabel(z_name)

# # transparent
# ax = fig.add_subplot(223, projection='3d')
# ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.25)
# ax.view_init(50, -120)
# cset = ax.contour(X, Y, Z, zdir='z', offset=1, cmap=cm.coolwarm)
# cset = ax.contour(X, Y, Z, zdir='x', offset=1, cmap=cm.coolwarm)
# cset = ax.contour(X, Y, Z, zdir='y', offset=1, cmap=cm.coolwarm)
# ax.set_xlabel(x_name)
# ax.set_ylabel(y_name)
# ax.set_zlabel(z_name)



# ax = fig.add_subplot(224, projection='3d')
# ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.25)
# cset = ax.contour(X, Y, Z, zdir='z', offset=1, cmap=cm.coolwarm)
# cset = ax.contour(X, Y, Z, zdir='x', offset=1, cmap=cm.coolwarm)
# cset = ax.contour(X, Y, Z, zdir='y', offset=1, cmap=cm.coolwarm)
# ax.view_init(15, 100)
# ax.set_xlabel(x_name)
# ax.set_ylabel(y_name)
# ax.set_zlabel(z_name)

# plt.show()
fig.tight_layout()
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(18.5,10.5)
fig.savefig(title.replace(" ","_").replace(":", "-").replace("^","-")+".png", dpi=100)
# fig.savefig(title.replace(" ","_").replace(":", "-").replace("^","-")+".eps", dpi=100)

