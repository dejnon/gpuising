#!/usr/bin/env python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv
from math import *
from scipy.optimize import curve_fit
import bigfloat

# First illustrate basic pyplot interface, using defaults where possible.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

SIZES = ["1000", "10000"]
# MODES = ["memorypool", "staticmemory", "threadperspin", "threadperspin_reduction", "threadperspin_updateflag"]
# MODES = ["memorypool", "staticmemory"]
MODES = ["threadperspin", "threadperspin_reduction", "threadperspin_updateflag"]

COLORS = ["b","g","r","c","m"]
LINES = ['-', '--', ':']
DOTS = ["^", "o", "v", "8", "s", "p", "*", "x", "d"]

def fitFunc(x, a, b, c):
    return a*np.exp(b*x) + c

def pick(i, arr):
  return arr[i%len(arr)]

averages = 5

X = {}
max_size = 39
for size in SIZES:
  for mode in MODES:
    X[size+"_"+mode+"_counter"] = [0]*max_size
    X[size+"_"+mode+"_min"] = [9999999999999.0]*max_size
    X[size+"_"+mode+"_max"] = [0.0]*max_size
    X[size+"_"+mode+"_avg"] = [0.0]*max_size
    with open(mode+"_"+size+".csv") as csv_file:
      reader = csv.reader(csv_file, delimiter='\t', quotechar='"')
      for row in reader:
        if mode == "memorypool" or mode == "staticmemory":
          threads = int(row[2])-1
        else:
          threads = int(round(pow(float(row[3]), 1.0/3.0)))-1
        time = float(row[4])
        if threads < max_size:
          if X[size+"_"+mode+"_min"][threads] > time:
             X[size+"_"+mode+"_min"][threads] = time
          if X[size+"_"+mode+"_max"][threads] < time:
             X[size+"_"+mode+"_max"][threads] = time
          X[size+"_"+mode+"_avg"][threads] += time
          X[size+"_"+mode+"_counter"][threads] += 1
          if X[size+"_"+mode+"_counter"][threads] == averages:
              X[size+"_"+mode+"_avg"][threads] = X[size+"_"+mode+"_avg"][threads] / float(averages)
              X[size+"_"+mode+"_counter"][threads]


x_axis = np.linspace(1,max_size ** 3,max_size)

i=0
for size in SIZES:
  for mode in MODES:
    min = np.array(X[size+"_"+mode+"_min"])
    max = np.array(X[size+"_"+mode+"_max"])
    avg = np.array(X[size+"_"+mode+"_avg"])
    bottom = (avg-min)
    top = (max-avg)
    # fitParams, fitCovariances = curve_fit(fitFunc, x_axis, avg)
    fitParams = np.polyfit(x_axis, avg, 4)
    polynomial = np.poly1d(fitParams)
    plt.plot(x_axis ,polynomial(x_axis), 
             color=pick(i, COLORS), 
             linestyle=pick(i, LINES), 
             marker="None")
    # plt.plot(x_axis ,fitFunc(x_axis, fitParams[0], fitParams[1], fitParams[2]), 
    #          color=pick(i, COLORS), 
    #          linestyle=pick(i, LINES), 
    #          marker=pick(i, DOTS), # marker="None", 
    #          label=mode+" "+size)
    plt.errorbar(x_axis ,avg, yerr=[bottom, top], label=mode+" "+size, marker=pick(i, DOTS), color=pick(i, COLORS), linestyle="None")
    i+=1

plt.ylabel('Time (s)', fontsize = 16)
plt.xlabel('Concurent simulations', fontsize = 16)

max_yticks = 20
yloc = plt.MaxNLocator(max_yticks)
ax.yaxis.set_major_locator(yloc)

max_xticks = 20
xloc = plt.MaxNLocator(max_xticks)
ax.xaxis.set_major_locator(xloc)

# plt.xlim(0,4.1)
# plt.ylim(-1,100)
# plt.title("Static vs dynamic memory allocation")
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=8, ncol=4, mode="expand", borderaxespad=0.)
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid()
plt.legend(loc='upper left')
# plt.show()

fig.tight_layout()
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(18.5,10.5)
fig.savefig("./1000and10000tperspin.png", dpi=150)
