#!/usr/bin/python

# zlib license:

# Copyright (c) 2019 Maxime Charlebois

# This software is provided 'as-is', without any express or implied
# warranty. In no event will the authors be held liable for any damages
# arising from the use of this software.

# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:

# 1. The origin of this software must not be misrepresented; you must not
#    claim that you wrote the original software. If you use this software
#    in a product, an acknowledgment in the product documentation would be
#    appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
#   misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.

# #########################################
# This code tries to reproduce the figure 3.5 of the book:
# Pattern Recognition and Machine Learning, 
# by Christopher M. Bishop (Springer 2006)
#
# Does it work? Try and see.
# #########################################

import numpy as np
from copy  import deepcopy

from pylab import *
import random
import scipy.optimize as opt


N = 25
M = 24
L = 50
lambda1 = exp(-2.4)
noise_level=0.5

#x axes (x_precise is just for plotting)
x_precise=np.linspace(0.0,1.0, num=20*N)
x=np.linspace(0.0,1.0, num=N)

target_curve=sin(2*np.pi*x)

def Gauss(x,mu,sigma):
  return exp(-(1.0/(2*sigma*sigma))*(x-mu)*(x-mu))

# create a list of data (L different sample with different noise):
data_l = []
for ll in range(L):
  y = deepcopy(target_curve) # the reason I use deepcopy is that often in python, the equal sign is just a reference to the other variable (and it caused many bugs for me in the past)
  for ii in range(N):
    ran = noise_level*(np.random.uniform(0,1) -0.5)
    y[ii] += ran 
  data_l.append(deepcopy(y))


# create M Gaussian basis centered around mu1 and with deviation sigma
subplot(2,2,1)
title('Gaussian basis')
W = []
W_precise = []
sigma1 = 0.03
mu1 = np.linspace(-0.2,1.2,M) # I need to go beyond (0.2 on each side), in order to have an "equivalent basis". In other words, the goal is that the points on the edges are influenced by gaussian on each side (as the points in the center). Is it very useful? did test much this detail.
for jj in range(M):
  w         = Gauss(x        ,mu1[jj],sigma1)
  w_precise = Gauss(x_precise,mu1[jj],sigma1)
  W.append(w)
  W_precise.append(w_precise)
  plot(x_precise,w_precise)  # plot basis functions

# calcale the curve from basis set W1 with weight w1
subplot(2,2,3)
plot(x,target_curve)
def y_D(W1,w1):
  val = np.zeros(len(W1[0]))
  for mm in range(M):
    val += w1[mm]*W1[mm]
  #print y
  return val

# fit each data of the L data samples by minimizing the 
# chi squared plus the regularizatin function (times lambda):
fits = []
averages = 0.0*deepcopy(y)
for ll in range(L):

  def Error_y_D(w1):
    diff = deepcopy(data_l[ll]) - y_D(W,w1) 
    return np.sum(diff*diff) + (lambda1/2.0) * np.dot(w1,w1)

  w0 = np.zeros(M) # start with approximation with every gaussian coeff set to 0

  fit_l = opt.minimize(Error_y_D, w0, method="BFGS")  
  fits.append(fit_l.x)
  print fit_l.fun
  if ll<20:
    plot(x_precise,y_D(W_precise,fit_l.x),'r-',linewidth=0.2)
  averages += (1.0/L)*y_D(W,fits[ll])

#plot the average of all the fits
subplot(2,2,4)
plot(x,averages,'r-')
plot(x,target_curve)

#plot one sample
subplot(2,2,2)
title('One example of data')
plot(x,y,'o')
plot(x,target_curve)

plt.savefig("figure1.pdf")
show()

exit()

