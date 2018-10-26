#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/10/2017
#
# (You are free to use this code in your solution if you wish.)
#
import sys
import re
from math import log
from math import exp
from math import sqrt

MAX_ITERS = 100

# Load data from a file
def read_data(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  data = []
  header = f.readline().strip()
  varnames = p.split(header)
  namehash = {}
  for l in f:
    example = [int(x) for x in p.split(l.strip())]
    x = example[0:-1]
    y = example[-1]
    data.append( (x,y) )
  return (data, varnames)

# sigmoid function
def sig(z):
  d = 1 + exp(-z)
  return 1/d

def dot(K, L):
   if len(K) != len(L):
      print "ERROR in dot()"
      return 0

   return sum(i[0] * i[1] for i in zip(K, L))

# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, lam):
  numvars = len(data[0][0])
  w = [0.0] * numvars
  b = 0.0

  for iter in range (0, 100):
    dw = [0.0] * numvars
    db = 0

    for x,y in data:
      sig_val = sig(-y * (dot(w,x) + b) )

      db -= sig_val * y 

      for i in range (0, numvars):
        dw[i] -= sig_val * y * x[i] 

    #db -= lam*b
    #for i in range (0, numvars):
    #  dw[i] -= lam*w[i]

# compute objective function (logiztic loss plus regularlizer)
# see slides loss function -log  


    # calculate magnitude of weight gradients
    mag = sqrt(sum(dwi**2 for dwi in dw) + db**2 )

    print "iter:", iter, "mag:", mag
    if (mag < 0.001):
       break

    # update weights
    #w = [v[0] - (v[1]*eta) for v in zip(w, dw)]
    for i in range(0, numvars):
      w[i] = w[i] - dw[i]*eta 
    
    b = b - db*eta

  return (w,b)

# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
  if (len(argv) != 5):
    print 'Usage: lr.py <train> <test> <eta> <lambda> <model>'
    sys.exit(2)
  (train, varnames) = read_data(argv[0])
  (test, testvarnames) = read_data(argv[1])
  eta = float(argv[2])
  lam = float(argv[3])
  modelfile = argv[4]

  # Train model
  (w,b) = train_lr(train, eta, lam)

  # Write model file
  f = open(modelfile, "w+")
  f.write('%f\n' % b)
  for i in xrange(len(w)):
    f.write('%s %f\n' % (varnames[i], w[i]))

  # Make predictions, compute accuracy
  correct = 0
  for (x,y) in test:
    prob = sig(-y * ( dot(w, x) + b) ) 
    #print prob 
    if (prob - 0.5) * y > 0:
      correct += 1
  acc = float(correct)/len(test)
  print "Accuracy:{0:.5f} ".format(acc)

if __name__ == "__main__":
  main(sys.argv[1:])
