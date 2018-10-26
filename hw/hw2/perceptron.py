#!/usr/bin/python
#
# CIS 472/572 - Perceptron Template Code
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
    # Each example is a tuple containing both x (vector) and y (int)
    data.append( (x,y) )
  return (data, varnames)


# Learn weights using the perceptron algorithm
def train_perceptron(data):
    # Initialize weight vector and bias
    numvars = len(data[0][0]) 
    w = [0.0] * numvars
    b = 0.0

    for i in range(1, 100):
      change = 0

      for (x,y) in data:

 	activation = sum ([ai*bi for ai,bi in zip(w,x)] ) + b

	if y*activation <= 0:
          change = 1
	  yxd = [ y * xd for xd in x ]
 	  w = [ai + bi for ai,bi in zip(w, yxd) ]
	  
 	  b = b + y

      if change != 1: 
        break
	
    return (w,b)


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
  # Process command line arguments.
  # (You shouldn't need to change this.)
  if (len(argv) != 3):
    print 'Usage: perceptron.py <train> <test> <model>'
    sys.exit(2)
  (train, varnames) = read_data(argv[0])
  (test, testvarnames) = read_data(argv[1])
  modelfile = argv[2]

  # Train model
  (w,b) = train_perceptron(train)

  # Write model file
  # (You shouldn't need to change this.)
  f = open(modelfile, "w+")
  f.write('%f\n' % b)
  for i in xrange(len(w)):
    f.write('%s %f\n' % (varnames[i], w[i]))

  # Make predictions, compute accuracy
  correct = 0
  for (x,y) in test:
    activation = sum ([ai*bi for ai,bi in zip(w,x)] ) + b
    if activation * y > 0:
      correct += 1
  acc = float(correct)/len(test)
  print "Accuracy:{0:.5f} ".format(acc)

if __name__ == "__main__":
  main(sys.argv[1:])
