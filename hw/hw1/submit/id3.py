#!/usr/bin/python
# 
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/20/2017
# You are not obligated to use any of this code, but are free to use
# anything you find helpful when completing your assignment.
#
import sys
import re
# Node class for the decision tree
import node
import math


# SUGGESTED HELPER FUNCTIONS:
# - collect counts for each variable value with each class label
def collect_counts(data):

  output = [item[len(data[0]) - 1] for item in data]
  pos = output.count(1)
  neg = output.count(0)

  return pos, neg

# - compute entropy of a 2-valued (Bernoulli) probability distribution 
def compute_entropy(v1, v2):

  if (v1 == 0 or v2 == 0):
    return 0

  p1 = v1 / float(v1 + v2)
  p2 = v2 / float(v1 + v2)

  e = -p1 * math.log(p1, 2) - p2 * math.log(p2, 2)
  
  return e

# - partition data based on a given variable 
def split_data(data, var):
  l_data = []
  r_data = []

  # right data 1, left data 0
  for i in range(0, len(data)):
    if data[i][var]:  
      r_data.append(data[i])
    else:
      l_data.append(data[i])

  return l_data, r_data 

# - compute information gain for a particular attribute
def compute_gain(data, var):
  
  if (len(data) == 0):
    print "ERROR data == 0"
    return
  
  # compute entropy of the root
  pos, neg = collect_counts(data)
  entropy_s = compute_entropy(pos, neg)

  # compute left and right entropy after splitting
  l_data, r_data = split_data(data, var)

  if (len(r_data) == 0 or len(l_data) == 0): 
    return 0
  
  l_pos, l_neg = collect_counts(l_data)
  entropy_l = compute_entropy(l_pos, l_neg)

  r_pos, r_neg = collect_counts(r_data)
  entropy_r = compute_entropy(r_pos, r_neg)

  # compute information gain
  l_p = len(l_data) / float(len(data))
  r_p = len(r_data) / float(len(data))
  gain = entropy_s - l_p*entropy_l - r_p*entropy_r
  
  return gain

# - find the best variable to split on, according to mutual information
def compute_max_gain(data, varnames):
  max_gain = 0
  max_index = -1

  # compute the max gain
  for i in range(0, len(varnames) - 1):
    gain = compute_gain(data, i)
    #print "var", varnames[i], "gain", gain

    if (gain > max_gain):
       max_gain = gain
       max_index = i

  #print "max_gain computed:", max_gain, "var:", varnames[max_index], "index:", max_index

  return max_index
  

# Load data from a file
def read_data(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  data = []
  header = f.readline().strip()
  varnames = p.split(header)
  namehash = {}
  for l in f:
    data.append([int(x) for x in p.split(l.strip())])
  return (data, varnames)

# Saves the model to a file.  Most of the work here is done in the 
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
  f = open(modelfile, 'w+')
  root.write(f, 0)

# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames, depth):
    #print "Current Depth:", depth

    if len(data) == 0:
      print "BAD SPLIT"
      return
    
    # compute the max gain
    split_index = compute_max_gain(data, varnames)

    # Base cases
    if split_index == -1:
      #print "LEAF CASE"
      #print data
      #print "\n"
      # choose whichever result is more common
      pos, neg = collect_counts(data)
      #print "pos:", pos, "neg:", neg
      if pos > neg:
        return node.Leaf(varnames, 1)
      else:
        return node.Leaf(varnames, 0)

    # split the data at max_index attribute
    l_data, r_data = split_data(data, split_index)
      
    # make new node split
    # left child - buildtree on left split 
    # right child - buildtree on right split 
    var = varnames[split_index]
    #print "SPLIT CASE:", var
    #print "\n"

    #print "***Recursing L_tree***"
    #print l_data
    L_tree = build_tree(l_data, varnames, depth+1)
    #print "***L_tree returned, depth=", depth

    #print "***Recursing R_tree***"
    #print r_data
    R_tree = build_tree(r_data, varnames, depth+1)
    #print "***R_tree returned, depth=", depth

    return node.Split(varnames, split_index, L_tree, R_tree)

# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
  if (len(argv) != 3):
    print 'Usage: id3.py <train> <test> <model>'
    sys.exit(2)
  # "varnames" is a list of names, one for each variable
  # "train" and "test" are lists of examples.  
  # Each example is a list of attribute values, where the last element in
  # the list is the class value.
  (train, varnames) = read_data(argv[0])
  (test, testvarnames) = read_data(argv[1])
  modelfile = argv[2]

  # build_tree is the main function you'll have to implement, along with
  # any helper functions needed.  It should return the root node of the
  # decision tree.
  root = build_tree(train, varnames, 0)

  print_model(root, modelfile)
  correct = 0
  # The position of the class label is the last element in the list.
  yi = len(test[0]) - 1
  for x in test:
    # Classification is done recursively by the node class.
    # This should work as-is.
    pred = root.classify(x)
    if pred == x[yi]:
      correct += 1
  acc = float(correct)/len(test)
  print "Accuracy: ",acc

if __name__ == "__main__":
  main(sys.argv[1:])
