x = [-2, -1, 0.5, 1, 2]
y = [-1, -1, -1, 1, 1]

C = 0.05

# i remove nothing

w = 4.0
b = -3.0

svm = sum( max(0, 1 - yi * (w * xi + b)) for xi, yi in zip(x, y) )
svm = (w*w) / 2 + C * svm

print "i:", svm
print "-b/w = ", -b / w

# ii remove x3

w = 1.0
b = 0.0

svm = sum( max(0, 1 - yi * (w * xi + b)) for xi, yi in zip(x, y) )
svm = (w*w) / 2 + C * svm

print "ii:", svm
print "-b/w = ", -b/w

# repeat for iii - vi, and all values of C

# iii remove x4

w = 1.33
b = -1.67

svm = sum( max(0, 1 - yi * (w * xi + b)) for xi, yi in zip(x, y) )
svm = (w*w) / 2 + C * svm

print "iii:", svm
print "-b/w = ", -b/w

# iv remove x2, x3

w = 0.67
b = 0.33

svm = sum( max(0, 1 - yi * (w * xi + b)) for xi, yi in zip(x, y) )
svm = (w*w) / 2 + C * svm

print "iv:", svm
print "-b/w = ", -b/w

# v remove x3, x4

w = 0.67
b = -0.33

svm = sum( max(0, 1 - yi * (w * xi + b)) for xi, yi in zip(x, y) )
svm = (w*w) / 2 + C * svm

print "v:", svm
print "-b/w = ", -b/w


# vi remove x2, x3, x4

w = 0.5
b = 0.0

svm = sum( max(0, 1 - yi * (w * xi + b)) for xi, yi in zip(x, y) )
svm = (w*w) / 2 + C * svm

print "vi:", svm
print "-b/w = ", -b/w
