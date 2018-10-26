import matplotlib.pyplot as plt

plt.plot([-2, -1, 0.5, 1, 2], [-1, -1, -1, 1, 1], 'ro')
plt.plot([0.5, 1], [-1, 1] )
plt.plot([-2, 2], [-1,  1] )



plt.plot([0], [0], 'bo')

plt.axis([-3, 3, -3, 3])
plt.show()
