import matplotlib.pyplot as plt
import numpy as np

def hello():
    x = np.array([0, 1, 2, 3])
    y = np.array([3, 8, 1, 10])       
    plt.plot(x,y)

    return plt
#plot 1:


myplt = plt.subplot(2, 2, 1)
myplt = hello()
myplt = plt.subplot(2, 2, 2)
myplt = hello()
myplt = plt.subplot(2, 2, 3)
myplt = hello()
myplt = plt.subplot(2, 2, 4)
myplt = hello()
myplt.text(0, 0, 'hello\nthere', ha='center', va='center', fontsize = 12)
myplt.show()

#adding text inside the plot
   
plt.show()


exit(0)
#plot 2:
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(2, 2, 2)
plt.plot(hello())


x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(2, 2, 3)
plt.plot(hello())


plt.show()

exit(0)


