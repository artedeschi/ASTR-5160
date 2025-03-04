import numpy as np
import matplotlib.pyplot as plt

def get_y(x):
    return x**2 + 3*x + 8
    
def plot_y(xs):
    ys = get_y(xs)
    
    plt.figure()
    plt.plot(xs,ys,'b-',label=r'$x^2+3x+8$')
    plt.xlabel('x-data', fontsize=15)
    plt.ylabel('y-data', fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid()
    plt.legend(fontsize=14)
    plt.show()
    
if __name__ == "__main__":
    xs = np.arange(0,10,.01)
    plot_y(xs)
