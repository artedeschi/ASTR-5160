import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

def GenLine(xs,m,b):  #ART Function used for curve_fit fitting
    return xs*m+b

def GenData(m,b):
    xs = np.sort(np.random.uniform(0,10,10)) 
    ys = GenLine(xs,m,b)  #ART Must use separate fuction that takes in x so I can use curve_fit
    scatter = np.random.normal(0,0.5,10)
    ys = ys + scatter	  #ART Data points made with random scatter
    yerrs = np.array([0.5]*10)
    
    return xs, ys, yerrs
    
def FitData(xs,ys,yerrs):
    popt,pcov = curve_fit(GenLine,xs,ys,sigma=yerrs)   #ART Finding best model with curve_fit. popt is best-fit parameters, and pcov is the covarience matrix
    
    m2 = popt[0]
    b2 = popt[1]
    
    return m2,b2
    
def PlotData(xs,ys,m,b,m2,b2,yerrs):   #ART Plots original model, scattered data with error bars, and fitted model.

    plt.figure('HW0',figsize=(8,8))
    plt.errorbar(xs,ys,yerr=yerrs,color='b', marker = 'o', capsize = 3, linestyle='None', label='Raw Data')
    plt.plot(xs,GenLine(xs,m,b),'r-',label='Original Model | m = '+str(m)+'  b = '+str(b))
    plt.plot(xs,GenLine(xs,m2,b2),'g--', label='Fitted Model | m = '+str(round(m2,3))+'  b = '+str(round(b2,3)))
    plt.xlabel('x data',fontsize=16)
    plt.ylabel('y data',fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid()
    plt.xlim(-0.5,10.5)
    plt.legend(fontsize=13)
    plt.savefig('HW0.png')  #ART Save fig in directory the code is in as HW0.png
    plt.show()
    
    
    
    
if __name__ == "__main__":  #ART Only runs this part if run from command line

    args = sys.argv   #ART m and b are pulled in as commandline arguments
    try:
        m = float(args[1])
        b = float(args[2])
    except:
        print('Invalid imputs for \'m\' and \'b\'. To properly run, type \"python HW0.py [m] [b]\"')   #ART throw error and explain how to run correctly if wrong commandline format
        sys.exit()
        
        
    xs,ys,yerrs = GenData(m,b)
    modys = GenLine(xs,m,b) #ART Original exact model based on user input
    m2,b2 = FitData(xs,ys,yerrs)
    PlotData(xs,ys,m,b,m2,b2,yerrs)
