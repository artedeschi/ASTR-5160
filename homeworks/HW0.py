"""
Code by Adam Tedeschi
For ASTR5160 at UWyo 2025
HW0.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

def GenLine(xs,m,b):  #ART Function used for curve_fit fitting
    """
    generates points of a line given x data, slope (m) and y-intercept (b)
    inputs: xs (array of floats); m (float); b (float)
    outputs: points (array of floats)
    """
    return xs*m+b

def GenData(m,b):
    """
    Generates linear data with random scatter given a slope (m) and y-intecept (b)
    inputs: m (float); b (float)
    outputs: xs [x data] (array of floats); ys [y data] (array of floats); yerrs [error in y data] (array of floats
    """
    xs = np.sort(np.random.uniform(0,10,10)) 
    ys = GenLine(xs,m,b)  #ART Must use separate fuction that takes in x so I can use curve_fit
    scatter = np.random.normal(0,0.5,10)
    ys = ys + scatter	  #ART Data points made with random scatter
    yerrs = np.array([0.5]*10)  #ART creates an array of 0.5 with a length of 10
    
    return xs, ys, yerrs
    
def FitData(xs,ys,yerrs):
    """
    Finds a linear best fit in slope (m) and y-intercept (b) given x data, y data, and error
    inputs: xs (array of floats); ys (array of floats); yerrs (array of floats)
    outputs: m2 [best fit slope] (float); b2 [best fit y-intercept] (float)    
    """

    popt,pcov = curve_fit(GenLine,xs,ys,sigma=yerrs)   #ART Finding best model with curve_fit. popt is best-fit parameters, and pcov is the covarience matrix
    
    m2 = popt[0]
    b2 = popt[1]
    
    return m2,b2
    
def PlotData(xs,ys,m,b,m2,b2,yerrs):
    """
    Plots original model, scattered data with error bars, and fitted model
    inputs" xs [generated x data] (array of floats); ys [generated y data] (array of floats);
    m [original slope] (float); b [original y-intercept] (float); m2 [fitted slope] (float);
    b2 [fitted y-intercept] (float); yerrs [error in ys] (array of floats)
    outputs: returns None; displays and saves plot
    """

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
