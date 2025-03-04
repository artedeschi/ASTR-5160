import matplotlib.pyplot as plt
import numpy as np
#importing from week4 tasks
from tasks.week4.Week4Tasks import PopulateSky, PlotData2

if __name__ == "__main__":
#Lesson 14 (successfully importing module from another directory.
    pop1,pop2 = PopulateSky()
    
    fig = plt.figure('2 populations')
    ax = fig.add_subplot(111)
    PlotData2(ax,pop1,'Pop 1',firstplot=True)
    PlotData2(ax,pop2,'Pop 2',color='g',marker='*')
    plt.show()
