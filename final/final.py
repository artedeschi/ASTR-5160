"""
Code by Adam Tedeschi
For ASTR5160 at UWyo 2025
final.py
"""


import numpy as np
import os
import matplotlib.pyplot as plt
import emcee
import corner
import argparse
from astropy.io import fits
from astropy.table import Table


def LinModel(xs,theta):
    """
    NAME: LinModel
 
    PURPOSE: Returns y values based on a linear model y = mx + b

    INPUTS:
    
     xs - 1D array-like of floats or single float.
     theta -  1D array-like of floats containing 2 entries: m (slope) and b (y-intercept)
     
    OUTPUTS: 
    
     ys - 1D array-like with the same size of 'xs' containing the results of
     m * xs + b
    
    COMMENTS: None.
    """

    m,b = theta
    ys = m*xs+b
    return ys
    
def QuadModel(xs,theta):
    """
    NAME: QuadModel
 
    PURPOSE: Returns y values based on a quadratic model y = a2*x^2 + a1*x + a0

    INPUTS:
    
     xs - 1D array-like of floats or single float.
     theta -  1D array-like of floats containing 3 entries: a2, a1 and a0
     
    OUTPUTS: 
    
     ys - 1D array-like with the same size of 'xs' containing the results of
     a2*xs**2 + a1*x + a0
    
    COMMENTS: None.
    """
    
    a2,a1,a0 = theta
    ys = a2*(xs**2) + a1*xs + a0
    return ys
    
def log_prior(theta):
    """
    NAME: log_prior
 
    PURPOSE: Sets proper priors on the parameter space.
    Returns either zero of -np.inf depending on constraints set for
    each parameter of each model.

    INPUTS:
    
     theta - 1D array-like containing current values of the parameters being fit by each model
    
    OUTPUTS: 
    
     returns zero if parameters are within bounds, returns -np.inf of parameters are outside bounds
    
    COMMENTS: Built to be used in tandem with log_Probability().
    """
    
    #ART Prior bounds were chosen through trial and error 
    if len(theta) == 2:
        m,b = theta
        if -8 < m < 0 and 0 < b < 8:
            return 0.0
    if len(theta) == 3:
        a2,a1,a0 = theta
        if a2 > -2 and -8 < a1 < 3 and -5 < a0 < 20:
            return 0.0
    return -np.inf
    
    
def log_likelihood(theta, x, y, yerr):
    """
    NAME: likelihood
 
    PURPOSE: Calculates log likelihood of the given parameters (theta) of
    a given model to fit the given data (x, y, and yerr). This calculation
    is made regardless of the priors.

    INPUTS:
    
     theta - 1D array-like containing current values of the parameters being fit by each model
     x - 1D numpy.ndarray of x values for the data to fit to
     y - 1D numpy.ndarray of y values for the data to fit to
     yerr - 1D numpy.ndarray of the 1 sigma uncertainties of 'y'
    
    OUTPUTS: 
    
     returns The log-likelihood of the model fitting the data. This equals the log-probability
     if the calulated log-prior is zero.
    
    COMMENTS: Built to be used in tandem with log_Probability(). x, y, and yerr must be the same length
    """
    
    #ART automatically chooses model based on length of parameters. If ndim is not 2 or 3, code will crash, but
    #that should not possible with how the code was made here.
    if len(theta) == 2:
        model = LinModel(x,theta)
    elif len(theta) == 3:
        model = QuadModel(x,theta)
    else:
        print('Invalid number of input parameters . Terminating script.')
        raise IndexError('Number of parameters is not 2 or 3')
    sigma2 = yerr**2 + model**2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
    
    
    

def log_probability(theta, x, y, yerr):
    """
    NAME: log_probability
    
    PURPOSE: Adds together results from log_likelihood() and log_prior()

    INPUTS:
    
     theta - 1D array-like containing current values of the parameters being fit by each model
     x - 1D numpy.ndarray of x values for the data to fit to
     y - 1D numpy.ndarray of y values for the data to fit to
     yerr - 1D numpy.ndarray of the 1 sigma uncertainties of 'y'
    
    OUTPUTS: 
    
     returns The log-probability of the model fitting the data. This equals the log-likelihood
     if the calulated log-prior is zero. returns -np.inf if log-prior is -np.inf.
    
    COMMENTS: Built to be used in tandem with run_MCMC(). x, y, and yerr must be the same length
    """    
    

    lp = log_prior(theta)
    #if not np.isfinite(lp):
    #    return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)



    
def run_MCMC(data, init_params, nwalkers, nsteps):
    """
    NAME: run_MCMC
    
    PURPOSE: fits linear or quadratic model onto 'data' given some inital parameters
    using the emcee library's MCMC algorithm 

    INPUTS:
    
     data - astropy.tables.Table or structured 2D numpy.ndarray containing keys for x data ('x'),
     y data ('y'), and the 1 sigma uncertainties in the y data ('yerr')
     init_params - 1D array-like containing inital guesses of the parameters being fit by each model
     nwalkers - integer representing number of walkers to use in the MCMC fitting
     nsteps - integer representing the number of steps each walker takes in the MCMC fitting


    
    OUTPUTS: 
    
     sampler - emcee.EnsembleSampler object containing all information about the MCMC run in its attributes
     and functions.
    
    COMMENTS: None
    """    

  #nwalkers = len(data)
    ndim = len(init_params)
    initpos = init_params + 1e-4*np.random.randn(nwalkers,ndim)
    
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(data['x'], data['y'], data['yerr'])
    )
    sampler.run_mcmc(initpos, nsteps, progress=True);
    
    samples = sampler.get_chain() #shape = (nsteps,nwalkers,ndim)
    flat_samples = sampler.get_chain(flat=True) #shape = (nsteps*nwalkers,ndim)
    #best_fits = np.median(flat_samples[burnin:],axis=0)
    

    #ART automatically chooses model based on length of parameters. 1st check already made
    # in log_likelihood() for correct ndim, so no need to check to exit code if that happens here.
    if ndim == 2:
        labels = ["m", "b"]
        title = 'Linear'
    elif ndim == 3:
        labels = ["a2","a1","a0"]
        title='Quadratic'

#    ART: This plotted the sequence of walkers through all steps. Removed since that wasn't part of the directions
#    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
#   
#    for i in range(ndim):
#        ax = axes[i]
#        ax.plot(samples[:, :, i], "k", alpha=0.3)
#        ax.set_xlim(0, len(samples))
#        ax.set_ylabel(labels[i])
#        ax.yaxis.set_label_coords(-0.1, 0.5)
    

    fig = corner.corner(
    flat_samples, labels=labels,
    );

    fig.suptitle(title)
    return sampler
    
    
    
def plot_fits(data,params):
    """
    NAME: plot_fits
    
    PURPOSE: Plots original data and fitted model on top using the best fit parameters

    INPUTS:
    
     data - astropy.tables.Table or structured 2D numpy.ndarray containing keys for x data ('x'),
     y data ('y'), and the 1 sigma uncertainties in the y data ('yerr')
     params - 1D array-like containing the best fit values of the parameters as fit by each model
    
    OUTPUTS: 
    
     None
    
    COMMENTS: None
    """       
    
    #ART reading in raw data
    xs = data['x']
    ys = data['y']
    yerrs = data['yerr']
    
    #ART create mock x and y data based on the fitted model
    fitxs = np.linspace(xs[0]-5,xs[-1]+5,1000)
    #ART automatically chooses model based on length of parameters. 1st check already made
    # in log_likelihood() for correct ndim, so no need to check to exit code if that happens here.+
    if len(params) == 2:
        fitys = LinModel(fitxs,params)
        model = 'Linear'
    elif len(params) == 3:
        fitys = QuadModel(fitxs,params)
        model = 'Quadratic'

    plt.figure(model)
    plt.errorbar(xs,ys,yerr=yerrs,fmt='o',color='b',capsize=3,label='Raw Data')
    plt.plot(fitxs,fitys,'r--',label=model+' Best Fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    plt.title(model)
    
    
def output_fits(sampler, burnin = 0):
    """
    NAME: run_MCMC
    
    PURPOSE: fits linear or quadratic model onto 'data' given some inital parameters
    using the emcee library's MCMC algorithm 

    INPUTS:
    
     sampler - emcee.EnsembleSampler object containing all information about the MCMC run in its attributes
     and functions.
     burnin - interger representing the number of inital steps to ignore in the MCMC fitting. burnin must
     be less than nsteps from run_MCMC(). | Default = 0

    
    OUTPUTS: 
    
     best_params - 1D array-like containing the best fit values of the parameters as fit by each model
     nerrs - 1D array-like containing the negative extent of the confidence interval around each best-fitted parameter
     perr -  1D array-like containing the positive extent of the confidence interval around each best-fitted parameter
    
    COMMENTS: Prints return values to terminal in a readable form
    """    
    
    #ART Gets values on the chain for all walkers combined into one series per parameter.
    #discards first 'burnin' amount of steps
    flat_samples = sampler.get_chain(flat=True, discard=burnin)
    
    #ART automatically chooses model based on length of parameters. If ndim is not 2 or 3, code will crash, but
    #that should not possible with how the code was made here.
    if flat_samples.shape[1] == 2:
        labs = ['m','b']
        model = 'linear'
    elif flat_samples.shape[1] == 3:
        labs = ['a2','a1','a0']
        model = 'quadratic'

    #ART Find best fit parameters and uncertainties using 16%, 50%, and 84% percentile ranges of the data
    best_params = []
    perrs = []
    nerrs = []
    print('Model: '+model)
    for i in range(flat_samples.shape[1]):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        #ART setting up format for printing fit values to terminal
        txt = "{3} = {0:.3f} (-{1:.3f} / +{2:.3f})"
        txt = txt.format(mcmc[1], q[0], q[1], labs[i])
        print(txt)
        #ART saving best-fit values
        best_params.append(mcmc[1])
        nerrs.append(q[0])
        perrs.append(q[1])
    #ART add new line for visual appeal
    print('\n')
    return best_params, nerrs, perrs
    
    
if __name__ == "__main__":


    #ART argparser includes options to set number of walkers, steps, and burnin from the commandline
    parser = argparse.ArgumentParser(description=
    """Uses an MCMC algorithm from emcee to fit data to both a linear or quadratic model. After
    fitting the data, this program will plot the best fit model on the data and the posterior
    probability distributions for the paramets as a corner plot""")
    parser.add_argument("-w", '--walkers', default=10, help="Set number of walkers | default = 10")
    parser.add_argument('-s', '--steps', default=5000, help="Set number of steps | default = 5000")
    parser.add_argument('-b', '--burnin', default=1000, help="Set length of burnin | default = 1000")

    args = parser.parse_args()
    
    nwalkers = int(args.walkers)
    nsteps = int(args.steps)
    burnin = int(args.burnin)

    #ART read in data file
    datfile = os.path.join(os.getenv("ASTR5160"), 'final','dataxy.fits')
    data = Table(fits.open(datfile)[1].data)

    #init_m = -3   inti_b = 4
    init_linparams = [-3,4]
    
    #ART running MCMC algorithm here
    lin_sampler = run_MCMC(data,init_linparams, nwalkers, nsteps)
    
    #init_a2 = 0.5, init_a1 = -3, init_a0 = 1
    init_quadparams = [0.5,-3,1]
    quad_sampler = run_MCMC(data,init_quadparams, nwalkers, nsteps)
    
    lin_fits,lin_perrs,line_nerrs = output_fits(lin_sampler, burnin = burnin)
    quad_fits,quad_perrs,quad_nerrs = output_fits(quad_sampler, burnin = burnin)
    
    #ART plot fitted models over data
    plot_fits(data,lin_fits)
    plot_fits(data,quad_fits)

    
    print(
"""The results show that a2 is largely degenerate with the other paramters. a2 is approximately zero
for a wide range of values for a1 and a0. Even though there is some variation in a2 (higher a1 gives lower a2
and higher a0 gives higher a2), these variations are very small compared to the other parameters. Zero is just
outside the uncertainty around the best value for a2, so while there could be something there, ~1 sigma away from zero
does not instill confidence. So there's a strong possibility the quadratic model is uncessary, but we would need
to do a Chi^2 or bayesian analysis to know for sure."""
    )

    #ART display plots
    plt.show()    
    
    
