import numpy as np
import os
import matplotlib.pyplot as plt


def proposal(m,b,xs,ys,var,stdev=0.1):

    newm = m+np.random.normal(loc=0,scale=stdev,size=1)[0]
    newb = b+np.random.normal(loc=0,scale=stdev,size=1)[0]
    
    oldlnP = calc_lnL(m,b,xs,ys,var)
    newlnP = calc_lnL(newm,newb,xs,ys,var)
    lnR = newlnP - oldlnP
    if lnR >= 0:
        # Accept automatically if new point has higher probability
        return newm,newb,newlnP
    else:
        # Accept with probability R = exp(lnR)
        if np.random.rand() < np.exp(lnR):
            return newm,newb,newlnP
        else:
            # Reject: keep current parameters
            return m,b,oldlnP
  

def calc_lnL(m,b,xs,ys,var):
    #priors
    if m < 0 or m > 6:
        return -np.inf
    if b < 0 or b > 8:
        return -np.inf
    tot = 0
    for i in range(len(ys)):
        s = (ys[i]-(m*xs[i]+b))**2/var[i]+np.log(2*np.pi*var[i])
        tot+=s
    lnL = -0.5*tot
    return lnL
    
def MC_walk(numiter,m,b,xs,ymeans,var):
    ms = [m]
    bs = [b]
    post = np.exp(calc_lnL(m,b,xs,ymeans,var))
    posts = [post]
    for n in range(numiter):
        m,b,post = proposal(m,b,xs,ymeans,var,stdev=.1)
        ms.append(m)
        bs.append(b)
        posts.append(np.exp(post))

    return ms,bs,posts
    
def get_best(x):
    burnin = len(x)//10
    goodxs = x[burnin:]
    bestx = np.mean(goodxs)
    sigx = np.std(goodxs)
    
    return bestx,sigx

if __name__ == "__main__":
    
#Lesson 25
    
    data = np.genfromtxt(os.path.join(os.getenv("ASTR5160"),'week13','line.data'))
    #print(data)
    
    ymeans = np.mean(data,axis=0)
    print(ymeans)
    
    var = np.var(data,axis=0,ddof=1,mean=ymeans)
    print(var)
    xs = np.arange(0,10)+0.5
    
    m = 2
    b = 5
    
    ms,bs,posts = MC_walk(10000,m,b,xs,ymeans,var)
    
    bestm,sigm = get_best(ms)
    bestb,sigb = get_best(bs)
        
    #print(ms)
    #print(bs)
    ns = np.arange(len(ms))
    plt.figure()
    plt.plot(ns,ms,'b-',label=f'Best m = {bestm:.3f} +/- {sigm:.3f}')
    plt.plot(ns,bs,'r-',label=f'Best b = {bestb:.3f} +/- {sigb:.3f}')
    plt.xlabel('iterations')
    plt.ylabel('value')
    plt.grid()
    plt.legend()
    #plt.show()
    
#---------------------------------------------------------------------------------------
#Lesson 26    
    def log_prior(theta):
        m,b = theta
        if 0 < m < 8 and 0.0 < b < 10.0:
            return 0.0
        return -np.inf
   
    def log_probability(theta, x, y, yerr):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, x, y, yerr)
    print(ymeans)
    import emcee
    
    def log_likelihood(theta, x, y, yerr):
        m, b = theta
        model = m * x + b
        sigma2 = yerr**2 + model**2
        return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

    pos = np.array([m-1,b+1])+ 1e-4 * np.random.randn(32, 2)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(xs, ymeans, np.sqrt(var))
    )
    sampler.run_mcmc(pos, 5000, progress=True);
        
        
    fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["m", "b"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    
    from IPython.display import display, Math

    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        print(labels[i], mcmc[1], q[0], q[1])
        
    
    
    plt.show()
