"""
 test_pymc3.py
 2021/6/15 S.O
"""

import theano
import pymc3 as pm

import numpy as np
import pandas as pd

print(f'{pm.__name__}: v. {pm.__version__}')
print(f'{theano.__name__}: v. {theano.__version__}')

if __name__ == '__main__':
    SEED = 12345
    np.random.seed(SEED)

    mu_r = 0
    sd_r = 1
    n_samples = 100
    y = np.random.normal(loc=mu_r, scale=sd_r, size=n_samples)

    # Bayesian modelling
    with pm.Model() as model:
    
        mu = pm.Normal('mu', mu=0, sd=1)
        sd = pm.HalfNormal('sd', sd=1)
    
        likelihood = pm.Normal('likelihood', mu=mu, sd=sd, observed=y)    
        trace = pm.sample(chains=2, cores=2, random_seed=SEED)

    print('result!!!',trace['mu'].mean(), trace['sd'].mean())