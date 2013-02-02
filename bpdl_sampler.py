#!/usr/bin/env python
'''
2013-01-30 15:47:47 by Dawen Liang <dl2771@columbia.edu>
'''

import sys
import logging

import numpy as np
import numpy.random as nr
from scipy import linalg
import scipy.stats as sstats
import scipy.io as sio

logging.basicConfig(filename='sampler.log', level=logging.INFO,
                    format='%(levelname)s %(name)s %(asctime)s '
                    '%(filename)s:%(lineno)d  %(message)s')
logger = logging.getLogger('bpdl_sampler')

class BPDL_Sampler(object):
    def __init__(self, X, **kwargs):
        self.X = X
        self.F, self.N = X.shape
        self._parse_args(**kwargs)

    def _parse_args(self, **kwargs):
        if 'a0' in kwargs and 'b0' in kwargs:
            self.a0, self.b0 = kwargs['a0'], kwargs['b0']
        else:
            self.a0, self.b0 = 1., 1. 
        if 'c0' in kwargs and 'd0' in kwargs:
            self.c0, self.d0 = kwargs['c0'], kwargs['d0']
        else:
            # d0: rate (1/scale)
            self.c0, self.d0 = 1e-6, 1e-6 
        if 'e0' in kwargs and 'f0' in kwargs:
            self.e0, self.f0 = kwargs['e0'], kwargs['f0']
        else:
            self.e0, self.f0 = 1e-6, 1e-6

    def _init_sampler(self, initOption, save):
        if initOption == 'Rand':
            self.r_s, self.r_e = 1., 1. 
            sigma_s, sigma_e = np.sqrt(1./self.r_s), np.sqrt(1./self.r_e)
            sigma_D = np.sqrt(1./self.F)
            self.D = nr.randn(self.F, self.K) * sigma_D
            self.S = nr.randn(self.K, self.N) * sigma_s
            self.Z = np.zeros((self.K, self.N), dtype=bool)
            self.pi = 0.01 * np.ones((self.K,))

        if initOption == 'SVD':
            self.a0, self.b0 = 1, self.N/8
            self.r_s, self.r_e = 1., 1. 
            U, S, Vh = linalg.svd(self.X, full_matrices=False)
            if self.F < self.K:
                self.D = np.zeros((self.F, self.K))
                self.D[:, 0:self.F] = U
                self.S = np.zeros((self.K, self.N))
                self.S[0:self.F, :] = np.dot(np.diag(S), Vh);
            else:
                self.D = U[0:self.F, 0:self.K]
                self.S = np.dot(np.diag(S), Vh)
                self.S = self.S[0:self.N, :]
            self.Z = np.ones((self.K, self.N), dtype=bool)
            self.pi = 0.5 * np.ones((self.K,))

        self.ll[0] = self.log_likelihood()

        if save:
            self._save(0)

    def sample(self, maxIter, K=512, initOption='SVD', updateOption='DkZkSk',
            save=False, iter_log=True):
        print 'Init the sampler with option: {}...'.format(initOption)
        if maxIter < 1:
            sys.exit('{} is not a valid iteration number'.format(maxIter))
        self.K = K
        self.ll = np.zeros((maxIter,))
        self._init_sampler(initOption, save)
        
        self.X = self.X - np.dot(self.D, self.Z * self.S)
        for iter in xrange(1, maxIter):
            if iter_log:
                import time
                start = time.time()

            self.sample_DZS(updateOption) 
            self.sample_pi() 
            self.sample_phi()

            if iter_log:
                end = time.time() - start
                logger.info('iter: {}\ttime: {:.2f}\tave_Z: {:.0f}\tM: {}\tNoiseVar: {:.4f}\tSVar: {:.4f}'.format(iter, end, np.mean(self.Z.sum(axis=0)), np.sum(self.pi>0.001), np.sqrt(1./self.r_e), np.sqrt(1./self.r_s))) 
            self.ll[iter] = self.log_likelihood()
            if save:
                self._save(iter)

    def sample_DZS(self, updateOption):
        if updateOption == 'DkZkSk':
            for k in xrange(self.K):
                self.X[:, self.Z[k,:]] = self.X[:,self.Z[k,:]] + np.dot(self.D[:,k].reshape(self.F,1), self.S[k,self.Z[k,:]].reshape(1,-1))
                self.sample_dk(k)
                self.sample_zk(k)
                self.sample_sk(k)

                self.X[:,self.Z[k,:]] = self.X[:,self.Z[k,:]] - np.dot(self.D[:,k].reshape(self.F,1), self.S[k,self.Z[k,:]].reshape(1,-1))
                
        if updateOption == 'DZS':
            ## TODO not implemented yet
            pass

    def _save(self, iter):
        save_name = 'K{}_F{}_T_iter{}.mat'.format(self.K, self.F, iter)
        mdict = {}
        mdict['rs'], mdict['re'] = self.r_s, self.r_e
        mdict['D'] = self.D
        mdict['S'] = self.S
        mdict['Z'] = self.Z
        mdict['pi'] = self.pi
        sio.savemat(save_name, mdict)

    def log_likelihood(self):
        sigma_s, sigma_e = np.sqrt(1./self.r_s), np.sqrt(1./self.r_e)
        sigma_D = np.sqrt(1./self.F)
        ll = sstats.norm.logpdf(self.D, 0, sigma_D).sum() 

        ll += sstats.norm.logpdf(self.S[:,0], 0, sigma_s).sum()
        ll += sstats.norm.logpdf(np.diff(self.S), 0, sigma_s).sum()
        ll += sstats.binom.logpmf(self.Z.T, 1, self.pi).sum()

        #ll += sstats.beta.logpdf(self.pi,self.a0/self.K,self.b0*(self.K-1)/self.K).sum() 
        ll += sstats.beta.logpdf(self.pi, self.a0, self.b0).sum()
        ll += sstats.norm.logpdf(self.X,0,sigma_e).sum()

        return ll


