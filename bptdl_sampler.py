#!/usr/bin/env python
'''
2013-01-30 15:47:25 by Dawen Liang <dl2771@columbia.edu>
'''
import logging

import numpy as np
import numpy.random as nr

import btdl_sampler

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s %(name)s %(asctime)s '
                    '%(filename)s:%(lineno)d  %(message)s')
logger = logging.getLogger('bptdl_sampler')

class BPTDL_Sampler(btdl_sampler.BPDL_Sampler):
    def __init__(self, X, **kwargs):
        super(BPTDL_Sampler, self).__init__(X, **kwargs)
    
    def _init_sampler(self, initOption, save):
        if initOption == 'Rand':
            self.r_s, self.r_e = 1., 1. 
            sigma_s, sigma_e = np.sqrt(1./self.r_s), np.sqrt(1./self.r_e)
            sigma_D = np.sqrt(1./self.F)
            self.D = nr.randn(self.F, self.K) * sigma_D
            self.S = np.zeros((self.K, self.N))
            self.S[:,0] = nr.randn(self.K) * sigma_s 
            for i in xrange(1, self.N):
                self.S[:,i] = nr.randn(self.K) * sigma_s + self.S[:,i-1]
            self.Z = np.zeros((self.K, self.N), dtype=bool)
            self.pi = 0.01 * np.ones((self.K,))

        if initOption == 'SVD':
            super(BPTDL_Sampler, self)._init_sampler(initOption, save)
            return

        self.ll[0] = self.log_likelihood()

        if save:
            self._save(0)

    def sample_D(self):
        print 'Sampling D...'
        for k in xrange(self.K):
            self.sample_dk(k)
        
    def sample_dk(self, k):
        sigma_dk =1./(self.F+self.r_e*(self.S[k,self.Z[k,:]]**2).sum()) 
        mu_D = (self.r_e*sigma_dk) * np.dot(self.X[:,self.Z[k,:]],self.S[k,self.Z[k,:]])
        
        self.D[:,k] = nr.randn(self.F) * np.sqrt(sigma_dk) + mu_D
        logger.debug('Sample d_{} from normal distribution with mean = {} and '
                'variance {}'.format(k, mu_D, sigma_dk))

    def sample_Z(self):
        print 'Sampling Z...'
        for k in xrange(self.K):
            self.sample_zk(k)

    def sample_zk(self, k):
        Sk = self.S[k,:]
        Sk[~self.Z[k,:]] = nr.randn(self.N-self.Z[k,:].sum())*np.sqrt(1./self.r_s)
        DTD = np.dot(self.D[:,k], self.D[:,k])
        tmp = -0.5*self.r_e*(Sk**2 * DTD - 2*Sk*np.dot(self.X.T, self.D[:,k]))
        tmp = np.exp(tmp) * self.pi[k]
        self.Z[k,:] = nr.rand(self.N) > (1-self.pi[k])/(tmp+1-self.pi[k])

        
    def sample_S(self): 
        print 'Sampling S...'
        for k in xrange(self.K):
            self.sample_sk(k)
    
    def sample_sk(self, k):
        # TODO: please speed up
        DTD = np.dot(self.D[:,k], self.D[:,k])
        sigS1 = 1./(2 * self.r_s + self.r_e*DTD)
        if self.Z[k,0]:
            muS1 = sigS1*(self.r_s*self.S[k,1] + self.r_e*np.dot(self.X[:,0], self.D[:,k]))
            self.S[k,0] = nr.randn(1)*np.sqrt(sigS1) + muS1
        else:
            self.S[k,0] = 0
        for i in xrange(1, self.N-1):
            if self.Z[k,i]:
                muS1 = sigS1*(self.r_s*(self.S[k,i-1] + self.S[k,i+1]) + self.r_e*np.dot(self.X[:,i], self.D[:,k]))
                self.S[k,i] = nr.randn(1)*np.sqrt(sigS1) + muS1
            else:
                self.S[k,i] = 0
        if self.Z[k,self.N-1]:
            muS1 = sigS1*(self.r_s*self.S[k,self.N-2] + self.r_e*np.dot(self.X[:,self.N-1], self.D[:,k]))
            self.S[k,self.N-1] = nr.randn(1)*np.sqrt(sigS1) + muS1
        else:
            self.S[k,i] = 0

    def sample_pi(self):
        sumZ = self.Z.sum(axis=1)
        #self.pi = nr.beta(self.a0/self.K + sumZ, self.b0*(self.K-1)/self.K + self.N - sumZ)
        self.pi = nr.beta(self.a0 + sumZ, self.b0 + self.N - sumZ)

    def sample_phi(self):
        cnew = self.c0 + 0.5*self.K*self.N
        diffs = np.hstack((self.S[:,0].reshape(self.K,1), np.diff(self.S)))
        #dnew = self.d0 + 0.5*np.sum(diffs**2)
        dnew = self.d0 + 0.5*np.sum(diffs**2) + 0.5*(self.K*self.N - self.Z.sum())*(1./self.r_s)
        self.r_s = nr.gamma(cnew, scale=1./dnew)

        enew = self.e0 + 0.5*self.F*self.N
        fnew = self.f0 + 0.5*np.sum(self.X**2)
        self.r_e = nr.gamma(enew, scale=1./fnew)




