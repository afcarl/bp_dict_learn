import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt

import librosa
import bptdl_sampler as bts

class Audio_DL(object):
    def __init__(self):
        pass

    def process(self, x, sr, n_fft=512, load_mat=False, filename='', log=False, norm_option='', iter=200, K=1024, dl=bts.BPTDL_Sampler(), visualize=True):
        self.Xc = librosa.stft(x, sr=sr, n_fft=n_fft)
        X = np.abs(self.Xc)
        ## Configure
        self.log = log
        self.n_fft = n_fft
        self.dl = dl 

        if load_mat:
            self.load_mat(filename)
        else:
            if log:
                X = 20*np.log10(X)
            if norm_option == 'Standarize':
                X = (X - np.mean(X))/np.sqrt(np.var(X))
            if norm_option == 'MeanSub':
                X = X - np.mean(X)
            if norm_option == 'MeanDiv':
                X = X/np.mean(X)
            self.dl.sample(X, iter, K=K)
        self.sidx = np.flipud(np.argsort(self.dl.pi))
        if visualize:
            plt.plot(self.dl.pi[self.sidx], '-o')

    def load_mat(self, filename):
        d = sio.loadmat(filename)
        self.dl.D = d['D']
        self.dl.S = d['S']
        self.dl.Z = d['Z']
        self.dl.pi = d['pi'].flatten()

    def separate(self, L, save=True):
        xl = [] 
        sidx = self.sidx
        den = np.dot(self.dl.D[:,sidx[:L]], self.dl.Z[sidx[:L],:]*self.dl.S[sidx[:L],:])
        den[den==0] += 1e-6
        for l in xrange(L): 
            XL = self.Xc * np.dot(self.dl.D[:,sidx[l]].reshape(-1,1), self.dl.Z[sidx[l],:]*self.dl.S[sidx[l],:].reshape(1,-1))/den
            xl.append(librosa.istft(XL, n_fft=self.n_fft))
        if save:
            sio.savemat('xl.mat', {'xl':np.array(xl)})
        return np.array(xl)

    def eval_SNR(self):
        pass

    def figures(self, top=20, save=False):
        plt.subplot(211)
        plt.imshow(20*np.log10(np.abs(self.Xc)), cmap=plt.cm.hot_r, origin='lower', aspect='auto', interpolation='nearest')
        plt.colorbar()
        plt.subplot(223)
        sidx = self.sidx
        plt.imshow(self.dl.D[:,sidx[:top]], cmap=plt.cm.hot_r, origin='lower', aspect='auto', interpolation='nearest')
        plt.colorbar()

        plt.subplot(224)
        plt.imshow(self.dl.S[sidx[:top],:]*self.dl.Z[sidx[:top],:], cmap=plt.cm.hot_r, origin='lower', aspect='auto', interpolation='nearest')
        plt.colorbar() 
        if save:
            plt.savefig('figure.eps')




