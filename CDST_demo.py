#!/usr/bin/env python
#coding:utf-8
# Author:  Chris Musselle -- <chris.j.musselle@gmail.com>
#
# Purpose: The Change-point Detecting Subspace Traker (CD-ST) Algorithm
#
# File defining the CDST object that contains all methods for the algorithm.
# Running this file will define a CDST object and run one experiment as defined
# by the parameters and choice of data set after the if __name__ == '__main__' 
# statement. 
#
# References and Acknowledgements: 
# The FHST_iter function is an implimentation of the Fast Row Householder 
# Subspace Tracking algorithm which first appeared in the following reference.
#
# Strobach, P. (2009). The fast recursive row-Householder subspace tracking algorithm. 
# Signal Process., 89(12), 2514-2528. Amsterdam, The Netherlands, Elsevier North-Holland, Inc.

# Created: 11/23/11


# Run from current path location
import sys
import os
sys.path.append(os.getcwd() + '/datasets')
sys.path.append(os.getcwd() + '/utils')

import numpy as np
from numpy import dot
import scipy as sp
from math import sqrt
import numpy.linalg as npl
import matplotlib.pyplot as plt

from plot_utils import plot_2x1, plot_3x1, plot_4x1, adjust_spines
from utils import fmeasure, clean_zeros, zscore, zscore_win
from load_data import load_data, load_ts_data
from gen_anom_data import gen_a_grad_persist, gen_a_peak_dip, gen_a_step, gen_a_periodic_shift
from SAX import SAX, plot_SAX, bp_lookup


class CDST(object):
  """ Class that holds all methods of CDST 

  version is a string specifying the combination of mudules to use 

  F = FHST version 
  A = Anomaly Detection Version
  S = Sax Version, still a work in progress. 

  Format = 'F-xxxxxxx.A-xxxxxxxx.S-xxxxxxxx'
  """

  def __init__(self, version, p, numStreams = 1):

    self.F_version = version.split('.')[0]
    self.A_version = version.split('.')[1]
    if len(version.split('.')) > 2:
      self.S_version = version.split('.')[2]
    else:
      self.S_version = 'none'

    self.numStreams = numStreams
    
    # Calculate threshold. Depends on whether test is one or two tailed. 
    if '+ve' in self.A_version or '-ve' in self.A_version:      
      p['t_thresh'] = sp.stats.t.isf(1.0 * p['FP_rate'], p['SRE_sample_N'])
    elif 'both' in self.A_version:
      p['t_thresh'] = sp.stats.t.isf(0.5 * p['FP_rate'], p['SRE_sample_N'])      
    
    self.p = p
    self.p['version'] = version        

    """ Initialise all CD-ST variables """

    r = self.p['init_r']
    
    # Q_0
    if self.p['fix_init_Q'] != 0:  # fix inital Q as identity 
      q_0 = np.eye(numStreams);
      Q = q_0
    else: # generate random orthonormal matrix N x r 
      Q = np.eye(numStreams) # Max size of Q
      Q_0, R_0 = npl.qr(np.random.rand(numStreams,r))   
      Q[:,:r] = Q_0          
    # S_0
    small_value = self.p['small_value']
    S = np.eye(numStreams) * small_value # Avoids Singularity    
    # v-1
    v = np.zeros((numStreams,1)) 
    # U(t-1) for eigenvalue estimation
    U = np.eye(numStreams)

    # Define st dictionary 
    """ This stores variables from one timestep to the next """
    self.st  = {'Q' : Q,         # Orthogonal dominant subspace vectors
                'S' : S,     # Energy
                'v' : v,     # used for S update
                'U' : U,     # Used for eigen value calculations 
                'r' : r,     # Previous rank of Q and number of hidden variables h
                't' : 0,     # Timestep, used for ignoreup2  
                'sumEz' : 0.0,        # Exponetial sum of zt Energy 
                'sumEh': 0.0,     # Exponential sum of ht energy  
                'anomaly': np.array([0]*self.numStreams,dtype = bool)} 
      
    # Vars for SAX usage
    if 'none' not in self.S_version:
      self.st['SAX_trigger_q'] = [] 
      self.st['SAX_snapshots'] = {}

  def re_init(self, numStreams):
  
    self.numStreams = numStreams
    
    # This deletes all tracked values 
    if hasattr(self, 'res'):
      del self.res
    
    """ Initialise all CD-ST variables """
   
    r = self.p['init_r']
    # Q_0
    if self.p['fix_init_Q'] != 0:  # fix inital Q as identity 
      q_0 = np.eye(numStreams);
      Q = q_0
    else: # generate random orthonormal matrix N x r 
      Q = np.eye(numStreams) # Max size of Q
      Q_0, R_0 = npl.qr(np.random.rand(numStreams,r))   
      Q[:,:r] = Q_0          
    # S_0
    small_value = self.p['small_value']
    S = np.eye(numStreams) * small_value # Avoids Singularity    
    # v-1
    v = np.zeros((numStreams,1)) 
    # U(t-1) for eigenvalue estimation
    U = np.eye(numStreams)
  
    # Define st dictionary 
    self.st  = {'Q' : Q,          # Orthogonal dominant subspace vectors
                'S' : S,          # Energy
                'v' : v,          # used for S update
                'U' : U,          # Used for eigen value calculations 
                'r' : r,          # Previous rank of Q and number of hidden variables h
                't' : 0,          # Timestep, used for ignoreup2  
                'sumEz' : 0.0,    # Exponetial sum of zt Energy 
                'sumEh': 0.0,     # Exponential sum of ht energy  
                'anomaly': np.array([0]*self.numStreams,  dtype = bool)}
  
    # vars for SAX usage
    if 'none' not in self.S_version:
      self.st['SAX_trigger_q'] = []  
      self.st['SAX_snapshots'] = {}

  def next_input(self, zt):
    # Run Subspace Tracking
    if 'FHST' in self.F_version:
      self.FHST_iter(zt)
    else:
      print 'Error: %s method for subspace tracking not recognised' % (self.F_version)

    # Run anomaly Detection 
    if 'SRE' in self.A_version:
      self.anomaly_SREstat_fast(zt)
    else:
      print 'Error: %s method for detect anomalies not recognised' % (self.A_version)    

    # Run Sax Method
    if 'none' not in self.S_version:
      self.SAX_simple(zt)

  def FHST_iter(self, zt):
    """ Iterable version of the Fast row Housholder Algorithm """
    ''' 
    zt = data at next time step 
    st = {'Q' : Q,     - Orthogonal dominant subspace vectors
              'S' : S,     - Energy 
              'U' : U,     - Orthonormal component of Orthogonal iteration around X.T
              'v' : v,     - Used for speed up of calculating X update 
              'r' : r,     - Previous rank of Q and number of hidden variables h
              't' : t,     - Timestep, used for ignoreup2  
              'sumEz' : Et,  - Exponetial sum of zt Energy 
              'sumEh': E_dash_t }- Exponential sum of ht energy 

    '''

    # Load Variables  
    
    st = self.st 
    p = self.p
    numStreams = self.numStreams
    r = st['r']

    # NOTE algorithm's Q, S, v and U  matrices/ vectors are kept at max size N (constant memory)
    # Create alias's for current value of r
    Qt  = st['Q'][:, :r]
    vt  = st['v'][:r, :]
    St  = st['S'][:r, :r]
    Ut  = st['U'][:r, :r]

    # Check S remains non-singular
    for idx in range(r):
      if St[idx, idx] < p['small_value']:
        St[idx,idx] = p['small_value']

    '''Begin main algorithm'''        
    ht = dot(Qt.T, zt) 
    Z = dot(zt.T, zt) - dot(ht.T , ht)

    if Z > 0 :

      # Flag for whether Z(t-1) > 0
      # Used for alternative eigenvalue calculation if Z < 0
      #st['last_Z_pos'] = bool(1)

      # Strobach version of FHST with use of extra u_vec terms
      u_vec = dot(St , vt)
      X = (p['alpha'] * St) + (2 * p['alpha'] * dot(u_vec, vt.T)) + dot(ht, ht.T)

      # Solve to find b_vec
      A = X.T
      B = sqrt(Z) * ht
      b_vec = npl.solve(A,B)      

      beta  = 4 * (dot(b_vec.T , b_vec) + 1)

      phi_sq = 0.5 + (1.0 / sqrt(beta))

      phi = sqrt(phi_sq)

      gamma = (1.0 - 2 * phi_sq) / (2 * phi)

      delta = phi / sqrt(Z)

      vt = gamma * b_vec 

      St = X - ((1 /delta) * dot(vt , ht.T))

      w = (delta * ht) - (vt) 

      ee = delta * zt - dot(Qt , w) 

      Qt = Qt - 2 * dot(ee , vt.T) 

    else: # if Z is not > 0
      # Implies norm of ht is > zt or zt = 0

      St = p['alpha'] * St # Continue decay of S matrix 

    '''Store Values''' 
    # Update stored values
    st['Q'][:,:r] = Qt
    st['v'][:r,:] = vt
    st['S'][:r, :r] = St
    st['U'][:r,:r] = Ut

    # Record hidden variables
    ht_vec = np.hstack((ht.T[0,:], np.array([np.nan]*(self.numStreams-r))))
    st['ht'] = ht_vec

    # Energy Ratio Calculations 
    st['Ez'] = np.sum(zt ** 2) # the norm squared of zt
    st['Eh'] = np.sum(ht ** 2) # the norm squared of ht
    st['sumEz'] = p['alpha']*st['sumEz'] + st['Ez'] # Energy of Data
    st['sumEh'] = p['alpha']*st['sumEh'] + st['Eh'] # Energy of Hidden Variables

    if st['sumEz'] == 0 : # Catch NaNs 
      st['e_ratio']  = 0.0
    else:
      st['e_ratio']  = st['sumEh'] / st['sumEz']

    self.st = st

  def anomaly_SREstat_fast(self, zt):
    """ Calculates a test statistic for ressidul of zt_reconstructed """

    st = self.st
    p = self.p  

    # Slow way 
    #st['recon'] = dot(st['Q'][:,:st['r']],st['ht'][:st['r']])
    #st['recon_err'] = zt.T - st['recon']
    #SRE = npl.norm(st['recon_err'])
    
    # Fast Way
    # Squared Reconstrunction Error (SRE) or
    # Squared norm of the residual error vector 
    SRE = (st['Ez'] - st['Eh']) 

    # Build/Slide recon_err_window
    if  st.has_key('SRE_win'):
      st['SRE_win'][:-1] = st['SRE_win'][1:] # Shift Window
      st['SRE_win'][-1] = SRE
    else:
      st['SRE_win'] = np.zeros(((p['SRE_sample_N'] + p['dependency_lag']), 1))
      st['SRE_win'][-1] = SRE

    # Differenced SRE 
    st['SRE_dif_sample'] = np.diff(st['SRE_win'], axis = 0)
    st['SRE_dif_t'] = st['SRE_dif_sample'][-1]
    
    SRE_sample_sum = (st['SRE_dif_sample'][-(p['SRE_sample_N'] + p['dependency_lag']):-p['dependency_lag']]**2).sum()

    # Calculate Test Statistic
    st['t_stat'] = st['SRE_dif_t'] / np.sqrt( SRE_sample_sum / (p['SRE_sample_N']-1.0)) 

    # Three Possible Versions to test threshold, +ve only, -ve only or both
    calc_anomaly = 0
    if st['t'] > p['ignoreUp2']:    
      if '+ve' in self.A_version and st['t_stat'] > p['t_thresh']:
        calc_anomaly = 1
      elif '-ve' in self.A_version and st['t_stat'] < -p['t_thresh']:
        calc_anomaly = 1
      elif 'both' in self.A_version and np.abs(st['t_stat']) > p['t_thresh']:
        calc_anomaly = 1

    if calc_anomaly:
      # Explicitly Calculate reconstruction error vector 
      recon = np.dot(st['Q'][:,:st['r']],st['ht'][:st['r']])
      error = zt[:,0] - recon
      # Anomaly is vector of bools showing which streams are responsible
      st['anomaly'] = np.abs(error) > np.abs(error).mean() # threshold value may need changing depending on application

    self.st = st 

  def SAX_simple(self, zt):
    """ simplest implimentation of SAX in FRAHST 
    
    uses sliding window over recent values - may be able to improve to itterative version later...
    
    Takes SAX snap shot when anomalous point is halfway down zt sample. 
    """ 

    p = self.p
    st = self.st
    
    if 'none' not in self.S_version: 
      
      # Build/Slide ht sample window
      if st.has_key('ht_sample'):
        st['ht_sample'][:-1] = st['ht_sample'][1:] # Shift Window
        st['ht_sample'][-1] = st['ht'][0]
      else:
        st['ht_sample'] = np.zeros(p['zt_sample_size'])
        st['ht_sample'][-1] = st['ht'][0]
      
      # Build/Slide zt sample window
      if  st.has_key('zt_sample'):
        st['zt_sample'][:-1,:] = st['zt_sample'][1:,:] # Shift Window
        st['zt_sample'][-1,:] = zt[:,0]
      else:
        st['zt_sample'] = np.zeros((p['zt_sample_size'], self.numStreams))
        st['zt_sample'][-1,:] = zt[:,0]
  
      # If anomaly at current time step
      if np.any(st['anomaly']):
        # Store time step in SAX_que
        st['SAX_trigger_q'].append(int(st['t'] + np.round(p['zt_sample_size']/2.)))
        
      if st['SAX_trigger_q']:
        # Once any of these times are reached, take SAX snapshot, and remove from que
        if st['t'] in st['SAX_trigger_q']:
          
          # get SAX of hidden Var
          SAX_array_ht, SAX_dic_ht, seg_means_ht = SAX(np.atleast_2d(st['ht_sample']), p['SAX_alphabet_size'], 
                                                      p['word_size'], minstd = 0.0001, pre_normed = False)        
          
          # Get SAX of data 
          SAX_array_zt, SAX_dic_zt, seg_means_zt = SAX(st['zt_sample'], p['SAX_alphabet_size'], 
                                              p['word_size'], minstd = 0.1, pre_normed = False)

          # Remove from que 
          st['SAX_trigger_q'].remove(st['t'])      
          
          # store for data 
          index = str(int(st['t'] - np.round(p['zt_sample_size']/2.)))
          
          st['SAX_snapshots'][index] = {'zt_SAX_a' :  SAX_array_zt , 
                                        'zt_SAX_d' :  SAX_dic_zt ,  
                                        'zt_seg_m' :  seg_means_zt }

          # store for hidden variable
          st['SAX_snapshots'][index].update({'ht_SAX_a' :  SAX_array_ht , 
                                                  'ht_SAX_d' :  SAX_dic_ht ,  
                                                  'ht_seg_m' :  seg_means_ht })         
          
        self.st = st

  def track_var(self, values = (), print_anom = 0):
    """ Tracks variables specified over time.
    At the very least must track time step of anomalies and anomalous streams flagged"""
    
    if not hasattr(self, 'res'):
      # initalise res
      self.res = {}
      for k in values:
        self.res[k] = self.st[k]
        
      self.res['anomalies'] = []
      self.res['anomalous_streams'] = []
    
    else:
      # stack values for all keys
      for k in values:
        self.res[k] = np.vstack((self.res[k], self.st[k]))
      
      # If anomaly is present, print if specified. 
      if np.any(self.st['anomaly']):
        if print_anom == 1:
          print 'Found Anomaly at t = {0}'.format(self.st['t'])
        self.res['anomalies'].append(self.st['t'])
        self.res['anomalous_streams'].append(self.st['anomaly'])
        
    # Increment time        
    self.st['t'] += 1    

  def plot_res(self, var, xname = 'Time Steps', ynames = None, title = None, hline= 1, anom = 1):
    """Plots each of the elements given in var. 
    
    var = list of  variables. Maximum = 4. if string, will look for them in self.res structure 
        
    hline = whether to plot threshold values on final plot.
    
    anom = whether to plot anomalous time ticks.
        
    """
    
    if ynames is None:
      ynames = ['']*4
      
    if title is None:
      title = (self.p['version'])
        
    if 'SRE' in self.A_version:
      thresh = self.p['t_thresh']
    
    num_plots = len(var)
    
    for i, v in enumerate(var):
      if type(v) == str :
        var[i] = getattr(self, 'res')[v]
    
    if num_plots == 1:
      plt.figure()
      plt.plot(var[0])
      plt.title(title)
      if anom == 1:
        for x in self.res['anomalies']:
          plt.axvline(x, ymin=0.9, color='r')        
      
    elif num_plots == 2:
      plot_2x1(var[0], var[1], ynames[:2], xname, main_title = title)
      
      if hline == 1:
        plt.hlines(-thresh, 0, self.res['ht'].shape[0], linestyles = 'dashed')
        plt.hlines(+thresh, 0, self.res['ht'].shape[0], linestyles = 'dashed')
        plt.ylim(-3*thresh,3*thresh)
        
      if anom == 1:
        f = plt.gcf()
        for ax in f.axes[:-1]:
          for x in self.res['anomalies']:
            ax.axvline(x, ymin=0.9, color='r')              
        
    elif num_plots == 3:
      plot_3x1(var[0], var[1], var[2], ynames[:3] , xname, main_title = title) 

      
      if hline == 1:
        plt.hlines(-thresh, 0, self.res['ht'].shape[0], linestyles = 'dashed')
        plt.hlines(+thresh, 0, self.res['ht'].shape[0], linestyles = 'dashed') 
        plt.ylim(-3*thresh,3*thresh)
        
      if anom == 1:
        f = plt.gcf()
        for ax in f.axes[:-1]:
          for x in self.res['anomalies']:
            ax.axvline(x, ymin=0.9, color='r')         
               
    elif num_plots == 4:
      plot_4x1(var[0], var[1], var[2], var[3], ynames[:4], xname, main_title = title)
      plt.title(title)
      
      if hline == 1:
        plt.hlines(-thresh, 0, self.res['ht'].shape[0], linestyles = 'dashed')
        plt.hlines(+thresh, 0, self.res['ht'].shape[0], linestyles = 'dashed')               
        plt.ylim(-3*thresh,3*thresh)
        
      if anom == 1:
        f = plt.gcf()
        for ax in f.axes[:-1]:
          for x in self.res['anomalies']:
            ax.axvline(x, ymin=0.9, color='r')     
      
    plt.draw()      

  

  def batch_analysis(self, gt_list, anomalies_list, epsilon = 0, accumulative = 1, keep_sets = 1):
      """ Calculate all anomally detection Metrics 
      
      # gt_list: list of gt_tables per initial condition 
                  gt_table: ground truth table, each entry has 
                  dtype = [('start','i4'),('loc','i4'),('len','i4'),('mag','i4'),('type','a10')])
      
      # epsilon: used to allow for lagged detections: if anomaly occurs in time window
                 anom_start - anom_end + eplsilon it is considered a TP.
      
      # accumulative: Whether multiple calls will act accumulateively to metric values. 
      
      # keep_sets: whether to store the sets fot TP, FP, TN etc.  
      
      """
      
      # For each initial condition 
      for k in xrange(len(anomalies_list)):
        
        gt_table = gt_list[k]['gt']
        anomalies = anomalies_list[k] 
      
        # Detections  
        D = np.array(anomalies)
        index =  D > self.p['ignoreUp2'] 
        D = set(list(D[index]))        
        
        # initalise metrics 
        if not hasattr(self, 'metric') or accumulative == 0:
          self.metric = { 'TP' : 0.0 ,
                     'FP' : 0.0 ,
                     'FN' : 0.0 ,
                     'TN' : 0.0,
                     'precision' : 0.0 ,
                     'recall' : 0.0 ,
                     'F1' : 0.0, 
                     'F2' : 0.0, 
                     'F05' : 0.0,
                     'FPR' : 0.0,
                     'FDR' : 0.0,
                     'ACC' : 0.0}
          self.detection_sets = []
          self.anom_detect_tab = []
  
        # set of point anomalies detected as true
        anom_TP = set()
        
        # Set of anomalous segments detected           
        anom_segments_detected_set  = set()  
        
        # Table to record frequency of anomalous segment detections
        anomalies_detected_tab  = np.zeros((len(gt_table['start']), 2))
        anomalies_detected_tab[:,0] = gt_table['start']
        
        # TRUE POSITIVES
        
        idx = 0
        for i in xrange(len(gt_table['start'])):
            count = 0
            # Run through the list of detections    
            for d in D:
              if d >= gt_table['start'][i]  and d <= gt_table['start'][i] + gt_table['len'][i] + epsilon:
                # if set does not yet contain the anomaly, add it and increment TP
                if not anom_segments_detected_set.issuperset(set([gt_table['start'][i]])):
                  
                  anom_segments_detected_set.add(gt_table['start'][i])
                  anom_TP.add(d)
                  self.metric['TP'] += 1
                  count += 1
                else: # if multiple detections in anomalous segment 
                  count += 1 
                  anom_TP.add(d)                    
                      
            anomalies_detected_tab[idx,1] = count   
            idx += 1     
        
        # FALSE Pos 
        anom_FP = D - anom_TP    
        self.metric['FP'] += len(anom_FP)
        # FALSE Neg     
        anom_FN = set(gt_table['start']) - anom_segments_detected_set
        self.metric['FN'] += len(anom_FN)
        # True Negatives
        self.metric['TN'] += (self.st['t'] - self.p['ignoreUp2'] - len(anom_FN) - len(anom_FP) - len(anom_TP))
  
        if self.metric['FP'] == 0 and self.metric['TP'] == 0:
          self.metric['precision'] += 0
          self.metric['FDR'] += 0
        else:
          self.metric['precision'] = self.metric['TP'] / (self.metric['TP'] + self.metric['FP'])          
          self.metric['FDR'] = self.metric['FP'] / (self.metric['FP'] + self.metric['TP'])    
  
        self.metric['recall'] = self.metric['TP'] / (self.metric['TP'] + self.metric['FN'])      
        self.metric['FPR'] = self.metric['FP'] / (self.metric['TN'] + self.metric['FP'])      
        self.metric['ACC'] = (self.metric['TP'] + self.metric['TN']) /  \
                        ( self.metric['TP'] + self.metric['FN'] + self.metric['TN'] + self.metric['FP'] )
                        
        self.metric['F1'] = self.fmeasure(1, self.metric['TP'], self.metric['FN'], self.metric['FP'])
        self.metric['F2'] = self.fmeasure(2, self.metric['TP'], self.metric['FN'], self.metric['FP'])
        self.metric['F05'] = self.fmeasure(0.5, self.metric['TP'], self.metric['FN'], self.metric['FP']) 
        
        if keep_sets == 1:
          sets = {'TP' : anom_TP,
                  'anom_seg_detected' : anom_segments_detected_set,
                  'FN' : anom_FN,
                  'FP' : anom_FP}     
          self.detection_sets.append(sets)
          self.anom_detect_tab.append(anomalies_detected_tab)
    
  def fmeasure(self, B, hits, misses, falses):
      """ General formular for F measure 
      
      Uses TP(hits), FN(misses) and FP(falses)
      """
      x = ((1 + B**2) * hits) / ((1 + B**2) * hits + B**2 * misses + falses)
      return x


  def plot_SAX(self, anomaly):
    """ For a given detected anomaly, plots the SAX snapshot around that time point """

    # Need to add input checks 

    segs = self.st['SAX_snapshots'][str(anomaly)]['zt_seg_m']
    plot_SAX(segs, self.p['SAX_alphabet_size'], self.p['comp_ratio'])    
    
  def plot_dat(self, anomaly, data, standardise = 1):
    """ For a given detected anomaly, plots the data around that time point """

    # Need to add input checks 

    sample_half = int(round( self.p['zt_sample_size']/2.)) 
    dat = data[anomaly-sample_half:anomaly+sample_half,:]
    
    if standardise:
      fig = plt.figure()
      ax = fig.add_subplot(111)
      ax.plot(zscore(dat))
      bpList = bp_lookup(self.p['SAX_alphabet_size'])
      for bp in bpList:
        ax.axhline(y=bp, xmin=0, xmax=dat.shape[0], ls = '--', color = 'k')
        
      adjust_spines(ax, ['bottom'])
      
      ax.set_yticklabels([])
      ax.yaxis.set_ticks([])
      ax.set_xticklabels(range(anomaly-sample_half,anomaly+sample_half+1))
      for tick in ax.xaxis.get_major_ticks():
                      tick.label.set_fontsize(18)       
      
      
    else: 
      plt.figure()
      plt.plot(dat)
      bpList = bp_lookup(self.p['SAX_alphabet_size'])
      plt.hlines(bpList, xmin=0, xmax=dat.shape[0]-1, linestyles = 'dashed', color = 'k')           
           
    


if __name__=='__main__':
  
  ''' Experimental Run Parameters '''
  p = {'alpha': 0.98,        # The exponential decay factor
        'init_r' : 2,        # the initial number of hidden variables 
        'fix_init_Q' : 0,    # whether to fix initial Q as Identity or make random
        'small_value' : 0.0001,    # Used to avoid non-signularities 
        'ignoreUp2' : 50,          # Starting time steps ignored for anomalies
        # Statistical Anomaly Detection
        'SRE_sample_N' : 20,       # Size of SRE sample 
        'dependency_lag' : 5,      # lag between current and sampled SRE
        't_thresh' : None,         # Threshold value for test statistic 
        'FP_rate' : 10**-5,        # Significance level of statistical test 
        # SAX Parameters
        'word_size' : 10,          # No. of symbols in each word
        'zt_sample_size' : 10,     # No. of data points in sample
        'SAX_alphabet_size' : 6,   # No. of characters in word
        'comp_ratio' : None}       # Compression Ratio
      
  # Calculate threshold 
  p['t_thresh'] = sp.stats.t.isf(1.0 * p['FP_rate'], p['SRE_sample_N'])
  p['comp_ratio'] = float(p['zt_sample_size']) / float(p['word_size'])

  
  # Anomalous Data Parameters for using synthetic data set 
  
  a = { 'N' : 50,    # No. of streams
        'T' : 1000,  # Total time steps 
        'periods' : [15, 50, 70, 90], # periods of sinosoids
        'L' : 10,    # Length of anomaly (start) 
        'L2' : 200,  # Length of anomaly (hold)
        'M' : 5,     # Magnitude of anomaly 
        'pA' : 0.1,  # Percentage of streams that are anomalous
        'noise_sig' : 0.0,  # noise added 
        'seed' : 234}         # Random seed 
  
  anomaly_type = 'grad_persist'  # choice of peak_dip, grad_persist and step
  gen_funcs = dict(peak_dip = gen_a_peak_dip,
                   grad_persist = gen_a_grad_persist,
                   step = gen_a_step)
  
  
  """ Choice of data sets. Comment/Uncomment to choose. """
  """----------------------------------------------------"""
  
  ''' Synthetic Data sets ''' 
  #data_name = 'synth'
  #D = gen_funcs[anomaly_type](**a)  
  #raw_data = D['data']
  #data = raw_data.copy()
  
  '''ISP data sets '''

  data_name = 'isp_routers'
  raw_data = load_ts_data(data_name, 'full')
  data = raw_data.copy()
  
  ''' Sensor Motes data sets '''
  #data_name = 'motes_l'
  #raw_data = load_data(data_name)
  #data = clean_zeros(raw_data, cpy=1)  
  
  
  ''' Data Preprocessing '''
  """ Data is loaded into memory, mean centered and standardised
  then converted to an iterable to read by the CD-ST each iteration"""
  
  #data = zscore_win(data, 100) # Sliding window implimentation
  data = zscore(data) # Batch method implimentation 
  
  data = np.nan_to_num(data) 
  z_iter = iter(data) 
  numStreams = data.shape[1]
  
  ''' Initialise CDST Algorithm '''
  CDST_alg = CDST('F-FHST.A-SREboth', p, numStreams)
  
  ''' Main Loop '''
  for zt in z_iter:
    zt = zt.reshape(zt.shape[0],1)   # Convert to a column Vector 
    
    # Reset anomaly flag if last iteration flagged anomaly
    if np.any(CDST_alg.st['anomaly']):
      CDST_alg.st['anomaly'][:] = False 
  
    ''' Next Input Method '''
    CDST_alg.next_input(zt)
  
    '''Store data''' 
    # Calculate reconstructed data if needed for plotting visulisations
    st = CDST_alg.st
    CDST_alg.st['recon'] = np.dot(st['Q'][:,:st['r']],st['ht'][:st['r']]) 
    
    tracked_values = ['ht','e_ratio','r', 't_stat', 'SRE_dif_t', 'Ez', 'Eh', 'recon']
    #tracked_values = ['ht','e_ratio','r', 't_stat', 'SRE_dif_t', 'Ez', 'Eh', 'recon']
  
    CDST_alg.track_var(tracked_values, print_anom = 1)
    #CDST_alg.track_var()
  
  ''' Plot Results '''
  #CDST_alg.plot_res([data, 'ht', 't_stat'])
  #CDST_alg.plot_res([data, 'recon', 't_stat'])
  
  CDST_alg.plot_res([data, 'ht', 't_stat'], ynames =['Standardised Data', 'Hidden Variables', 'Test Statistic'])
  CDST_alg.plot_res([data, 'SRE_dif_t', 't_stat'], ynames =['Standardised Data', 'SRE', 'Test Statistic'])
  CDST_alg.plot_res([raw_data, data, 't_stat'], ynames =['Raw Data', 'Standardised Data', 'Test Statistic'])