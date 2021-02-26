#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 12:03:25 2021

@author: julien
"""



import numpy as np
from itertools import product



def phi_u_spectrum(x,u,k):
    
    """
    Inputs:
    x: DNA sequence (with ATGC letters)
    u: subsequence of interest
    
    Output:
    occur: number of occurences of u in x
    """
    occur=0
    
    for i in range(len(x)-k+1):
        subsequence = x[i:i+k]
        if subsequence==u:
            occur+=1
    return occur
            
def phi(x,k):
    """
    ------------------------------------------------------------------------------
    Compute the non-linear projection of DNA sequence on a vector of dimension 4**k
    ------------------------------------------------------------------------------
    Inputs:
    x: DNA sequence (with ATGC letters)
    k: length of subsequences considered
    
    Output:
    feat_vec: embedding of input x
    """
    feat_vec=np.zeros(4**k)
    for idx,substring_tuple in enumerate(product('ACGT', repeat=k)):
        u=''.join(substring_tuple)
        feat_vec[idx]=phi_u_spectrum(x,u,k)
        
    return feat_vec













