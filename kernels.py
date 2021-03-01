#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 12:03:25 2021

@author: julien
"""



import numpy as np
from itertools import product
from fuzzysearch import find_near_matches
from Bio import pairwise2


def phi_u_spectrum(x,u,k):
    
    """
    Inputs:
    x: DNA sequence (with ATGC letters)
    u: subsequence of interest
    k: len(u) : length of subsequence
    
    Output:
    occur: number of occurences of u in x
    """
    occur=0
    
    for i in range(len(x)-k+1):
        subsequence = x[i:i+k]
        if subsequence==u:
            occur+=1
    return occur


def phi_u_substring(x,u,tolerance):
    
    """
    Inputs:
    x: DNA sequence (with ATGC letters)
    u: subsequence of interest
    
    Output:
    occur: number of occurences of u in x with tolerance max_l_dist 
    """
    
    occur=0
    
    matches=find_near_matches(u, x, max_l_dist=tolerance)
    distances=np.array([match.dist for match in matches])
    for importance in range(tolerance):
        occur+=(tolerance-importance)*(distances==importance).sum()
    
    return occur

def seq2subseq(sequence,k):
    """
    Inputs:
    sequence: DNA sequence (with ATGC letters)
    u: subsequence of interest
    k: len(u) : length of subsequence
    
    Output:
    occur: number of occurences of u in x
    """
    
    subseqs=[]
    for i in range(len(sequence)-k+1):
        subseqs.append(sequence[i:i+k])
    return subseqs

def subseq2idx(subseq):
    
    """
    Inputs:
    subsequence: DNA subsequence (with ATGC letters)
    
    Output:
    idx: corresponding idx
    """
    idx=0
    for power,letter in enumerate(subseq):
        if letter=='A':
            idx+=3*4**power
        elif letter=='T':
            idx+=2*4**power
        elif letter=='G':
            idx+=1*4**power
        elif letter!='C':
            print("letter is not in ATGC")
            #do nothing
    return idx

            
def phi(x,k,kernel='spectrum_efficient',tolerance=1):
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
    
    if kernel=='spectrum':
        for idx,substring_tuple in enumerate(product('ACGT', repeat=k)):
            u=''.join(substring_tuple)
            feat_vec[idx]=phi_u_spectrum(x,u,k)
    
    elif  kernel=='substring':
        for idx,substring_tuple in enumerate(product('ACGT', repeat=k)):
            u=''.join(substring_tuple)
            feat_vec[idx]=phi_u_substring(x,u,tolerance)
            
    elif kernel=='spectrum_efficient':
        subseqs=seq2subseq(x,k)
        for subseq in subseqs:
            feat_vec[subseq2idx(subseq)]+=1
                                
    return feat_vec


def K(seq1,seq2,k=5,tolerance=1,kernel='bio'):
    """
    ------------------------------------------------------------------------------
    Compute directly the kernel K(seq1,seq2)
    The projection is not made explicit
    ------------------------------------------------------------------------------
    
    Inputs:
    seq1,seq2: DNAs sequences (with ATGC letters)
    k: length of subsequences considered
    
    Output:
    Kernel function (a real number)
    """
    result=0
    
    if kernel=='mismatch':
        subseqs1=seq2subseq(seq1,k)
        for subseq1 in subseqs1:
            result+=len(find_near_matches(subseq1,seq2, max_l_dist=tolerance))
            
        subseqs2=seq2subseq(seq2,k)
        for subseq2 in subseqs2:
            result+=len(find_near_matches(subseq2,seq1, max_l_dist=tolerance))
    
    elif kernel=='bio':
        result = pairwise2.align.globalxx(seq1,seq2,score_only=True)
        
    return result
    









