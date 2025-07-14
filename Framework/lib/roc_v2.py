# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 12:43:29 2023

@author: Tahasanul Abraham
"""

#%% Initialization of Libraries and Directory

import sys, os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0,grandparentdir)

grandgrandparentdir = os.path.dirname(grandparentdir)
sys.path.insert(0,grandgrandparentdir)

import numpy as np


class roc:
    def __init__(self, deci=5, dual=False):
        self.deci = deci
        self.dual = dual

    # ─────────────────────────────────────────────────────────────
    # common big-H core  → returns K and the per-class numerators
    # ─────────────────────────────────────────────────────────────
    def _big_h_core(self, detectors: dict):
        v_labels, v_scores, v_ious = [], [], []

        # explode multi-label detectors → virtual singletons
        for det in detectors.values():
            if isinstance(det, dict):
                if self.dual is True:
                    for lbl, vals in det.items():
                        if not (isinstance(vals, (list, tuple)) and len(vals) == 2):
                            raise ValueError(f"Detection for label {lbl} must be [score, iou]. Got: {vals}")
                        s, iou = vals
                        v_labels.append(lbl)
                        v_scores.append(float(s))
                        v_ious.append(float(iou))
                else:
                    for lbl, s in det.items():
                        v_labels.append(lbl)
                        v_scores.append(float(s))

        if self.dual is True:
            if len(v_scores) < 2 or len(v_ious) < 2:
                raise ValueError("need ≥2 singleton masses")
                
            # soft-max
            H1_xi = np.exp(v_scores)
            H1_xi /= H1_xi.sum()
    
            # outer product
            H1 = np.outer(H1_xi, H1_xi)
            
            # soft-max
            H2_xi = np.exp(v_ious)
            H2_xi /= H2_xi.sum()
    
            # outer product
            H2 = np.outer(H2_xi, H2_xi)
            
            H_unnormalized = np.multiply(H1, H2)
            
            H = H_unnormalized / np.sum(H_unnormalized)
            
            
        else:
            if len(v_scores) < 2:
                raise ValueError("need ≥2 singleton masses")
                
            # soft-max
            xi = np.exp(v_scores)
            xi /= xi.sum()
    
            # outer product
            H = np.outer(xi, xi)

        # accumulate support & conflict
        m_tilde = {lbl: 0.0 for lbl in set(v_labels)}
        K = 0.0
        for i, li in enumerate(v_labels):
            for j, lj in enumerate(v_labels):
                prod = H[i, j]
                if li == lj:
                    m_tilde[li] += prod
                else:
                    K += prod
        return K, m_tilde

    # ─────────────────────────────────────────────────────────────
    # Dempster: renormalise by (1-K)
    # ─────────────────────────────────────────────────────────────
    def perform_ds(self, **kargs):
        K, m_tilde = self._big_h_core(kargs)
        if K >= 1.0:
            raise ValueError("total conflict K = 1")
    
        masses = {lbl: round(v / (1.0 - K), self.deci)
                  for lbl, v in m_tilde.items()}
    
        K_rounded = round(K, self.deci)               # ← round K, too
        return K_rounded, masses

    # ─────────────────────────────────────────────────────────────
    # Yager: keep conflict separate, no renormalisation
    # ─────────────────────────────────────────────────────────────
    def perform_ygr(self, **kargs):
        K, m_tilde = self._big_h_core(kargs)
    
        K_rounded = round(K, self.deci)       # keep K separately
        masses = {lbl: round(v, self.deci)    # round singletons
                  for lbl, v in m_tilde.items()}
        masses['Conflicts'] = K_rounded       # return rounded K
    
        return K_rounded, masses

    
#%% Standalone Run

if __name__ == "__main__":
    
    
    det1 = {'Person': 0.92}
    det2 = {'Person': 0.90}
    det3 = {'Dog':    0.94}
    multi1 = {'Person': 0.92, 'Cat': 0.92}
    multi2 = {'Person': 0.90, 'Tiger': 0.92}
    single = {'Dog': 0.94}
    
    r = roc()                      # constructor signature unchanged
    
          
    print("----------------DEMPSTER----------------")
    K, m = r.perform_ds(m1=det1, m2=det2, m3=det3)
    # K ≈ 0.4488,  m {'Person': 0.7902, 'Dog': 0.2098}
    print("Singletone Mode:")
    print("results:")
    print(f"k = {K}")
    print(*(f"{k}: {v}" for k, v in m.items()), sep="\n")
    print() 
    
    K, m = r.perform_ds(d1=multi1, d2=multi2, d3=single) 
    
    print("Multi Mode:")
    print("results:")
    print(f"k = {K}")
    print(*(f"{k}: {v}" for k, v in m.items()), sep="\n")
    print("----------------------------------------")
    print()
    print("-----------------YAGER------------------")
    K, m = r.perform_ygr(m1=det1, m2=det2, m3=det3)
    # K ≈ 0.4488,  m {'Person': 0.7902, 'Dog': 0.2098}
    print("Singletone Mode:")
    print("results:")
    print(f"k = {K}")
    print(*(f"{k}: {v:.3f}" for k, v in m.items()), sep="\n")
    print() 
    
    K, m = r.perform_ygr(d1=multi1, d2=multi2, d3=single) 
    
    print("Multi Mode:")
    print("results:")
    print(f"k = {K}")
    print(*(f"{k}: {v:.3f}" for k, v in m.items()), sep="\n")
    print()
    print("----------------------------------------")