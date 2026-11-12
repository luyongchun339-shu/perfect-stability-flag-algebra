from __future__ import annotations
import numpy as np
from scipy import linalg

def lower_indices(d):
    c,r=np.triu_indices(d)
    return r,c

def svec(matrix):
    r,c=lower_indices(len(matrix))
    return matrix[r,c]*np.where(r==c,1.0,np.sqrt(2.0))

def smat(vector,d):
    r,c=lower_indices(d);matrix=np.zeros((d,d))
    matrix[r,c]=vector/np.where(r==c,1.0,np.sqrt(2.0))
    matrix[c,r]=matrix[r,c]
    return matrix

def dense_counts(block,index):
    d=len(block['flags']);matrix=np.zeros((d,d),dtype=np.int64)
    for i,j,w in block['matrices_upper'][index]: matrix[i,j]=matrix[j,i]=w
    return matrix

def numerical_blocks(model,face=None):
    matrices=[];bases=[];triangular=[]
    for bi,block in enumerate(model['blocks']):
        d=len(block['flags'])
        if face is None:
            O=np.eye(d);R=np.eye(d)
        else:
            record=face['blocks'][bi]
            N=np.array(record['N'],dtype=float)/record['N_denominator']
            O,R=linalg.qr(N,mode='economic')
        B=np.stack([O.T@(dense_counts(block,j)/block['denominator'])@O
                    for j in range(len(model['partitions']))])
        matrices.append(B);bases.append(O);triangular.append(R)
        print('NUMERICAL_BLOCK',block['type'],'dimension',O.shape[1],flush=True)
    return matrices,bases,triangular
