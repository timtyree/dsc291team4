#!/usr/bin/env python3
# coding: utf-8
# ## measuring memory latency 
from numpy import *
import numpy as np
from numpy.random import rand
import datetime
import time
from os.path import isfile,isdir
from os import mkdir
import os
from lib.measureRandomAccess import measureRandomAccess
# from lib.PlotTime import PlotTime

def measureRandomAccessMemBlocks(sz,k=1000,batch=100):
    """Measure the distribution of random accesses in computer memory.

    :param sz: size of memory block.
    :param k: number of times that the experiment is repeated.
    :param batch: The number of locations poked in a single experiment (multiple pokes performed using numpy, rather than python loop)
    :returns: (_mean,std,T):
              _mean = the mean of T
              _std = the std of T
              T = a list the contains the times of all k experiments
    :rtype: tuple

    """
    # Prepare buffer.
    A=np.zeros(sz,dtype=np.int8)
            
    # Read and write k*batch times from/to buffer.
    sum=0; sum2=0
    T=np.zeros(k)
    for i in range(k):
        if (i%100==0): print('\r',i, end=' ')
        loc=np.int32(rand(batch)*sz)
        t=time.time()
        x=A[loc]
        A[loc]=loc
        d=(time.time()-t)/batch
        T[i]=d
        sum += d
        sum2 += d*d
    _mean=sum/k; var=(sum2/k)-_mean**2; _std=np.sqrt(var)
    return (_mean,_std,T)

if __name__ == "__main__":
    current_datetime = str(datetime.datetime.now()).split('.')[0]
    cwd = os.getcwd()
    #initialize
    n=100 # size of single block (1MB)
    k=100000;  # number of repeats
    m_list=[1, 10000000, 20000000, 30000000, 40000000, 50000000, 60000000]# size of file in blocks
    #m_legend=['1B', '10MB', '20MB', '30MB', '40MB', '50MB', '60MB']
    # print('n=%d, k=%d, m_list='%(n,k),m_list)
    log_root='./logs'
    if not isdir(log_root): mkdir(log_root)
    # TimeStamp=str(int(time.time()))
    TimeStamp = current_datetime.replace(' ', '_')
    log_dir=log_root+'/'+TimeStamp
    mkdir(log_dir)
    os.chdir(log_dir)
    exec_dir = log_dir
    # get_ipython().run_line_magic('cd', '$log_dir')

    #open stats.txt
    # stat=open('stats.txt','w')
    # def tee(line):
    #     print(line)
    #     stat.write(line+'\n')
    # tee('current_datetime is {}'.format(current_datetime))
    # tee('input parameters are n:%d, k:%d\n'%(n,k))
    creation_stats = {'current_datetime':current_datetime,
    'batch':n, 'k':k}
    out_dict = {}
    out_dict['creation_stats']=creation_stats

    Tmem=[]
    TFile=[]
    Random_pokes=[]
    L=len(m_list)
    _mean=zeros([L])   #0: using disk, 1: using memory
    _std=zeros([L])
    TMem=[0]*L

    # tee('m_i,m,memory__mean,memory__std,memory__largest__mean')
    for m_i in range(L):
        m=m_list[m_i]
        # print('Memory array %d Bytes'%m)
        out = measureRandomAccessMemBlocks(m,k=k,batch=n)
        (_mean[m_i],_std[m_i],TMem[m_i]) = out
        TMem[m_i].sort()
        # mlm = np.median(TMem[m_i][-100:])
        # tee(f"{m_i},{m:.0e},{_mean[m_i]:0.6e},{_std[m_i]:0.6e},{mlm:0.6e}")
        
        # tee('\rMemory pokes _mean='+str(_mean[m_i])+', Memory _std='+str(_std[m_i]))
        Random_pokes.append({'m_i':m_i,
                            'm':m,
                            'memory__mean': _mean[m_i],
                            'memory__std': _std[m_i],
                            'memory__largest': TMem[m_i][-100:],
                    })
                            # 'memory__largest__median': ),
    # tee(Random_pokes)
    out_dict['Random_pokes'] = Random_pokes
    import pickle as pk
    # os.chdir(exec_dir)
    # with open('multi_processing_stats.pkl','wb') as stats_pkl:
    with open(f'out.{TimeStamp}.pkl','wb') as stats_pkl:
        pk.dump(out_dict,stats_pkl,protocol=pk.HIGHEST_PROTOCOL)
    os.chdir(cwd)
    # stat.close()


    





