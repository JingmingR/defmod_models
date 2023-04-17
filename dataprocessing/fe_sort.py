#!/usr/bin/env python
# %matplotlib qt
import numpy as np
import os,sys
from itertools import islice
import glob as glb
import warnings
warnings.filterwarnings("ignore")
import h5py
from functools import reduce
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse

def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s with %s type' %(key,type(item)))

def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

def printname(name):
    print (name)


ap=argparse.ArgumentParser()
ap.add_argument('-f') # input file (ex.,"P3DX")
ap.add_argument('-dmn') # dimension
ap.add_argument('-slip') # fault slip (containing fault data)
ap.add_argument('-xflt') # fault intersection
ap.add_argument('-dyn') # dynamic simulation
ap.add_argument('-rsf') # rsf/slip weakening
ap.add_argument('-poro') # poroelastic
ap.add_argument('-nobs') # number of observation
ap.add_argument('-seis') # number of observation

name_sol = ap.parse_args().f
dmn = int(ap.parse_args().dmn)
slip = int(ap.parse_args().slip)
xflt = int(ap.parse_args().xflt)
dyn = int(ap.parse_args().dyn)
rsf = int(ap.parse_args().rsf)
p = int(ap.parse_args().poro)
nobs = int(ap.parse_args().nobs)
seis = int(ap.parse_args().seis)

# default
alpha=0
clean=False

# File sets
if dyn: 
    log_file     = name_sol+".log"
    log_dyn_file = name_sol+"_dyn.log"
    dat_log = np.loadtxt(log_file, delimiter=" ",skiprows=1, unpack=False, dtype=np.uint32)[:]
    dat_log_dyn = np.loadtxt(log_dyn_file, delimiter=" ",skiprows=1, unpack=False, dtype=np.uint32)
    dt_dyn = np.loadtxt(log_dyn_file, dtype=np.float)[0]
    dt = np.loadtxt(log_file, delimiter=" ", usecols=[0], unpack=False, dtype=np.float)[0]
    files_obs    = sorted(glb.glob(name_sol+"_ofe_*.h5"))

    print (np.loadtxt(log_file, delimiter=" ",skiprows=1, unpack=False, dtype=np.uint32)[:])

if seis:
    print ('Merge observation grid/data...')
    # Read from ofe.h5 files
    idx=[]
    for file_obs in files_obs:
        h5 = h5py.File(file_obs, 'r') 
        ids=h5['idx'][:]
        print (ids)
        idx.append(ids)
    nsta=h5['uu_sta'].shape[-1]
    ndyn=h5['uu_dyn'].shape[-1]
    # nobs=len(reduce(np.union1d,idx))
    ocoord=np.empty(shape=[nobs,dmn],dtype=np.float)
    if dmn==2:
        dat_qs=np.empty(shape=[nobs,2+p+3,nsta],dtype=np.float)
        dat_seis=np.empty(shape=[nobs,2,ndyn],dtype=np.float)
    else:
        dat_qs=np.empty(shape=[nobs,3+p+6,nsta],dtype=np.float)
        dat_seis=np.empty(shape=[nobs,3,ndyn],dtype=np.float)
    for file_obs,i in zip(files_obs,range(len(files_obs))):
        h5 = h5py.File(file_obs,'r')
        ocoord[idx[i]-1,:]=h5['obs_x'][:,:]
        dat_qs[idx[i]-1,:dmn+p,:]=h5['uu_sta'][:,:,:]
        dat_qs[idx[i]-1,dmn+p:,:]=h5['st_sta'][:,:,:]
        dat_seis[idx[i]-1,:,:]=h5['uu_dyn'][:,:,:]

# For equivalent implicit dynamic solver
if alpha:
    log_dyn_file_alpha = name_sol+"-alpha_dyn.log"
    dt_alpha=np.loadtxt(log_dyn_file_alpha, dtype=np.float)[0]
    dat_log_dyn_alpha = np.loadtxt(log_dyn_file_alpha, delimiter=" ",skiprows=1, unpack=False, dtype=np.uint32)

    # Read alpha h5 files
    files_obs_alpha=sorted(glb.glob(name_sol+"-alpha_dyn_ofe_*.h5"))
    idx=[];
    for file_obs_alpha in files_obs_alpha:
        h5 = h5py.File(file_obs_alpha, 'r')
        ids=h5['idx'][:]
        idx.append(ids)
    ndyn=h5['uu_dyn'].shape[-1]
    dat_seis_alpha=np.empty(shape=[nobs,3,ndyn],dtype=np.float)
    for file_obs_alpha,i in zip(files_obs_alpha,range(len(files_obs_alpha))):
        h5 = h5py.File(file_obs_alpha, 'r')
        dat_seis_alpha[idx[i]-1,:,:]=h5['uu_dyn'][:,:,:]

if slip:
    # File sets
    if dyn:
        log_slip_file = name_sol+"_slip.log"  # 
        dt_slip = np.loadtxt(log_slip_file, dtype=np.float64)[0] # 1e-2 #
        dat_log_slip = np.loadtxt(log_slip_file, dtype=np.uint32)[1:] # 
        nframe=dat_log_slip[-1] # 


    files_slip = sorted(glb.glob(name_sol+"_flt_*.h5"))
    h5 = h5py.File(files_slip[0], 'r')
    nframe_qs = np.shape(h5['slip_sta'])[-1]
    print ('Merge seismic fault slip grid/data...')
    nfnd_all=0
    for f in files_slip:
        h5 = h5py.File(f, 'r')
        nfnd_all+=np.shape(h5['slip_sta'])[0]
    fcoord=np.empty(shape=[nfnd_all,dmn*3],dtype=np.float)
    if dyn:
        dat_slip_tmp=np.empty(shape=[nfnd_all,dmn,nframe],dtype=np.float)
        dat_trac_tmp=np.empty(shape=[nfnd_all,dmn+dmn,nframe],dtype=np.float) # +2
    h5 = h5py.File(files_slip[0], 'r')
    dat_slip_sta=np.empty(shape=[nfnd_all,dmn,nframe_qs],dtype=np.float)
    dat_trac_sta=np.empty(shape=[nfnd_all,dmn+p,nframe_qs],dtype=np.float)
    if (xflt):
        dat_slipx_sta=np.empty(shape=[nfnd_all,dmn,nframe_qs],dtype=np.float)
        dat_tracx_sta=np.empty(shape=[nfnd_all,dmn+p,nframe_qs],dtype=np.float)
    fopen=[];n_lmnd=[]
    j0=0
    for f in files_slip:
        h5 = h5py.File(f, 'r')
        j1=j0+len(h5['fault_x'])
        fcoord[j0:j1,:]=h5['fault_x']
        dat_slip_sta[j0:j1,:,:]=h5['slip_sta']
        dat_trac_sta[j0:j1,:,:]=h5['trac_sta']
        if (xflt):
            dat_slipx_sta[j0:j1,:,:]=h5['slipx_sta']
            dat_tracx_sta[j0:j1,:,:]=h5['tracx_sta']            
        if (dyn):
            dat_slip_tmp[j0:j1,:,:]=h5['slip_dyn']
            dat_trac_tmp[j0:j1,:,:]=h5['trac_dyn'] # why there is 4 dimension?
        j0=j1
        print ('fault patch ' + str(j0) +"/"+str(nfnd_all)+ " merged")
    nfnd=len(fcoord)

    # Equivalent g-alpha solver
    if alpha:
        log_slip_file_alpha = name_sol+"-alpha_slip.log"
        dat_log_slip_alpha = np.loadtxt(log_slip_file_alpha, dtype=np.uint32)[1:]
        dt_slip_alpha = np.loadtxt(log_slip_file_alpha, dtype=np.float64)[0]
        nframe=dat_log_slip_alpha[-1]
        dat_slip_alpha=np.empty(shape=[nfnd_all,dmn,nframe],dtype=np.float)
        files_slip_alpha = sorted(glb.glob(name_sol+"-alpha_fe_*.h5"))
        j0=0
        for f in files_slip_alpha:
            h5 = h5py.File(f, 'r')
            j1=j0+len(h5['fault_x'])
            dat_slip_alpha[j0:j1,:,:]=h5['slip_dyn']
            j0=j1

    if rsf==1:
        log_rsf_file=name_sol+"_rsf.log"
        try:
            dt_rsf=np.loadtxt(log_rsf_file,dtype=np.float)[0]
            pseudo=True
        except:
            pseudo=False
        if pseudo:
            print ('Merge pseudo time fault grid/data...')
            dat_log_rsf=np.loadtxt(log_rsf_file,dtype=np.uint32)[1:]
            files_rsf=[file_slip.replace('slip','rsf') for file_slip in files_slip]
            fcoord=np.empty(shape=[0,dmn+1],dtype=np.float)
            dat_rsf_tmp=np.empty(shape=[0,2],dtype=np.float)
            fopen=[];n_lmnd=[]
            for file_rsf in files_rsf: fopen.append(open(file_rsf))
            for f in fopen:
                n=int(np.genfromtxt(islice(f,1),delimiter=" ",unpack=False,dtype=np.uint32))
                fcoord=np.vstack((fcoord,np.genfromtxt(islice(f,n),delimiter=" ",unpack=False,dtype=np.float)))
                n_lmnd.append(n)
            fsort=np.argsort(fcoord[:,-1])
            fcoord=fcoord[fsort,:dmn]
            nfnd=sum(n_lmnd)
            nframe=dat_log_rsf[-1]
            for i in range(nframe):
                dat_tmp=np.empty(shape=[0,2],dtype=np.float)
                for f,j in zip(fopen,range(len(fopen))):
                    dat_tmp=np.vstack((dat_tmp,np.loadtxt(islice(f,n_lmnd[j]),unpack=False,dtype=np.float)))
                dat_rsf_tmp=np.vstack((dat_rsf_tmp,dat_tmp[fsort,:]))
                if np.remainder(i+1,24)*np.remainder(i+1,nframe)==0:
                    print ("frame " + str(i+1) +"/"+str(nframe)+ " merged")

# Sort fault slip by frame
if slip and rsf==1 and pseudo: # RSF pseudo time
    dat_rsf=np.empty(shape=[nfnd,2,len(dat_rsf_tmp)/nfnd], dtype=np.float)
    for i in range(len(dat_rsf_tmp)/nfnd):
        dat_rsf[:,:,i]=dat_rsf_tmp[i*nfnd:(i+1)*nfnd,:]

# +
# Sort seismic/slip data by event
dat_seis_sort={}
if alpha: dat_seis_alpha_sort={}
if slip:
    dat_slip_sort = {}
    dat_trac_sort = {}
    if alpha:
        dat_slip_alpha_sort={}
if dyn and len(dat_log_dyn.shape)==0:dat_log_dyn=[dat_log_dyn.item()]
if alpha:
    if len(dat_log_dyn_alpha.shape)==0: dat_log_dyn_alpha=[dat_log_dyn_alpha.item()]
        
if dyn:
    for i in range(len(dat_log_dyn)):
        if i==0:
            start=0
        else:
            start=dat_log_dyn[i-1]
        end=dat_log_dyn[i]
        if seis:
            if len(dat_log) == 2:
                dat_seis_sort['step '+str(dat_log[0])] = dat_seis[:,:,start:end] #
            else:
                dat_seis_sort['step '+str(dat_log[i,0])] = dat_seis[:,:,start:end] #
        if alpha:
            if i==0:
                start=0
            else:
                start=dat_log_dyn_alpha[i-1]
            end=dat_log_dyn_alpha[i]
            dat_seis_alpha_sort['step '+str(dat_log[i,0])] = dat_seis_alpha[:,:,start:end]
        if slip and dyn:
            if i==0:
                start=0
            else:
                start=dat_log_slip[i-1]
            end=dat_log_slip[i]
            if len(dat_log) == 2:
                dat_slip_sort['step '+str(dat_log[0])] = dat_slip_tmp[:,:,start:end]
                dat_trac_sort['step '+str(dat_log[0])] = dat_trac_tmp[:,:,start:end]
            else:
                dat_slip_sort['step '+str(dat_log[i,0])] = dat_slip_tmp[:,:,start:end]
                dat_trac_sort['step '+str(dat_log[i,0])] = dat_trac_tmp[:,:,start:end]
            if alpha:
                if i==0:
                    start=0
                else:
                    start=dat_log_slip_alpha[i-1]
                end=dat_log_slip_alpha[i]
                dat_slip_alpha_sort['step '+str(dat_log[i,0])] = dat_slip_alpha_tmp[:,:,start:end]
# -

# Sort rate state data by quasi-static time step.
if slip and rsf==1 and pseudo:
    dat_rsf_sort=[]
    for i in range(len(dat_log_rsf)):
        if i==0:
            start=0
        else:
            start=dat_log_rsf[i-1]
        dat_rsf_sort.append(dat_rsf[start:dat_log_rsf[i],:])

# # Store sorted data to .mat files
h5file = name_sol+'_fe.h5'

# +
# Save to .h5 file
mdict = {}
if seis:
    mdict={'obs_sta': dat_qs,
           'obs_dyn': dat_seis_sort,
           'crd_obs': ocoord,
           'dt': dt,
           'dt_dyn': dt_dyn,
           'log': dat_log,
           'log_dyn': np.array(dat_log_dyn)}
    
if dyn:
    mdict={'dt': dt,
           'dt_dyn': dt_dyn,
           'log': dat_log,
           'log_dyn': np.array(dat_log_dyn)}
    
if slip:
    mdict['crd_flt'] = fcoord
    mdict['trac_sta'] = dat_trac_sta
    mdict['slip_sta'] = dat_slip_sta
    if xflt:
        dat_slipx_sta
        mdict['tracx_sta'] = dat_tracx_sta
        mdict['slipx_sta'] = dat_slipx_sta
    if dyn:
        mdict['dt_slip'] =  dt_slip
        mdict['slip_dyn'] = dat_slip_sort
        mdict['log_slip'] = dat_log_slip
        mdict['trac_dyn'] = dat_trac_sort
    if alpha:
        mdict['dt_slip_alpha'] = dt_slip_alpha
        mdict['slip_alpha'] = dat_slip_alpha_sort
    if rsf and pseudo:
        tmp_arr = np.zeros((len(dat_rsf_sort),), dtype=np.object)
        for i in range(len(tmp_arr)):
            tmp_arr[i] = dat_rsf_sort[i]
        dat_rsf_sort = tmp_arr
        mdict['dt_rsf'] = dt_rsf
        mdict['log_rsf'] = dat_log_rsf
        mdict['rsf'] = dat_rsf_sort
if alpha:
    mdict['dt_dyn_alpha'] = dt_alpha
    mdict['obs_alpha'] = dat_seis_alpha_sort
print ('writing to '+ h5file +'...')
save_dict_to_hdf5(mdict, h5file)
print (h5file + ' created')
# -
