#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import h5py
import math
import os, sys
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
sys.path.append('/home/jingmingruan/ownCloud/dataprocessing')
# sys.path.append('/home/jingmingruan/TUD/DeepNL/cloud/dataprocessing')
from plot_3D_simple45 import *


# # Load data

# In[3]:


dyn = 1
seis = 0
vline = 1.0
axis = 1
mask = 0.6
delta = 0
xyaxis = 1
projection=True
vline = 0.8

Zeerijp  = plot3D("/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/30_lessdp2/Buijze3D_fe.h5",dyn=dyn,seis=seis)
tri_file = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/30/flt_tri_nob.npy"


# In[4]:


if dyn:
    print (Zeerijp.dat_trac_sort.keys())
    print (Zeerijp.dat_log)
    print (Zeerijp.dat_log_dyn)


# In[5]:


xlim0 = 1.5-0.2 # 1.5 0.2
xlim1 = 2.5-0.2 # 2.5 0.8
zlim1 = -2.7
zlim0 = -3.3
step_list = [4,10,12,16]

step_ = [18] # begin from 0
step_dyn = [18]


# In[5]:


font = {'family' : 'serif',
        'size'   : 20}
# plt.style.use('seaborn-pastel') 
matplotlib.rc('font', **font)
matplotlib.rc('image', cmap='jet')
# https://matplotlib.org/stable/tutorials/colors/colormaps.html


# In[6]:


Zeerijp.plot_static_tri(filename=tri_file,tstep_list=[20],xyaxis=xyaxis,axis=3,
                        delta=1,mask=0,edgecolor=1,xlim0=xlim0,xlim1=xlim1,zlim0=zlim0,zlim1=zlim1)


# In[101]:


levels = np.linspace(0,1,51)
levels = np.concatenate((levels,np.array([1.01])))
print (levels)


# In[10]:


# incremental 
Zeerijp.plot_induced_static_tri_group(filename=tri_file,tstep_list=step_,xyaxis=xyaxis,
                        delta=1,mask=mask,edgecolor=0,xlim0=xlim0,xlim1=xlim1,zlim0=zlim0,zlim1=zlim1,interp=1)


# In[11]:


Zeerijp.plot_induced_static_tri_group(filename=tri_file,tstep_list=step_,xyaxis=xyaxis,
                        delta=0,mask=mask,edgecolor=0,xlim0=xlim0,xlim1=xlim1,zlim0=zlim0,zlim1=zlim1,interp=1)


# In[88]:


Zeerijp.plot_induced_static_tri_group(filename=tri_file,tstep_list=step_,xyaxis=xyaxis,
                        delta=0,mask=mask,edgecolor=0,xlim0=xlim0,xlim1=xlim1,zlim0=zlim0,zlim1=zlim1,interp=1)


# In[14]:


Zeerijp.plot_induced_static_tri_group(filename=tri_file,tstep_list=step_,xyaxis=xyaxis,
                        delta=0,mask=mask,edgecolor=0,xlim0=xlim0,xlim1=xlim1,zlim0=zlim0,zlim1=zlim1,interp=1)


# In[15]:


Zeerijp.plot_induced_static_tri_group(filename=tri_file,tstep_list=step_,xyaxis=xyaxis,
                        delta=0,mask=mask,edgecolor=0,xlim0=xlim0,xlim1=xlim1,zlim0=zlim0,zlim1=zlim1)


# # Random plot for try

# In[7]:


font = {'family' : 'serif',
        'size'   : 20}
# plt.style.use('seaborn-pastel') 
matplotlib.rc('font', **font)
matplotlib.rc('image', cmap='seismic')
# https://matplotlib.org/stable/tutorials/colors/colormaps.html


# In[6]:


# 30-60

p0 = np.array([1.768,-2.800])
p1 = np.array([2.000,-3.100])
p2 = np.array([1.922,-2.800])
p3 = np.array([2.000,-3.100])
points = np.vstack((p0,p1,p2,p3))
filename = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60_lessdp/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=[6,15],xyaxis=1,step_cri=[20, 23],delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                        xlim0=1.7,xlim1=2.0,zlim0=-3.05,zlim1=-2.88,
                        points=points,hlines=[])


# In[7]:


# 30-45
# [20, 51] # 51 for 45, 20 for 30 23 for 60
p0 = np.array([1.768,-2.800])
p1 = np.array([2.000,-3.100])
p2 = np.array([1.866,-2.800])
p3 = np.array([2.000,-3.100])
points = np.vstack((p0,p1,p2,p3))
filename = "/media/jingmingruan/BACKUP/TUD/delftblue/3D/3D_mfvaryingoffset/intersection_angle/45_001mpaseismic/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/delftblue/3D/3D_mfvaryingoffset/intersection_angle/45/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=[6,23],xyaxis=1,step_cri=[20, 51],delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                        xlim0=1.7,xlim1=2.0,zlim0=-3.05,zlim1=-2.88,
                        points=points,hlines=[])


# In[20]:


# 30-60
# new
p0 = np.array([1.768,-2.800])
p1 = np.array([2.000,-3.100])
p2 = np.array([1.922,-2.800])
p3 = np.array([2.000,-3.100])
points = np.vstack((p0,p1,p2,p3))
filename = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60_lessdp/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=[6,15],xyaxis=1,step_cri=[20, 23],delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                        xlim0=1.7,xlim1=2.0,zlim0=-3.05,zlim1=-2.88,
                        points=points,hlines=[])


# In[21]:


# 30-45
# new
p0 = np.array([1.768,-2.800])
p1 = np.array([2.000,-3.100])
p2 = np.array([1.866,-2.800])
p3 = np.array([2.000,-3.100])
points = np.vstack((p0,p1,p2,p3))
filename = "/media/jingmingruan/BACKUP/TUD/delftblue/3D/3D_mfvaryingoffset/intersection_angle/45_001mpaseismic/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/delftblue/3D/3D_mfvaryingoffset/intersection_angle/45/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=[6,23],xyaxis=1,step_cri=[20, 51],delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                        xlim0=1.7,xlim1=2.0,zlim0=-3.05,zlim1=-2.88,
                        points=points,hlines=[])


# In[8]:


# 30-60

p0 = np.array([1.768,-2.800])
p1 = np.array([2.000,-3.100])
p2 = np.array([1.922,-2.800])
p3 = np.array([2.000,-3.100])
points = np.vstack((p0,p1,p2,p3))
filename = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60_lessdp/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=[6,15],xyaxis=1,step_cri=[20, 23],delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                        xlim0=1.7,xlim1=2.0,zlim0=-3.05,zlim1=-2.88,
                        points=points,hlines=[])


# In[18]:


# 30-60

p0 = np.array([1.768,-2.800])
p1 = np.array([2.000,-3.100])
p2 = np.array([1.922,-2.800])
p3 = np.array([2.000,-3.100])
points = np.vstack((p0,p1,p2,p3))
filename = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60_lessdp/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=[6,15],xyaxis=1,step_cri=[20, 23],delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                        xlim0=1.7,xlim1=2.0,zlim0=-3.05,zlim1=-2.88,
                        points=points,hlines=[])


# In[23]:


# 30-60
p0 = np.array([1.768,-2.800])
p1 = np.array([2.000,-3.100])
p2 = np.array([1.922,-2.800])
p3 = np.array([2.000,-3.100])
points = np.vstack((p0,p1,p2,p3))
filename = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60_lessdp/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=[6,15],xyaxis=1,step_cri=[20, 23],delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                        xlim0=1.7,xlim1=2.0,zlim0=-3.05,zlim1=-2.88,
                        points=points,hlines=[])


# In[24]:


# 30-60
p0 = np.array([1.768,-2.800])
p1 = np.array([2.000,-3.100])
p2 = np.array([1.922,-2.800])
p3 = np.array([2.000,-3.100])
points = np.vstack((p0,p1,p2,p3))
filename = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60_lessdp/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=[6,15],xyaxis=1,step_cri=[20, 23],delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                        xlim0=1.7,xlim1=2.0,zlim0=-3.05,zlim1=-2.88,
                        points=points,hlines=[])


# In[27]:


# 30-60
p0 = np.array([1.768,-2.800])
p1 = np.array([2.000,-3.100])
p2 = np.array([1.922,-2.800])
p3 = np.array([2.000,-3.100])
points = np.vstack((p0,p1,p2,p3))
filename = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60_lessdp/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=[6,15],xyaxis=1,step_cri=[20, 23],delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                        xlim0=1.7,xlim1=2.0,zlim0=-3.05,zlim1=-2.88,
                        points=points,hlines=[])


# In[34]:


# 30-60
p0 = np.array([1.768,-2.800])
p1 = np.array([2.000,-3.100])
p2 = np.array([1.922,-2.800])
p3 = np.array([2.000,-3.100])
points = np.vstack((p0,p1,p2,p3))
filename = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60_lessdp/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=[6,15],xyaxis=1,step_cri=[20, 23],delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                        xlim0=1.7,xlim1=2.0,zlim0=-3.05,zlim1=-2.88,
                        points=points,hlines=[])


# In[35]:


# 30-60
p0 = np.array([1.768,-2.800])
p1 = np.array([2.000,-3.100])
p2 = np.array([1.922,-2.800])
p3 = np.array([2.000,-3.100])
points = np.vstack((p0,p1,p2,p3))
filename = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60_lessdp/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=[6,15],xyaxis=1,step_cri=[20, 23],delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                        xlim0=1.7,xlim1=2.0,zlim0=-3.05,zlim1=-2.88,
                        points=points,hlines=[])


# In[13]:


# 30-60

p0 = np.array([1.768,-2.800])
p1 = np.array([2.000,-3.100])
p2 = np.array([1.922,-2.800])
p3 = np.array([2.000,-3.100])
points = np.vstack((p0,p1,p2,p3))
filename = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60_lessdp/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=[6,15],xyaxis=1,step_cri=[20, 23],delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                       xlim0=1.7,xlim1=2.0,zlim0=-3.05,zlim1=-2.88,
                        points=points,hlines=[])


# In[8]:


Zeerijp.plot_induced_static_tri_group(filename=tri_file,tstep_list=[6],xyaxis=xyaxis,
                        delta=0,mask=mask,edgecolor=0,xlim0=1.6,xlim1=2.0,zlim0=-3.15,zlim1=-2.75,interp=1)


# In[8]:


Zeerijp.plot_induced_static_tri_group(filename=tri_file,tstep_list=[6],xyaxis=xyaxis,
                        delta=0,mask=mask,edgecolor=0,xlim0=1.6,xlim1=2.0,zlim0=-3.15,zlim1=-2.75,interp=1)


# In[10]:


# 30-60
p0 = np.array([1.768,-2.800])
p1 = np.array([2.000,-3.100])
p2 = np.array([1.922,-2.800])
p3 = np.array([2.000,-3.100])
points = np.vstack((p0,p1,p2,p3))
filename = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60_lessdp/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=[6,15],xyaxis=1,step_cri=[20, 23],delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                       xlim0=1.7,xlim1=2.0,zlim0=-3.05,zlim1=-2.88,
                        points=points,hlines=[])


# In[41]:


# 30-60
p0 = np.array([1.768,-2.800])
p1 = np.array([2.000,-3.100])
p2 = np.array([1.922,-2.800])
p3 = np.array([2.000,-3.100])
points = np.vstack((p0,p1,p2,p3))
filename = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60_lessdp/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=[6,15],xyaxis=1,step_cri=[20, 23],delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                       xlim0=1.7,xlim1=2.0,zlim0=-3.05,zlim1=-2.88,
                        points=points,hlines=[])


# In[29]:


# 30-60
filename = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60_lessdp/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=[6,15],xyaxis=1,step_cri=[20, 23],delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                       xlim0=1.7,xlim1=2.0,zlim0=-3.05,zlim1=-2.88,
                        points=[],hlines=[])


# In[25]:


# 30-60
filename = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60_lessdp/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=[6,15],xyaxis=1,step_cri=[20, 23],delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                       xlim0=1.7,xlim1=2.0,zlim0=-3.05,zlim1=-2.88,
                        points=[],hlines=[])


# In[13]:


# 30-60
filename = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60_lessdp/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=[6,15],xyaxis=1,step_cri=[20, 23],delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                       xlim0=1.7,xlim1=2.0,zlim0=-3.05,zlim1=-2.88,
                        points=[],hlines=[])


# In[32]:


# 30-60
filename = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60_lessdp/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=[6,15],xyaxis=1,step_cri=[20, 23],delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                       xlim0=1.7,xlim1=2.0,zlim0=-3.05,zlim1=-2.88,
                        points=[],hlines=[])


# In[24]:


# 30-60
filename = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60_lessdp/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=[6,15],xyaxis=1,step_cri=[20, 23],delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                       xlim0=1.6,xlim1=2.0,zlim0=-3.15,zlim1=-2.75,
                        points=[],hlines=[])


# In[18]:


# 30-60
filename = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60_lessdp/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=[6,15],xyaxis=1,step_cri=[20, 23],delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                       xlim0=1.6,xlim1=2.0,zlim0=-3.15,zlim1=-2.75,
                        points=[],hlines=[])


# In[10]:


# 30-60
filename = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60_lessdp/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=[6,15],xyaxis=1,step_cri=[20, 23],delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                       xlim0=1.6,xlim1=2.0,zlim0=-3.15,zlim1=-2.75,
                        points=[],hlines=[])


# In[27]:


filename = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60_lessdp/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=step_,xyaxis=1,delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                       xlim0=1.6,xlim1=2.0,zlim0=-3.1,zlim1=-2.75,
                        points=[],hlines=[])


# In[12]:


font = {'family' : 'serif',
        'size'   : 20}
# plt.style.use('seaborn-pastel') 
matplotlib.rc('font', **font)
matplotlib.rc('image', cmap='seismic')
# https://matplotlib.org/stable/tutorials/colors/colormaps.html


# In[19]:



filename = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60_lessdp/Buijze3D_fe.h5"
tri_file1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"
Zeerijp.diff_data_triint(filename=filename,step_=step_,xyaxis=1,delta=1,filename_tri0=tri_file,filename_tri1=tri_file1,
                       xlim0=1.6,xlim1=2.0,zlim0=-3.1,zlim1=-2.8,
                        points=[],hlines=[])


# # Plot data with TRI

# In[13]:


# A - B
p0 = np.array([1.768,-2.800])
p1 = np.array([2.000,-3.100])
points = np.vstack((p0,p1))
hlines = [-2.8,-3.0,-2.85,-3.05,-2.9,-3.1]
Zeerijp.diff_data(filename="/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/Buijze3D_fe.h5",filename_tri=tri_file,xyaxis=xyaxis,axis=2,
                  step=[23],delta=1,friction=0.6,xlim0=xlim0,xlim1=xlim1,zlim0=zlim0,zlim1=zlim1,points=points,hlines=hlines
                 )


# In[15]:


# A - B
p0 = np.array([1.768,-2.800])
p1 = np.array([2.000,-3.100])
points = np.vstack((p0,p1))
hlines = [-2.8,-3.0,-2.85,-3.05,-2.9,-3.1]
Zeerijp.diff_data(filename="/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/Buijze3D_fe.h5",filename_tri=tri_file,xyaxis=xyaxis,axis=2,
                  step=[23],delta=1,friction=0.6,xlim0=xlim0,xlim1=xlim1,zlim0=zlim0,zlim1=zlim1,points=points,hlines=hlines
                 )


# In[16]:


# A - B
p0 = np.array([1.768,-2.800])
p1 = np.array([2.000,-3.100])
points = np.vstack((p0,p1))
hlines = [-2.8,-3.0,-2.85,-3.05,-2.9,-3.1]
Zeerijp.diff_data(filename="/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/Buijze3D_fe.h5",filename_tri=tri_file,xyaxis=xyaxis,axis=2,
                  step=[23],delta=1,friction=0.6,xlim0=xlim0,xlim1=xlim1,zlim0=zlim0,zlim1=zlim1,points=points,hlines=hlines
                 )


# In[17]:


Zeerijp.plot_static_tri(filename=tri_file,tstep_list=step_,xyaxis=xyaxis,axis=0,
                        delta=1,mask=mask,edgecolor=1,xlim0=xlim0,xlim1=xlim1,zlim0=zlim0,zlim1=zlim1)


# In[19]:


Zeerijp.plot_static_list(xyaxis=xyaxis,vline=2.0, tstep_list=step_list, axis=0, delta=1, mask=0,zlim0=zlim0,zlim1=zlim1)


# In[37]:


Zeerijp.plot_static_tri(filename=tri_file,tstep_list=step_,xyaxis=xyaxis,axis=0,
                        delta=1,mask=0,edgecolor=1,xlim0=xlim0,xlim1=xlim1,zlim0=zlim0,zlim1=zlim1)


# In[9]:


Zeerijp.plot_static_tri(filename=tri_file,tstep_list=step_,xyaxis=xyaxis,axis=3,
                        delta=1,mask=0,edgecolor=1,xlim0=xlim0,xlim1=xlim1,zlim0=zlim0,zlim1=zlim1)


# In[16]:


Zeerijp.plot_static_tri(filename=tri_file,tstep_list=step_,xyaxis=xyaxis,axis=1,
                        delta=1,mask=0,edgecolor=1,xlim0=xlim0,xlim1=xlim1,zlim0=zlim0,zlim1=zlim1)


# In[17]:


Zeerijp.plot_static_tri(filename=tri_file,tstep_list=step_,xyaxis=xyaxis,axis=6,
                        delta=1,mask=0,edgecolor=1,xlim0=xlim0,xlim1=xlim1,zlim0=zlim0,zlim1=zlim1)


# In[18]:


Zeerijp.plot_static_tri(filename=tri_file,tstep_list=step_,xyaxis=xyaxis,axis=2,
                        delta=1,mask=0,edgecolor=1,xlim0=xlim0,xlim1=xlim1,zlim0=zlim0,zlim1=zlim1)


# In[19]:


Zeerijp.plot_static_tri(filename=tri_file,tstep_list=step_,xyaxis=xyaxis,axis=4,
                        delta=1,mask=0,edgecolor=0,xlim0=xlim0,xlim1=xlim1,zlim0=zlim0,zlim1=zlim1)


# In[20]:


Zeerijp.plot_static_tri(filename=tri_file,tstep_list=step_,xyaxis=xyaxis,axis=4,
                        delta=0,mask=0.,edgecolor=0,xlim0=xlim0,xlim1=xlim1,zlim0=zlim0,zlim1=zlim1)


# In[26]:


Zeerijp.plot_static_tri(filename=tri_file,tstep_list=step_,xyaxis=xyaxis,axis=6,
                        delta=0,mask=mask,edgecolor=0)


# In[27]:


Zeerijp.plot_static_tri(filename=tri_file,tstep_list=step_,xyaxis=xyaxis,axis=4,
                        delta=0,mask=0,edgecolor=0)


# In[28]:


# initial state
Zeerijp.plot_static_tri(filename=tri_file,tstep_list=step_,xyaxis=xyaxis,axis=2,
                        delta=delta,mask=0.,edgecolor=0)


# In[29]:


# initial state
Zeerijp.plot_static_tri(filename=tri_file,tstep_list=step_,xyaxis=xyaxis,axis=4,
                        delta=0,mask=0.4,edgecolor=0)


# In[30]:


for i in step_list:
    Zeerijp.plot_static_tri(filename=tri_file,tstep_list=[i],xyaxis=xyaxis,axis=1,
                            delta=0,mask=0.,edgecolor=0)


# ## Initial loadling

# In[31]:


# initial state
Zeerijp.plot_static_tri(filename=tri_file,tstep_list=[3],xyaxis=xyaxis,axis=0,
                        delta=0,mask=0.,edgecolor=0)


# In[32]:


# initial state
Zeerijp.plot_static_tri(filename=tri_file,tstep_list=[3],xyaxis=xyaxis,axis=1,
                        delta=0,mask=0.,edgecolor=0)


# In[33]:


# initial state
Zeerijp.plot_static_tri(filename=tri_file,tstep_list=[3],xyaxis=xyaxis,axis=2,
                        delta=0,mask=0.,edgecolor=0)


# In[34]:


# initial state
Zeerijp.plot_static_tri(filename=tri_file,tstep_list=[10],xyaxis=xyaxis,axis=4,
                        delta=0,mask=0.5,edgecolor=0)


# # Particle view

# In[35]:


Zeerijp.plot_static(tstep_list=step_,xyaxis=xyaxis,vline=vline,axis=0,delta=delta,projection=False,scatter=False,mask=mask,azimuth=10)


# In[36]:


Zeerijp.plot_static(tstep_list=step_,xyaxis=xyaxis,vline=vline,axis=1,delta=delta,projection=False,scatter=False,mask=mask,azimuth=10)


# In[37]:


Zeerijp.plot_static(tstep_list=step_,xyaxis=xyaxis,vline=vline,axis=2,delta=delta,projection=False,scatter=False,mask=mask,azimuth=10)


# In[38]:


# Zeerijp.plot_static(tstep_list=step_,xyaxis=xyaxis,vline=vline,axis=3,delta=True,projection=False,scatter=False,mask=mask,azimuth=10)


# In[39]:


Zeerijp.plot_static(tstep_list=[9],xyaxis=xyaxis,vline=vline,axis=4,delta=True,projection=False,scatter=False,mask=0.6,azimuth=10)


# # Projection & line interpretation

# In[40]:


Zeerijp.plot_static(tstep_list=step_,xyaxis=xyaxis,vline=False,axis=0,delta=delta,projection=projection,scatter=False,mask=mask)


# In[41]:


Zeerijp.plot_static(tstep_list=step_,xyaxis=xyaxis,vline=1.38,axis=2,delta=1,projection=projection,scatter=False,mask=mask)


# In[42]:


Zeerijp.plot_static(tstep_list=step_,xyaxis=xyaxis,vline=0.8,axis=2,delta=1,projection=projection,scatter=False,mask=mask)


# In[43]:


Zeerijp.plot_static(tstep_list=step_,xyaxis=xyaxis,vline=vline,axis=2,delta=delta,projection=projection,scatter=False,mask=mask)


# In[44]:


Zeerijp.plot_static(tstep_list=step_,xyaxis=xyaxis,vline=False,axis=3,delta=delta,projection=projection,scatter=False,mask=mask)


# In[45]:


Zeerijp.plot_static(tstep_list=step_,xyaxis=xyaxis,vline=False,axis=4,delta=delta,projection=projection,scatter=False,mask=0)


# In[46]:


Zeerijp.plot_static(tstep_list=step_,xyaxis=xyaxis,vline=0,axis=4,delta=delta,projection=projection,scatter=False,mask=0.4)


# In[47]:


# stress path ratio
Zeerijp.plot_static(tstep_list=step_,xyaxis=xyaxis,vline=0,axis=5,delta=delta,projection=projection,scatter=False,mask=1,zlim0=-2.8,zlim1=-3.0)


# In[48]:


Zeerijp.plot_static_list(xyaxis=xyaxis,vline=vline, tstep_list=step_list, axis=4, delta=delta, mask=0)


# In[49]:


Zeerijp.plot_static_list(xyaxis=xyaxis,vline=vline, tstep_list=step_list, axis=1, delta=delta, mask=0)


# In[50]:


Zeerijp.plot_static_list(xyaxis=xyaxis,vline=vline, tstep_list=step_list, axis=2, delta=delta, mask=0,)


# In[51]:


Zeerijp.plot_static_list(xyaxis=xyaxis,vline=vline, tstep_list=step_list, axis=5, delta=delta, mask=2.0)


# In[52]:


Zeerijp.plot_static_list(xyaxis=xyaxis,vline=vline, tstep_list=step_list, axis=5, delta=delta, mask=2.0, zlim0=-3.0, zlim1=-2.8)


# In[53]:


help(Zeerijp.plot_static_list)


# # Plot dynamic

# In[23]:


Zeerijp.plot_dyn_slip_tri_group(tri_file_1=tri_file,frame_list=[1,10,60],step='step 51',axis=1,vabs=0,
                                mask=0,xyaxis=xyaxis,vline=False,zlim0=zlim0,zlim1=zlim1,xlim0=xlim0-1,xlim1=xlim1+1,
                                edgecolor=0)


# In[20]:


Zeerijp.plot_dyn_slip_tri_group(tri_file_1=tri_file,frame_list=[1,10,60],step='step 36',axis=1,vabs=0,
                                mask=0,xyaxis=xyaxis,vline=False,zlim0=zlim0,zlim1=zlim1,xlim0=xlim0-1,xlim1=xlim1+1,
                                edgecolor=0)


# In[14]:


Zeerijp.plot_dyn_slip_tri_group(tri_file_1=tri_file,frame_list=[1,20,40],step='step 51',axis=1,vabs=0,
                                mask=0,xyaxis=xyaxis,vline=False,zlim0=zlim0,zlim1=zlim1,xlim0=xlim0-1,xlim1=xlim1+1,
                                edgecolor=0)


# In[8]:


Zeerijp.plot_dyn_slip_tri_group(tri_file_1=tri_file,frame_list=[1,10,20],step='step 37',axis=1,vabs=0,
                                mask=0,xyaxis=xyaxis,vline=False,zlim0=zlim0,zlim1=zlim1,xlim0=xlim0-1,xlim1=xlim1+1,
                                edgecolor=0)


# In[7]:


Zeerijp.plot_dyn_slip_tri_group(step='step 23',axis=1,vabs=0,
                                mask=0,xyaxis=xyaxis,vline=False,zlim0=zlim0,zlim1=zlim1,xlim0=xlim0-1,xlim1=xlim1+1,
                                edgecolor=0)


# In[6]:


Zeerijp.plot_dyn_slip_tri_group(step='step 24',axis=1,vabs=0,
                                mask=0,xyaxis=xyaxis,vline=False,zlim0=zlim0,zlim1=zlim1,xlim0=xlim0-1,xlim1=xlim1+1,
                                edgecolor=0)


# In[18]:


frame_list = np.arange(0,80,4)
for i in frame_list:
    Zeerijp.plot_dyn_slip_tri(filename=tri_file,step='step 24',axis=1,frame=i,vabs=0,
                        mask=0,xyaxis=xyaxis,vline=False,zlim0=zlim0,zlim1=zlim1,xlim0=xlim0-1,xlim1=xlim1+1,
                        edgecolor=0)


# In[56]:


# frame = 0
# Zeerijp.plot_dyn_slip_tri(filename=tri_file,step='step 19',axis=1,frame=frame,vabs=0,
#                     mask=0,xyaxis=xyaxis,vline=False,zlim0=-3.01,zlim1=-2.88,xlim0=0.0,xlim1=1.2,
#                     edgecolor=0)


# In[41]:


frame_list = np.arange(0,80,4)
for i in frame_list:
    Zeerijp.plot_dyn_slip_tri(filename=tri_file,step='step 24',axis=1,frame=i,vabs=0,
                        mask=0,xyaxis=xyaxis,vline=False,zlim0=zlim0,zlim1=zlim1,xlim0=xlim0-1,xlim1=xlim1+1,
                        edgecolor=0)


# In[35]:


frame_list = np.arange(0,80,4)
for i in frame_list:
    Zeerijp.plot_dyn_slip_tri(filename=tri_file,step='step 23',axis=1,frame=i,vabs=0,
                        mask=0,xyaxis=xyaxis,vline=False,zlim0=zlim0,zlim1=zlim1,xlim0=xlim0-1,xlim1=xlim1+1,
                        edgecolor=1)


# In[58]:



# plt.rcParams["animation.html"] = "jshtml"
# plt.rcParams['figure.dpi'] = 150  
# plt.ioff()
# fig = plt.figure(figsize=(15,15))


# def animate(i):
#     Zeerijp.plot_dyn_slip_ani(filename=tri_file,step='step 29',axis=0,frame=i,vabs=0,
#                         mask=0,xyaxis=xyaxis,vline=False,zlim0=zlim0,zlim1=zlim1,xlim0=xlim0,xlim1=xlim1,
#                         edgecolor=0)

# ani = matplotlib.animation.FuncAnimation(fig, animate, frames=np.arange(0,20,1))
# ani.save('./animation.gif', writer='imagemagick', fps=1)


# In[59]:


frame_list = np.arange(0,10,1)
for i in frame_list:
    Zeerijp.plot_dyn_slip_tri(filename=tri_file,step='step 29',axis=0,frame=i,vabs=0,
                        mask=0,xyaxis=xyaxis,vline=False,zlim0=zlim0,zlim1=zlim1,xlim0=xlim0,xlim1=xlim1,
                        edgecolor=1)


# In[ ]:


frame_list = np.arange(0,10,1)
for i in frame_list:
    Zeerijp.plot_dyn_slip_tri(filename=tri_file,step='step 23',axis=3,frame=i,vabs=0,
                        mask=0,xyaxis=xyaxis,vline=False,
                        edgecolor=1)


# In[ ]:




