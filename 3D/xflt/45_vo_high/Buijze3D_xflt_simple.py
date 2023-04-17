#!/usr/bin/env python
# coding: utf-8
# %%
# conventional fault intersection:
# normal speration of the fault plane.
# pos_node in the main fault includ FX
# pos_node in the sec fault not includ Fx
# Therefore: flt_ele (vecfsvd) on the FX in main fault 


# %%
import numpy as np
import h5py

import os, sys
import meshio
sys.path.append('/home/jingmingruan/ownCloud/dataprocessing')
# sys.path.append('/home/jingmingruan/dataprocessing')
# sys.path.append('/vardim/home/ruanjm/DeepNL/dataprocessing')
# sys.path.append('/home/jingmingruan/TUD/DeepNL/cloud/dataprocessing')
from inpfunc_MP import *
from plot_3D_Xflt import *

# %%
# convert msh to h5
# %run /home/jingmingruan/ownCloud/dataprocessing/Gmsh_h5_3D.py -m Zeerijp -e2 "triangle" -e3 "tetra" -mo 0 -km 0
# %run /home/jingmingruan/TUD/DeepNL/cloud/dataprocessing/Gmsh_h5_3D.py -m Zeerijp -e2 "triangle" -e3 "tetra" -mo 1 -km 0


# %%
# Zeerijp = plot3D("./Buijze3D_fe.h5",dyn=0,xflt=0,seis=0)

# Zeerijp.fcoord.shape
# print (Zeerijp.fcoord[:,3:6]) # fs
# print (Zeerijp.fcoord[:,6:9]) # fd
# np.cross(Zeerijp.fcoord[-1,3:6],Zeerijp.fcoord[-1,6:9]) #fn

# %%
pre_vec_fs_main = np.array([0.00000e+00, -1.00000e+00,  0.00000e+00])
pre_vec_fd_main = np.array([4.06737e-01,  0.00000e+00,  -9.13545e-01])
pre_vec_fn_main = np.array([9.1354500e-01, 0.00000e+00, 4.0673700e-01])

pre_vec_fs_aux = np.array([-7.07107000e-01, -7.07107000e-01,  0.00000e+00])
pre_vec_fd_aux = np.array([    0.00000e+00,     0.00000e+00, -1.00000e+00])
pre_vec_fn_aux = np.array([ 7.07107000e-01, -7.07107000e-01,  0.00000e+00])

# %%
h5_m1 = h5py.File('Buijze3D.h5', 'r')
name_out = "Buijze3D.inp"
save_checkpoint = 0
load_checkpoint = 1
conv_Xflt = 1 # turn this off if xflt
nmp = 40

init_sigmay = 1
load_vtk = 0

load_init_BC = 0
load_init_Xflt = 0

dbl_trac = 0
lock_wall = 0
rotate_angle = 0
make_patch = 0


# %%
phy_flt = []
phy_fb = []
phy_bound = [15,17,13,16,18,14] # -x -y- z +x +y +z


# %%
# Model header
dsp_hyb = 1
poro = 1
dim = 3
ndy = 28
grad_BC = 1

t = 3600*24*ndy; dt = 3600*24*1; nviz = 1 
if init_sigmay or load_vtk: t = 3600*24*1
# 1e-6 0.3m lc
dt_dyn=2e-5; t_dyn = 0.1; t_lim = 1; dsp=1; Xflt=2; rsf=0; v_bg=1E-12
bod_frc=1; hyb=1; nviz_dyn=int(1e-2/dt_dyn); nviz_wave=int(1e-2/dt_dyn); nviz_slip=int(1e-2/dt_dyn)
alpha= 0.; beta = 0.00125; rfrac = 0

init=0; init_BC=0; reform_BC=0
if load_init_BC: init_BC=1
if conv_Xflt: Xflt=1 # conventional intersection
if load_init_Xflt: Xflt=2

if poro:
    line1 = ["fault-p tet 36"]
else:
    line1 = ["fault tet 36"]
line3 = np.array([t, dt, nviz, dsp]).reshape(1,4)
line4 = np.array([t_dyn, dt_dyn, nviz_dyn, t_lim, dsp_hyb, Xflt, bod_frc,hyb,rsf,init,init_BC,reform_BC]).reshape(1,12)
if rsf==0:
    line5 = np.array([nviz_wave,nviz_slip]).reshape(1,2)
else:
    line5 = np.array([nviz_wave,nviz_slip,v_bg]).reshape(1,3)
line6 = np.array([alpha, beta]).reshape(1,2)


# %%
# inputs
print ('Extracting FE mesh...')
coord = (h5_m1['Domain/Vertices'][:,:3]) # input(km)
nnd_ori = len(coord)
rho=np.array([2400.,2400.,2400.])
E=np.array([1.5e10,1.5e10,1.5e10])
nu=np.array([0.15,0.15,0.15])
# E=rho*vs**2*(3*vp**2-4*vs**2)/(vp**2-vs**2)
# nu=(vp**2-2*vs**2)/2/(vp**2-vs**2)

E_dyn= E # Rick 2018 assumed dynamic E_dyn = 2 * E
nu_dyn=nu
K=np.array([9E-11,9E-11,9E-11,9E-11])/1.5E-4
vs = np.sqrt(E_dyn/2./rho/(1.+nu_dyn))
vp = np.sqrt(E_dyn*(1.-nu_dyn)/rho/(1.+nu_dyn)/(1.-2*nu_dyn))

#solid viscosity; power law
visc = 1E25 # solid viscosity [Pa·s] (visc optional)
r = 1. # viscoelastic power law parameter, (visc optional)
B=1. #Biot coef
phi=.20 #porosity
phir=.20 #(NAM,2016)
cf= 2.2e9 #fluid bulk modulus(water)
cfr=2.2e9 # 1.4e5 (Phung K.T. Nguyen1and Myung Jin Nam1,2,*) 180Mpa

mat = [[E[0], nu[0], visc, r, rho[0], K[0], B, phi, cf, 0,E_dyn[0],nu_dyn[0]],
       [E[1], nu[1], visc, r, rho[1], K[1], B, phir,cfr,0,E_dyn[1],nu_dyn[1]],
       [E[2], nu[2], visc, r, rho[2], K[2], B, phi, cf, 0,E_dyn[2],nu_dyn[2]]]
mat_typ = np.empty(shape = (0,1), dtype=np.uint32)

tet_node = (h5_m1['Domain/Cells'][:,1:])
print ('%d nodes, %d elements' %(nnd_ori, len(tet_node)))

HF=1.0
hhf = [[HF, HF, HF, HF, HF, HF, HF, HF, HF],
       [HF, HF, HF, HF, HF, HF, HF, HF, HF],
       [HF, HF, HF, HF, HF, HF, HF, HF, HF]]

#convert gmsh numbering to exodusII numbering
work = tet_node
work2 = work.T[[0,3,1,2]]
tet_node = work2.T


# %%
print (coord[:,0].min(),coord[:,0].max(),coord[:,0].min()-coord[:,0].max())
print (coord[:,1].min(),coord[:,1].max(),coord[:,1].min()-coord[:,1].max())
print (coord[:,2].min(),coord[:,2].max(),coord[:,2].min()-coord[:,2].max())


# %%
# h5_m2.visit(printname)
# Print info based on header input
if Xflt>1:
    print ("Expecting multiple faults and intersection(s)")


# %%
# Observation locations
ogrid=np.array([[ 1.0, 1.0, -0.1],
                [ 1.0, 2.0, -0.1],
                [ 1.0, 3.0, -0.1],
                [ 2.0, 1.0, -0.1],
                [ 2.0, 2.0, -0.1],
                [ 2.0, 3.0, -0.1],
                [ 3.0, 1.0, -0.1],
                [ 3.0, 2.0, -0.1],
                [ 3.0, 3.0, -0.1],
                [ 1.5, 1.5, -0.1],
                [ 1.5, 2.5, -0.1],
                [ 2.5, 1.5, -0.1],
                [ 2.5, 2.5, -0.1],
               ])


# %%
mat_typ = np.zeros((len(tet_node)),dtype=np.uint32).reshape(-1)
cell_mat = []
# ! input materal dict
options = {}

for i in range(1,5):
    options[i] = 1
for i in range(5,9):
    options[i] = 2
for i in range(9,13):
    options[i] = 3
    
for i in range(1,13): # debug: check the group number for the cell ! input
    print (i)
    cell_mat.append(h5_m1['Regions/group%d/Cell Ids' %(i)][:])

for i in range(len(cell_mat)): # bug: manually check the layers with multiple physical groups
    for j in range(len(cell_mat[i])):
        mat_typ[cell_mat[i][j]-1] = options[i+1]


# %%
# Duplicate the fault nodes (off-boundary) (CHANGE!) 
# ! input
flt_tri = np.empty(shape=[0, 3], dtype=np.uint32) # should be flt_lin but made same as 3D
flt_tri_bdr = np.empty(shape=[0, 2], dtype=np.uint32)
flt_tri_index = np.empty(shape=(0), dtype=np.uint32) # should be flt_lin but made same as 3D
flt_tri_seg = []
for i in range(19,34): # input first main fault then aux fault (or just one fault)
    felem = (h5_m1['Regions/group%d/Vertex Ids' %(i)][:,1:])
    flt_tri_seg.append(felem)
    flt_tri = np.vstack((flt_tri, felem))
    flt_tri_index = np.hstack((flt_tri_index,i*np.ones(len(felem),dtype=np.uint32)))
# flt_node = np.unique(flt_tri)

for i in [37,38]: # input fault boundary node & intersection
    if i == 10: # one-node intersection
        felem = (h5_m1['Regions/group%d/Vertex Ids' %(i)][1:])
    else:
        felem = (h5_m1['Regions/group%d/Vertex Ids' %(i)][:,1:])
    flt_tri_bdr = np.vstack((flt_tri_bdr, felem))
flt_xbdr_node = np.unique(flt_tri_bdr)

ft_pos_nodes_seg = []
ft_pos_nodes = np.empty(shape=[0, 3], dtype=np.uint32)
ft_pos_dir_main = np.empty(shape=[0,3], dtype=np.uint32)
ft_pos_dir_aux = np.empty(shape=[0,3], dtype=np.uint32)

# assigning the fault vector
ft_pos_dir_main = np.empty(shape=[0,3], dtype=np.uint32)
ft_pos_dir_main = np.empty(shape=[0,3], dtype=np.uint32)
ft_pos_dir_main = np.empty(shape=[0,3], dtype=np.uint32)
ft_pos_dir_main = np.empty(shape=[0,3], dtype=np.uint32)
ft_pos_dir_main = np.empty(shape=[0,3], dtype=np.uint32)
ft_pos_dir_main = np.empty(shape=[0,3], dtype=np.uint32)

for i in range(19,29): #input first main fault 
    felem = (h5_m1['Regions/group%d/Vertex Ids' %(i)][:,1:])
    flt_node = np.unique(felem)
    flt_node = flt_node[np.in1d(flt_node,flt_xbdr_node,invert=True)]
    ft_pos_nodes_seg.append(np.unique(flt_node))
    ft_pos_nodes = np.vstack((ft_pos_nodes, felem))
    ft_pos_dir_main = np.vstack((ft_pos_dir_main, felem))
ft_pos_dir_main[:,:] = 0 # positive direction for the main fault x y z 

for i in range(29,34): #input  then aux fault 
    felem = (h5_m1['Regions/group%d/Vertex Ids' %(i)][:,1:])
    flt_node = np.unique(felem)
    flt_node = flt_node[np.in1d(flt_node,flt_xbdr_node,invert=True)]
    ft_pos_nodes_seg.append(np.unique(flt_node))
    ft_pos_nodes = np.vstack((ft_pos_nodes, felem))
    ft_pos_dir_aux = np.vstack((ft_pos_dir_aux, felem))
ft_pos_dir_aux[:,:] = 0 # positive direction for the aux fault x y z 
    
ft_pos_dir = np.vstack((ft_pos_dir_main, ft_pos_dir_aux))

# indice main 
# indice aux
# do the same thing as the pos_dir

print (ft_pos_nodes.shape)    
print (ft_pos_dir.shape)
flt_node, indices = np.unique(ft_pos_nodes, return_index=True)
flt_pos = ft_pos_dir.flatten()[indices]

ft_pos_nodes = flt_node[np.in1d(flt_node,flt_xbdr_node,invert=True)]
ft_pos_dir = flt_pos[np.in1d(flt_node,flt_xbdr_node,invert=True)]

print (ft_pos_dir.shape)
print (ft_pos_nodes.shape)

# get rid of competley boundary flt_tri
mask = np.in1d(flt_tri, flt_xbdr_node)
mask_rsh = mask.reshape((-1,3))
work_sum = np.sum(mask_rsh,axis=1)
flt_tri = flt_tri[np.where(work_sum<3)]


# %%
print (indices.shape)
print (flt_pos.shape)
print (ft_pos_nodes.shape)


# %%
# plotting positive direction
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(5,5),constrained_layout=True)
plt.scatter(coord[ft_pos_nodes-1,0],ft_pos_dir)
plt.show()
plt.savefig("pos_dir.png")


# %%
# # saving for tri plot
# mask = np.in1d(flt_tri, flt_xbdr_node)
# mask_rsh = mask.reshape((-1,3))
# work_sum = np.sum(mask_rsh,axis=1)
# flt_tri_save = flt_tri[np.where(work_sum==0)]
# np.save("flt_tri_nob",flt_tri_save)


# ### Skip to load checkpoint if needed!!!

# %%
# testing auto detecting negative side
# Debug: auto Replace the negative side 
# ! input postive direction

# fault segment based on the interesction line (list in list)
# input first main fault then aux fault (include all branches!!)
fault_seg = [[19,20,21,22,23],[24,25,26,27,28], [29,30,31,32,33],[34,35,36]]

if save_checkpoint:

    model = inpclass(tet_node,coord,flt_tri,ft_pos_nodes,"tet",pos_dir=ft_pos_dir)
    model.Crack3D_MP(mp=nmp)
        
    # find out the element contain the fault elements
    # model.SearchFltele3D_MP(mp=nmp) # comment this to save time if assign vec_fs...
        
    # auto replace intersecting nodes
    if Xflt>1:
        ft_xinit_nodes = np.unique(h5_m1['Regions/group38/Vertex Ids'][1:])
        ft_xinit_nodes_bdr = np.unique(h5_m1['Regions/group37/Vertex Ids'][1:])
        ft_xinit_nodes = ft_xinit_nodes[np.in1d(ft_xinit_nodes,ft_xinit_nodes_bdr,invert=True)]

        flt_tri_all = np.empty(shape=[0, 3], dtype=np.uint32) # should be flt_lin but made same as 3D
        flt_tri_index_all = np.empty(shape=(0), dtype=np.uint32) # should be flt_lin but made same as 3D
        flt_tri_seg_all = []

        for i in fault_seg: # input first main fault then aux fault (include all branches!!) (based on segment +/-)
            felem_seg = np.empty(shape=[0, 3], dtype=np.uint32)
            for j in i:
                felem = (h5_m1['Regions/group%d/Vertex Ids' %(j)][:,1:])
                felem_seg = np.vstack((felem_seg, felem))
                flt_tri_all = np.vstack((flt_tri_all, felem))
                flt_tri_index_all = np.hstack((flt_tri_index_all,j*np.ones(len(felem),dtype=np.uint32)))
            flt_tri_seg_all.append(felem_seg)
            
        # tet_node,coord,ft_x_nodes = Xflt3D(tet_node,coord,flt_tri_all,flt_tri_seg_all,ft_xinit_nodes)
        model.Xflt3D(flt_tri_all,flt_tri_seg_all,ft_xinit_nodes)
    else:
        model.ft_x_nodes = np.empty(shape=(0), dtype=np.uint32)
        
    if conv_Xflt:
        ft_xinit_nodes = np.unique(h5_m1['Regions/group38/Vertex Ids'][1:])
        ft_xinit_nodes_bdr = np.unique(h5_m1['Regions/group37/Vertex Ids'][1:])
        ft_xinit_nodes = ft_xinit_nodes[np.in1d(ft_xinit_nodes,ft_xinit_nodes_bdr,invert=True)]

        flt_tri_all = np.empty(shape=[0, 3], dtype=np.uint32) # should be flt_lin but made same as 3D
        flt_tri_index_all = np.empty(shape=(0), dtype=np.uint32) # should be flt_lin but made same as 3D
        flt_tri_seg_all = []

        for i in fault_seg: # input first main fault then aux fault (include all branches!!) (based on segment +/-)
            felem_seg = np.empty(shape=[0, 3], dtype=np.uint32)
            for j in i:
                felem = (h5_m1['Regions/group%d/Vertex Ids' %(j)][:,1:])
                felem_seg = np.vstack((felem_seg, felem))
                flt_tri_all = np.vstack((flt_tri_all, felem))
                flt_tri_index_all = np.hstack((flt_tri_index_all,j*np.ones(len(felem),dtype=np.uint32)))
            flt_tri_seg_all.append(felem_seg)
            
        # tet_node,coord,ft_x_nodes = Xflt3D(tet_node,coord,flt_tri_all,flt_tri_seg_all,ft_xinit_nodes)
        model.Xflt3D_conv(flt_tri_all,flt_tri_seg_all,ft_xinit_nodes)

    ft_x_nodes = model.ft_x_nodes
    
    if rotate_angle>0:
        model.rotate_model(rotate_angle)
    # ========================================================================
    # model.Vecfsn3D_MP(mp=nmp) # comment this to save time if assign vec_fs...
    # ========================================================================
    model.nnd = len(model.coord)
    model.nxfnd = len(model.ft_x_nodes) * 2 # cross-link constraint is double the equations than the normal
    model.nfnd = len(model.ft_pos_nodes) + model.nxfnd
    model.vec_fn = np.zeros(((model.nfnd), 3), dtype=float)
    model.vec_fs = np.zeros(((model.nfnd), 3), dtype=float)
    model.vec_fd = np.zeros(((model.nfnd), 3), dtype=float)
    # main fault
    model.vec_fs[:,:] = pre_vec_fs_main
    model.vec_fd[:,:] = pre_vec_fd_main
    model.vec_fn[:,:] = pre_vec_fn_main
    # aux fault
    model.vec_fs[:,:] = pre_vec_fs_aux
    model.vec_fd[:,:] = pre_vec_fd_aux
    model.vec_fn[:,:] = pre_vec_fn_aux

    if model.nxfnd>0: 
        model.vec_fs[-1,:] = 0
        model.vec_fn[-1,:] = 0
        model.vec_fd[-1,:] = 0
    
    if Xflt>1:
        # input aux fault (all the branches)
        fault_seg_aux = [[29,30,31,32,33],[34,35,36]]
        # calculate the vec on the intersection based on the auxillary faults.
        flt_tri_aux = np.empty(shape=[0, 3], dtype=np.uint32)
        for i in fault_seg_aux: # input first main fault then aux fault (include all branches!!) (based on segment +/-)
            felem_seg = np.empty(shape=[0, 3], dtype=np.uint32)
            for j in i:
                felem = (h5_m1['Regions/group%d/Vertex Ids' %(j)][:,1:])
                felem_seg = np.vstack((felem_seg, felem))
            flt_tri_aux = np.vstack((flt_tri_aux, felem_seg))
        model.VecXfsn3D_X(flt_tri_aux)
        # print (model.vecx_fn)
        # print (model.vecx_fs)
        # print (model.vecx_fd)
        
        # input main fault
        fault_seg_main = [[19,20,21,22,23],[24,25,26,27,28]]
        # calculate the vec on the intersection based on the auxillary faults.
        flt_tri_main = np.empty(shape=[0, 3], dtype=np.uint32)
        for i in fault_seg_main: # input first main fault then aux fault (include all branches!!) (based on segment +/-)
            felem_seg = np.empty(shape=[0, 3], dtype=np.uint32)
            for j in i:
                felem = (h5_m1['Regions/group%d/Vertex Ids' %(j)][:,1:])
                felem_seg = np.vstack((felem_seg, felem))
            flt_tri_main = np.vstack((flt_tri_main, felem_seg))        
        model.Vecfsn3D_X(flt_tri_main)
        print (model.vec_fn)
        print (model.vec_fs)
        print (model.vec_fd)
        
    if conv_Xflt:
        # input main fault
        fault_seg_main = [[19,20,21,22,23],[24,25,26,27,28]]
        # calculate the vec on the intersection based on the auxillary faults.
        flt_tri_main = np.empty(shape=[0, 3], dtype=np.uint32)
        for i in fault_seg_main: # input first main fault then aux fault (include all branches!!) (based on segment +/-)
            felem_seg = np.empty(shape=[0, 3], dtype=np.uint32)
            for j in i:
                felem = (h5_m1['Regions/group%d/Vertex Ids' %(j)][:,1:])
                felem_seg = np.vstack((felem_seg, felem))
            flt_tri_main = np.vstack((flt_tri_main, felem_seg))
        model.Vecfsn3D_X(flt_tri_main)
        # print (model.vec_fn)
        # print (model.vec_fs)
        # print (model.vec_fd)

    # forming fault constaint coeffecient (Q: should this be on the positive side?)
    # 3D fault constraint coefficient
    nnd = len(model.coord)
    
    if conv_Xflt:
        nxfnd = len(model.ft_x_nodes) * 1
    else:
        nxfnd = len(model.ft_x_nodes) * 2 # why multiplied by two? (constraint equation num * 2)
    
    nfnd = len(model.ft_pos_nodes) + nxfnd
    bnd_el = []
    # ! input (boundary elements) in the order of abs & tract  
    for i in phy_bound:  # in the order of -x -y- z +x +y +z
        bnd_el.append(h5_m1['Regions/group%d/Vertex Ids' %(i)][:,1:])
    model.BCeleside3D_MP(bnd_el,mp=nmp)
    


# ### Computationally heavy part Finished! 

# %%
# saving check point for parameter fixing
# tet_node, coord, ft_neg_nodes, flt_tet, ft_x_nodes
# vec_fs, vec_fd, vec_fn
# cell_map_bc
if save_checkpoint:
    print ("Saving checkpoint data...")
    fout_h5 = 'Zeerijp3D_checkpoint.h5'
    try:
        os.remove(fout_h5)
    except OSError:
        pass

    f = h5py.File(fout_h5, "w")
    grp = f.create_group('All')
    grp.create_dataset('tet_node', data=model.fe_node, dtype=np.int32)
    grp.create_dataset('coord', data=model.coord, dtype=np.float64)
    grp.create_dataset('ft_neg_nodes', data=model.ft_neg_nodes, dtype=np.int32)
    grp.create_dataset('ft_x_nodes', data=model.ft_x_nodes, dtype=np.int32)
    grp.create_dataset('vec_fs', data=model.vec_fs, dtype=np.float64)
    grp.create_dataset('vec_fd', data=model.vec_fd, dtype=np.float64)
    grp.create_dataset('vec_fn', data=model.vec_fn, dtype=np.float64)
    if Xflt>1:
        grp.create_dataset('vecx_fs', data=model.vecx_fs, dtype=np.float64)
        grp.create_dataset('vecx_fd', data=model.vecx_fd, dtype=np.float64)
        grp.create_dataset('vecx_fn', data=model.vecx_fn, dtype=np.float64)
    # grp.create_dataset('flt_el', data=model.flt_el, dtype=np.int32) # comment if assign vec_fs
    # grp.create_dataset('loc_tet', data=np.asarray(model.loc_tet), dtype=np.int32) # comment if assign vec_fs
    for i in range(len(model.cell_map_bc)):
        grp.create_dataset("BC_group%d" % ((i)), data=model.cell_map_bc[i], dtype=np.int32)
    f.close()
    print (fout_h5 + '...checkpoint created.')
    
    raise ValueError('Now load and run again!')
    
    
if load_checkpoint:
    print ("Loading check point data...")
    f_checkpoint = h5py.File('Zeerijp3D_checkpoint.h5', 'r')
    tet_node = f_checkpoint['All/tet_node'][:]
    coord = f_checkpoint['All/coord'][:]
    ft_neg_nodes = f_checkpoint['All/ft_neg_nodes'][:]
    ft_x_nodes = f_checkpoint['All/ft_x_nodes'][:]
    vec_fs = f_checkpoint['All/vec_fs'][:]
    vec_fd = f_checkpoint['All/vec_fd'][:]
    vec_fn = f_checkpoint['All/vec_fn'][:]
    if Xflt>1:
        vecx_fs = f_checkpoint['All/vecx_fs'][:]
        vecx_fd = f_checkpoint['All/vecx_fd'][:]
        vecx_fn = f_checkpoint['All/vecx_fn'][:]
    # vec_flt_el = f_checkpoint['All/flt_el'][:] # comment if assign vec_fs 
    # loc_tet = f_checkpoint['All/loc_tet'][:] # comment if assign vec_fs
    bnd_el = []
    # ! input (boundary elements) in the order of abs & tract  
    for i in phy_bound: 
        bnd_el.append(h5_m1['Regions/group%d/Vertex Ids' %(i)][:,1:])
    cell_map_bc = []
    for i in range(6):
        cell_map_bc.append(f_checkpoint['All/BC_group%d' % ((i))][:])
        
    nnd = len(coord)
    if conv_Xflt:
        nxfnd = len(ft_x_nodes) * 1
    else:
        nxfnd = len(ft_x_nodes) * 2 # why multiplied by two? (constraint equation num * 2)
    nfnd = len(ft_pos_nodes) + nxfnd
    


# %%
print (ft_x_nodes.shape)
print (ft_pos_nodes.shape)
print (vec_fs.shape)
print (nfnd)
print (nxfnd)
print (vec_fs.shape[0]-ft_pos_nodes.shape[0])
# # Friction paremters

# %%
print (ft_x_nodes)
print (ft_pos_nodes)

# %%
# input for fault parameters !!
if poro: perm = np.zeros((nfnd,1)) # set it to zero if impose offset pressure profile

# RSF 
if rsf==1:   
#     a_list = [0.00251,0.00251,0.00612,0.02195]  # a
#     b_list = [0.00201,0.00201,0.00368,0.01953]  # b
#     b0_list =[0.658,0.658,0.619,0.494] # reference friction coefficient
#     V0_list =[0.1, 0.1, 0.1, 0.1] # reference slip velocity
#     L_list = [0.00278,0.00278,0.02329,0.00379]  # characteristic slip distance    
    
    a_list = [0.015,0.015,0.015,0.015]  # a
    b_list = [0.019,0.019,0.019,0.019]  # b
    b0_list =[0.6,0.6,0.6,0.6] # reference friction coefficient
    V0_list =[1e-6,1e-6,1e-6,1e-6] # reference slip velocity
    L_list = [2e-4,2e-4,2e-4,2e-4]  # characteristic slip distance    
    
    a = .015*np.ones((nfnd,1)); b0=.6*np.ones((nfnd,1));V0=1E-6*np.ones((nfnd,1))
    b=.019*np.ones((nfnd,1)); L=.02*np.ones((nfnd,1))
    dtau0 = np.zeros((nfnd,1))
    coh = np.zeros((nfnd,1))
    dcoh = np.ones((nfnd,1))
    theta_init = np.empty((nfnd,1),dtype=np.float)
    st_init = np.zeros((nfnd,3))
    st_init_ini = np.zeros((nfnd,3))
    frc=np.ones((nfnd,1),dtype=np.uint32)

# slip weakening    
else:
    st_init = np.zeros((nfnd,3))
    frc=np.ones((nfnd,1),dtype=np.uint32)
    fc=0.6*np.ones((nfnd,1),dtype=np.float)
    fcd=0.45*np.ones((nfnd,1),dtype=np.float)
    dc=0.005*np.ones((nfnd,1),dtype=np.float)
    coh=np.zeros((nfnd,1))
    dcoh=np.ones((nfnd,1))
    biot=1. * np.ones((nfnd,1))


# %%
# change the array order
if save_checkpoint:
    abs_bc1  = model.cell_map_bc[0] # -x
    abs_bc2  = model.cell_map_bc[1] # -y
    abs_bc3  = model.cell_map_bc[2] # bot
    trac_el1 = model.cell_map_bc[3] # +x
    trac_el2 = model.cell_map_bc[4] # +y
    trac_el3 = model.cell_map_bc[5] # top
else:
    abs_bc1  = cell_map_bc[0] # -x
    abs_bc2  = cell_map_bc[1] # -y
    abs_bc3  = cell_map_bc[2] # bot
    trac_el1 = cell_map_bc[3] # +x
    trac_el2 = cell_map_bc[4] # +y
    trac_el3 = cell_map_bc[5] # top


# # Loading initial stress

# %%
from plot_3D import *
if load_init_BC:
    ini_stress = plot3D("./Buijze3D_fe.h5",dyn=0,seis=0)


# %%
# setting initial stress based on idcase
# Make the fault permeable at certain segment; add initial stress 
# permeable at certain depth as well the friction law
x_min = min(coord[:,0]); x_max = max(coord[:,0])
y_min = min(coord[:,1]); y_max = max(coord[:,1])
z_min = min(coord[:,2]); z_max = max(coord[:,2])

if init_BC:
    rho_g = 200
    rho_f = 1150
    P_init = 35e6
    nu = 0.15
    g = 9.80665
    top_tra = -2350 * 2020 * g/1e6

    Kx = [0.748, 0.748, 0.748] # 0.748
    Ky = [0.795, 0.795, 0.795] # 0.795

    j = 0
    cap_node = 0
    # overwrite the initial stress with given stress
    if conv_Xflt:
        st_ft_pos_nodes = np.hstack((ft_pos_nodes,ft_x_nodes[:,0]))
    else:
        st_ft_pos_nodes = ft_pos_nodes
    
    for node_pos, i in zip(st_ft_pos_nodes, range(len(st_ft_pos_nodes))):
        x = coord[node_pos-1,:][0]
        y = coord[node_pos-1,:][1]
        z = coord[node_pos-1,:][2]
        if load_init_BC:
            sts = ini_stress.dat_trac_sta[:,0,0][i]
            std = ini_stress.dat_trac_sta[:,1,0][i]
            stn = ini_stress.dat_trac_sta[:,2,0][i] # effective normal stress
            pressure = ini_stress.dat_trac_sta[:,3,0][i]
            tot_shear = np.sqrt(sts**2+std**2)
#       cap maximum shear stress with reference friction
        if rsf==1 and np.abs(tot_shear/stn)>max(b0_list):
            cap_node += 1
            ratio = max(b0_list)*np.abs(stn)/tot_shear
            sts  = sts*ratio
            std  = std*ratio
        if rsf==0 and np.abs(tot_shear/stn)>(fc[i]):
            cap_node += 1
            ratio = fc[i]*np.abs(stn)/tot_shear * 0.95
            sts  = sts*ratio
            std  = std*ratio
        if make_patch==1:
            if abs(y-2) > 0.2:
                ratio = 0.01*fc[i]*np.abs(stn)/tot_shear
                sts  = sts*ratio
                std  = std*ratio
            else:
                sts  = std*0.6
                std  = std
        st_init[i,:]=[sts,std,(stn-pressure)]
        if rsf: st_init_ini[i,:]=[sts,std,stn]
    print (cap_node)
    
if load_init_Xflt:
    flt_tri_bdr = np.empty(shape=[0, 2], dtype=np.uint32)
    for i in [37,38]:
        if i == 10: # one-node intersection
            felem = (h5_m1['Regions/group%d/Vertex Ids' %(i)][1:])
        else:
            felem = (h5_m1['Regions/group%d/Vertex Ids' %(i)][:,1:])
        flt_tri_bdr = np.vstack((flt_tri_bdr, felem))
    flt_xbdr_node = np.unique(flt_tri_bdr)

    ft_pos_nodes_aux = np.empty(shape=[0, 3], dtype=np.uint32)
    for i in range(29,34): #aux fault (branchone) 
        felem = (h5_m1['Regions/group%d/Vertex Ids' %(i)][:,1:])
        flt_node = np.unique(felem)
        flt_node = flt_node[np.in1d(flt_node,flt_xbdr_node,invert=True)]
        ft_pos_nodes_aux = np.vstack((ft_pos_nodes_aux, felem))

    flt_node = np.unique(ft_pos_nodes_aux)
    ft_pos_nodes_aux = flt_node[np.in1d(flt_node,flt_xbdr_node,invert=True)]
    
        
    print ("assgining initial stress on Xflt based on the two faults...")
    stx_init_main = st_init[len(ft_pos_nodes):,:]
    print ("Secondary fault...")
    bcoord = coord[ft_pos_nodes_aux-1,:]
    stx_init_aux = np.zeros((len(ft_x_nodes[:,0]),3))
    k = 0
    for j in ft_x_nodes[:,0]:
        bnode = coord[j-1,:]
    
        dis = np.linalg.norm(bcoord - np.dot(np.ones((len(bcoord), 1)),                np.array(bnode).reshape(1,3)), axis=1)
        row = np.argsort(dis)[0]
        bnode_nr = ft_pos_nodes_aux[row]
        bnode_index = np.argwhere(ft_pos_nodes==bnode_nr)[0][0]
        # print (bnode_nr)
        # print (bnode)
        # print (coord[bnode_nr-1])
        # print (st_init[bnode_index]/1e6)
        
        stx_init_aux[k,:] = st_init[bnode_index]
        k += 1

    # inital stress on the main fault
    print ("Main fault...")
    sts = ini_stress.dat_trac_sta[:,0,0]
    std = ini_stress.dat_trac_sta[:,1,0]
    stn = ini_stress.dat_trac_sta[:,2,0] # effective normal stress
    pressure = ini_stress.dat_trac_sta[:,3,0]

    bcoord = ini_stress.fcoord[:,:3]
    k = 0
    stx_init_main = np.zeros((len(ft_x_nodes[:,0]),3))
    for j in ft_x_nodes[:,0]:
        bnode = coord[j-1,:]
        dis = np.linalg.norm(bcoord - np.dot(np.ones((len(bcoord), 1)),                np.array(bnode).reshape(1,3)), axis=1)
        row = np.argsort(dis)[0]
        # print (bnode)
        # print (bcoord[row,:])
        # print ([sts[row]/1e6,std[row]/1e6,(stn-pressure)[row]/1e6])
        stx_init_main[k,:] = [sts[row],std[row],(stn-pressure)[row]]
        k += 1
        
    st_init[len(ft_pos_nodes):len(ft_pos_nodes)+len(ft_x_nodes),:] = stx_init_main[:,:]
    st_init[len(ft_pos_nodes)+len(ft_x_nodes):,:] = stx_init_main[:,:]


# %%
print (ft_x_nodes.shape)
print (ft_pos_nodes.shape)
print (nxfnd)
print (nfnd)
print (len(ft_pos_nodes))
# # define RSF parameters on fault nodes

# %%
# set up RSF firction
if rsf and init_BC:
    pos_id = np.arange(len(ft_pos_nodes))
    res_check = np.zeros(len(ft_pos_nodes))
    for j in tqdm(range(loc_tet.shape[0])):
        flt_el_work_id = loc_tet[j][0]
        flt_el_work = tet_node[flt_el_work_id]
        loc_pos_flt = pos_id[(np.in1d(ft_pos_nodes[:],flt_el_work))]
        mat_id = mat_typ[flt_el_work_id]

        a_work=a_list[mat_id-1]
        b_work=b_list[mat_id-1]
        b0_work=b0_list[mat_id-1]
        V0_work=V0_list[mat_id-1]
        V_work=v_bg
        L_work=L_list[mat_id-1]
        
        tau0 = np.sqrt(st_init_ini[loc_pos_flt,0]**2 + st_init_ini[loc_pos_flt,1]**2)
        sn0 = np.abs(st_init_ini[loc_pos_flt,2])
        theta_init_work = np.abs(L_work/V0_work*np.exp((a_work*np.log(2.*np.sinh(tau0/a_work/sn0))                                                           -b0_work-a_work*np.log(v_bg/V0_work))/b_work))
        for i in range(len(loc_pos_flt)):
            loc_work = loc_pos_flt[i]
            if res_check[loc_work]==0:
                a[loc_work] = a_work
                b[loc_work] = b_work
                b0[loc_work] = b0_work
                V0[loc_work] = V0_work
                L[loc_work] = L_work
                theta_init[loc_work,0] = theta_init_work[i]
            
        if mat_id == 3:
            res_check[loc_pos_flt] = 1
        frc[loc_pos_flt]=1


# # critical element size $h^*$

# %%


lbd=rho*(vp**2-2*vs**2); mu=rho*vs**2
if rsf==1:
    L_h=np.mean(L_list); a_h=np.mean(a_list); b_h=np.mean(b_list)
    sigma_e=max(map(abs,st_init_ini[:,2]))
    hstar=min(2*mu*L_h/(b_h-a_h)/np.pi/sigma_e)
    print ("Critical RSF distance h*=%0.3f m" %hstar)
Lc=min(dt_dyn*np.sqrt(E_dyn/rho))
print (("Critical element length h=%0.3f m" %Lc))


# %%


# plotting initial stress
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# tcoord = coord[ft_pos_nodes-1,:]
# zcnt = tcoord[:,2]
if Xflt>1:
    plot_pos_node = np.hstack((ft_pos_nodes,ft_x_nodes[:,0],ft_x_nodes[:,0]))
    tcoord = coord[plot_pos_node-1,:]
    zcnt = tcoord[:,2]
else:
    plot_pos_node = np.hstack((ft_pos_nodes,ft_x_nodes[:,0]))
    tcoord = coord[plot_pos_node-1,:]
    zcnt = tcoord[:,2]


fig = plt.figure(figsize=(15,5),constrained_layout=True)
spec2 = gridspec.GridSpec(ncols=3, nrows=1, left=0.05, right=0.98, wspace=0.05, figure=fig)
f2_ax1 = fig.add_subplot(spec2[0])
f2_ax1.scatter(zcnt,st_init[:,0]/1e6)

f2_ax2 = fig.add_subplot(spec2[1])
f2_ax2.scatter(zcnt,st_init[:,1]/1e6)

f2_ax1 = fig.add_subplot(spec2[2])
f2_ax1.scatter(zcnt,st_init[:,2]/1e6)

plt.savefig("st_init.png")


# %%


if poro:
    bc_typ = np.ones((nnd,4), dtype=np.int8)
else:
    bc_typ = np.ones((nnd,3), dtype=np.int8)

# input
# fixed nodes & sync pressure on the boudary and reservoir
bcx_nodes = bnd_el[0]
bcy_nodes = bnd_el[1]
bcz_nodes = bnd_el[2]

for node in bcx_nodes.reshape(-1):
    bc_typ[node - 1, 0] = 0
for node in bcy_nodes.reshape(-1):
    bc_typ[node - 1, 1] = 0
for node in bcz_nodes.reshape(-1):
    bc_typ[node - 1, 2] = 0

if dbl_trac or load_vtk:
    for node in bcx_nodes.reshape(-1):
        bc_typ[node - 1, 0] = 1
    for node in bcy_nodes.reshape(-1):
        bc_typ[node - 1, 1] = 1

if lock_wall:
    for node in bcx_nodes.reshape(-1):
        bc_typ[node - 1, 0] = 0
    for node in bcy_nodes.reshape(-1):
        bc_typ[node - 1, 1] = 0
        
if init_BC or init_sigmay:
    bcx_nodes = bnd_el[3]
    bcy_nodes = bnd_el[4]
    # bcz_nodes = bnd_el[5]

    for node in bcx_nodes.reshape(-1):
        bc_typ[node - 1, 0] = 0
    for node in bcy_nodes.reshape(-1):
        bc_typ[node - 1, 1] = 0
    # for node in bcz_nodes.reshape(-1):
    #     bc_typ[node - 1, 2] = 0
        
# Pressure sync (debug)
if poro:
    bc_typ[:, 3] = 2


# %%


# apply extrac boundary condition when using traction-pair boundary condition (only on the top)
if (dbl_trac or load_vtk) and not (lock_wall):
    i = 0
    for side in [1,4]:  # - + y
        bcy_nodes = bnd_el[side]
        bcoord = coord[bcy_nodes.reshape(-1)-1,:]
        bnode = [[(coord[:,0].min()+coord[:,0].max())/2, coord[:,1].min(), coord[:,2].max()]
                 ,[(coord[:,0].min()+coord[:,0].max())/2, coord[:,1].max(), coord[:,2].max()]]
        dis = np.linalg.norm(bcoord - np.dot(np.ones((len(bcoord), 1)),                np.array(bnode[i]).reshape(1,3)), axis=1)
        row = np.argsort(dis)[0]
        bnode_nr = bcy_nodes.reshape(-1)[row]
        print (bnode_nr)
        print (coord[bnode_nr-1])
        bc_typ[bnode_nr-1,0] = 0 
        i += 1
    i = 0    
    for side in [0,3]:  # - + x
        bcy_nodes = bnd_el[side]
        bcoord = coord[bcy_nodes.reshape(-1)-1,:]
        bnode = [[coord[:,0].min(), (coord[:,1].min()+coord[:,1].max())/2, coord[:,2].max()]
                 ,[coord[:,0].max(), (coord[:,1].min()+coord[:,1].max())/2, coord[:,2].max()]]
        dis = np.linalg.norm(bcoord - np.dot(np.ones((len(bcoord), 1)),                np.array(bnode[i]).reshape(1,3)), axis=1)
        row = np.argsort(dis)[0]
        bnode_nr = bcy_nodes.reshape(-1)[row]
        print (bnode_nr)
        print (coord[bnode_nr-1])
        bc_typ[bnode_nr-1,1] = 0
        i += 1

    bcc_typ = np.copy(bc_typ)


# %%


# (test) changing boundary conditions during run
bcc_typ = np.copy(bc_typ)
if reform_BC:
    bcx_nodes = bnd_el[3]
    bcy_nodes = bnd_el[4]
    # bcz_nodes = bnd_el[5]

    for node in bcx_nodes.reshape(-1):
        bcc_typ[node - 1, 0] = 0
    for node in bcy_nodes.reshape(-1):
        bcc_typ[node - 1, 1] = 0
    # for node in bcz_nodes.reshape(-1):
    #     bcc_typ[node - 1, 2] = 0


# %%


if poro:
    trac_bc1 = np.zeros(shape=[len(trac_el1), 6])
    trac_bc2 = np.zeros(shape=[len(trac_el2), 6])
    trac_bc3 = np.zeros(shape=[len(trac_el3), 6])
    if dbl_trac:
        trac_bc4 = np.zeros(shape=[len(abs_bc1), 6]) # -x 
        trac_bc5 = np.zeros(shape=[len(abs_bc2), 6]) # -y
else:
    trac_bc1 = np.zeros(shape=[len(trac_el1), 5])
    trac_bc2 = np.zeros(shape=[len(trac_el2), 5])
    trac_bc3 = np.zeros(shape=[len(trac_el3), 5])

#--------------gradient traction BC------------ 
if grad_BC:
    
    dep = -0.0; g = 9.80665; p0 = -23.6e6 #top depth, gravity constant, top pressuure
    rho = np.array(mat)[:,4]
    
#     p0 = (-1.2)*rho[0]*g*1000

    # density 
    rhox = np.array([rho[0],rho[1],rho[2]])
    rhoy = np.array([rho[0],rho[1],rho[2]])
    # layer boundaries
    formbdx = np.sort(np.array([-1.2,-2.85,-3.05,-5]))
    formbdy = np.sort(np.array([-1.2,-2.85,-3.05,-5]))
    # Horizontal vertical stress ratio
    Kx = [0.748, 0.748, 0.748, 0.748] # 0.748 
    Ky = [0.795, 0.795, 0.795, 0.795] # 0.795

    # uniform vertical traction on the top
    # trac_bc3[:,2] = -p0+dep*1E3*rho[0]*g 
    trac_bc3[:,2] = p0
    # gradient traction on the sides
    for el,i in zip(trac_el1,range(len(trac_bc1))):
        el_node = tet_node[el[0]-1,:] 
        tcoord = coord[el_node-1,:]
        zcnt = np.mean(tcoord,0)[2]
        sigma_x = Kx[0]*(p0-dep*1E3*rhox[0]*g) # east to west
        for j in range(len(formbdx)-1): # integrate in depth 
            sigma_x += -1*g*1E3*Kx[j]*rhox[j]*((zcnt<=formbdx[j])*(formbdx[j+1]-formbdx[j])+(zcnt>formbdx[j] and zcnt<=formbdx[j+1])*(formbdx[j+1]-zcnt))
        trac_bc1[i,0] = sigma_x
        #trac_bc1[i,0] = -2.8E7 #Kx[-1]*(-p0+zcnt*1E3*rho[-1]*g)
    for el,i in zip(trac_el2,range(len(trac_bc2))):
        el_node = tet_node[el[0]-1,:] 
        tcoord = coord[el_node-1,:]
        zcnt = np.mean(tcoord,0)[2]
        sigma_y = Ky[0]*(p0-dep*1E3*rhoy[0]*g) # south to north
        for j in range(len(formbdy)-1): # integrate in depth 
            sigma_y += -1*g*1E3*Ky[j]*rhoy[j]*((zcnt<=formbdy[j])*(formbdy[j+1]-formbdy[j])+(zcnt>formbdy[j] and zcnt<=formbdy[j+1])*(formbdy[j+1]-zcnt))
        trac_bc2[i,1] = sigma_y
        #trac_bc2[i,1] = 2.4E7 #Ky[-1]*(p0-zcnt*1E3*rho[-1]*g)
        
    if dbl_trac and init_BC==0:
        formbdx = np.sort(np.array([-1.2,-2.80,-3.00,-5]))
        formbdy = np.sort(np.array([-1.2,-2.80,-3.00,-5]))
        for el,i in zip(abs_bc1,range(len(trac_bc4))):
            el_node = tet_node[el[0]-1,:] 
            tcoord = coord[el_node-1,:]
            zcnt = np.mean(tcoord,0)[2]
            sigma_x = Kx[0]*(p0-dep*1E3*rhox[0]*g) # south to north
            for j in range(len(formbdx)-1): # integrate in depth 
                sigma_x += -1*g*1E3*Kx[j]*rhox[j]*((zcnt<=formbdx[j])*(formbdx[j+1]-formbdx[j])+(zcnt>formbdx[j] and zcnt<=formbdx[j+1])*(formbdx[j+1]-zcnt))
            trac_bc4[i,0] = -1*sigma_x
        for el,i in zip(abs_bc2,range(len(trac_bc5))):
            el_node = tet_node[el[0]-1,:] 
            tcoord = coord[el_node-1,:]
            zcnt = np.mean(tcoord,0)[2]
            sigma_y = Ky[0]*(p0-dep*1E3*rhoy[0]*g) # south to north
            for j in range(len(formbdy)-1): # integrate in depth 
                sigma_y += -1*g*1E3*Ky[j]*rhoy[j]*((zcnt<=formbdy[j])*(formbdy[j+1]-formbdy[j])+(zcnt>formbdy[j] and zcnt<=formbdy[j+1])*(formbdy[j+1]-zcnt))
            trac_bc5[i,1] = -1*sigma_y

    if poro: # time for applying traction
        trac_bc3[:,4] = 0.; trac_bc3[:,5] = 0. 
        trac_bc1[:,4] = 0.; trac_bc1[:,5] = 0. 
    else:
        trac_bc3[:,3] = 0.; trac_bc3[:,4] = 0.
        
#--------------uniform traction BC------------ 
else:
    trac_val=[-0E6, -0E6, -0E6] # xyz
    if poro:
        trac_bc1[:,0]=trac_val[0]; trac_bc1[:,4]=0.; trac_bc1[:,5]=0.
        trac_bc2[:,1]=trac_val[1]; trac_bc2[:,4]=0.; trac_bc2[:,5]=0.
        trac_bc3[:,2]=trac_val[2]; trac_bc3[:,4]=0.; trac_bc3[:,5]=0.
    else:
        trac_bc1[:,0] = trac_val[0]; trac_bc1[:,3] = 0.; trac_bc1[:,4] = 0
        trac_bc2[:,1] = trac_val[1]; trac_bc2[:,3] = 0.; trac_bc2[:,4] = 0
        trac_bc3[:,2] = trac_val[2]; trac_bc3[:,3] = 0.; trac_bc3[:,4] = 0


# %%


print (coord[bnd_el[0]-1][:,:,0].max()-coord[bnd_el[0]-1][:,:,0].min())
print (coord[bnd_el[1]-1][:,:,1].max()-coord[bnd_el[0]-1][:,:,1].min())
print (coord[bnd_el[2]-1][:,:,2].max()-coord[bnd_el[0]-1][:,:,2].min())


# %%


# loading vertical stress from vtk
abs_bc1  = cell_map_bc[0] # -x
abs_bc2  = cell_map_bc[1] # -y

Ky = [0.748, 0.748, 0.748, 0.748] # 0.748
Kx = [0.795, 0.795, 0.795, 0.795] # 0.795

if load_vtk:
    mesh = meshio.read('foo_00.vtk')
    vtk_coord = mesh.points
    ss_z = mesh.point_data['ss'][:,2]
    vtk_pressure = mesh.point_data['pressure']
    
    vtk_coord_bdr = vtk_coord[vtk_coord[:,0]==coord[:,0].max()]
    ss_z_bdr = ss_z[vtk_coord[:,0]==coord[:,0].max()]
    vtk_pressure_bdr = vtk_pressure[vtk_coord[:,0]==coord[:,0].max()]
    for el,i in zip(trac_el1,range(len(trac_bc1))):
        sigma_x = 0.
        el_node = tet_node[(el[0])-1,:] 
        tcoord = np.round(coord[el_node-1,:],10)
        tcoord = tcoord[tcoord[:,0]==coord[:,0].max()]
        if len(tcoord)==0: raise ValueError("Boundary node not found!")
        for j in tcoord:
            dis = np.linalg.norm(vtk_coord_bdr - np.dot(np.ones((len(vtk_coord_bdr), 1)),                     np.array(j).reshape(1,3)), axis=1)
            row = np.argsort(dis)[0]
            if dis[row]> 1e-3: print (dis[row],j,vtk_coord_bdr[row])
            sigma_x += ss_z_bdr[row]- vtk_pressure_bdr[row]
        trac_bc1[i,0] = Kx[0]*sigma_x/len(tcoord) # postive direction
        
    vtk_coord_bdr = vtk_coord[vtk_coord[:,1]==coord[:,1].max()]
    ss_z_bdr = ss_z[vtk_coord[:,1]==coord[:,1].max()]
    vtk_pressure_bdr = vtk_pressure[vtk_coord[:,1]==coord[:,1].max()]
    for el,i in zip(trac_el2,range(len(trac_bc2))):
        sigma_y = 0.
        el_node = tet_node[(el[0])-1,:] 
        tcoord = np.round(coord[el_node-1,:],10)
        tcoord = tcoord[tcoord[:,1]==coord[:,1].max()]
        if len(tcoord)==0: raise ValueError("Boundary node not found!")
        for j in tcoord:
            dis = np.linalg.norm(vtk_coord_bdr - np.dot(np.ones((len(vtk_coord_bdr), 1)),                     np.array(j).reshape(1,3)), axis=1)
            row = np.argsort(dis)[0]
            if dis[row]> 1e-3: print (dis[row],j,vtk_coord_bdr[row])
            sigma_y += ss_z_bdr[row]- vtk_pressure_bdr[row]
        trac_bc2[i,1] = Ky[0]*sigma_y/len(tcoord) # postive direction
    
    trac_bc4 = np.zeros(shape=[len(abs_bc1), 6]) # -x 
    trac_bc5 = np.zeros(shape=[len(abs_bc2), 6]) # -y
    
    vtk_coord_bdr = vtk_coord[vtk_coord[:,0]==coord[:,0].min()]
    ss_z_bdr = ss_z[vtk_coord[:,0]==coord[:,0].min()]
    vtk_pressure_bdr = vtk_pressure[vtk_coord[:,0]==coord[:,0].min()]
    for el,i in zip(abs_bc1,range(len(trac_bc4))):
        sigma_x = 0.
        el_node = tet_node[(el[0])-1,:] 
        tcoord = np.round(coord[el_node-1,:],10)
        tcoord = tcoord[tcoord[:,0]==coord[:,0].min()]
        if len(tcoord)==0: raise ValueError("Boundary node not found!")
        for j in tcoord:
            dis = np.linalg.norm(vtk_coord_bdr - np.dot(np.ones((len(vtk_coord_bdr), 1)),                     np.array(j).reshape(1,3)), axis=1)
            row = np.argsort(dis)[0]
            if dis[row]> 1e-3: print (dis[row],j,vtk_coord_bdr[row])
            sigma_x += ss_z_bdr[row]- vtk_pressure_bdr[row]
        trac_bc4[i,0] = -1*Kx[0]*sigma_x/len(tcoord) # postive direction
        
    vtk_coord_bdr = vtk_coord[vtk_coord[:,1]==0] # coord[:,1].min() !!!
    ss_z_bdr = ss_z[vtk_coord[:,1]==0]
    vtk_pressure_bdr = vtk_pressure[vtk_coord[:,1]==0] # coord[:,1].min() !!!
    for el,i in zip(abs_bc2,range(len(trac_bc5))):
        sigma_y = 0.
        el_node = tet_node[(el[0])-1,:] 
        tcoord = np.round(coord[el_node-1,:],10)
        tcoord = tcoord[tcoord[:,1]==0] #coord[:,1].min() !!!
        if len(tcoord)==0: raise ValueError("Boundary node not found!")
        for j in tcoord:
            dis = np.linalg.norm(vtk_coord_bdr - np.dot(np.ones((len(vtk_coord_bdr), 1)),                     np.array(j).reshape(1,3)), axis=1)
            row = np.argsort(dis)[0]
            if dis[row]> 1e-3: print (dis[row],j,vtk_coord_bdr[row])
            sigma_y += ss_z_bdr[row]- vtk_pressure_bdr[row]
        trac_bc5[i,1] = -1*Ky[0]*sigma_y/len(tcoord) # postive direction


# %%


if init_BC:
    trac_bc1[:,:]=0.
    trac_bc2[:,:]=0.
    trac_bc3[:,:]=0.
    
if init_sigmay:
    trac_bc1[:,0]=0.
    trac_bc2[:,1]=0.
    # trac_bc3[:,2]=-1.5e7

trac_el = np.vstack((trac_el1, trac_el2, trac_el3))
trac_bc = np.vstack((trac_bc1, trac_bc2, trac_bc3))

if dbl_trac or load_vtk:
    trac_el = np.vstack((trac_el, abs_bc1, abs_bc2))
    trac_bc = np.vstack((trac_bc, trac_bc4, trac_bc5))


# %%


# plotting traction
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
el_node = tet_node[trac_el1[:,0]-1,:] 
tcoord = coord[el_node-1,:]
zcnt = np.mean(tcoord,1)[:,2]

fig = plt.figure(figsize=(9,5),constrained_layout=True)
spec2 = gridspec.GridSpec(ncols=2, nrows=1, left=0.05, right=0.98, wspace=0.05, figure=fig)
f2_ax1 = fig.add_subplot(spec2[0])
f2_ax1.scatter(trac_bc1[:,0],zcnt,  alpha=0.6,cmap='jet')

el_node2 = tet_node[trac_el3[:,0]-1,:] 
tcoord2 = coord[el_node2-1,:]
zcnt2 = np.mean(tcoord2,1)[:,2]

f2_ax2 = fig.add_subplot(spec2[1])
f2_ax2.scatter(trac_bc3[:,2]/1e6,zcnt2,  alpha=0.6,cmap='jet')


if dbl_trac or load_vtk:
    fltt_node = tet_node[(abs_bc1[:,0].astype(int)-1),:] 
    tcoord = coord[fltt_node-1,:]
    zcnt = np.mean(tcoord,1)[:,2]
    f2_ax1.scatter(trac_bc4[:,0],zcnt,alpha=0.6,cmap='jet')
    
    fltt_node = tet_node[(abs_bc2[:,0].astype(int)-1),:] 
    tcoord = coord[fltt_node-1,:]
    zcnt = np.mean(tcoord,1)[:,2]
    f2_ax2.scatter(trac_bc5[:,1],zcnt,alpha=0.6,cmap='jet')


# f2_ax2.scatter(st_init[:,0]/st_init[:,1],coord_fltele,  alpha=0.6,cmap='jet')
# f2_ax2.set_xlim(e6, 5e6)
# f2_ax2.set_ylim( -5.0, -2.6)
plt.savefig("trac1.png")


# %%


# plotting traction
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
el_node = tet_node[trac_el1[:,0]-1,:] 
tcoord = coord[el_node-1,:]
zcnt = np.mean(tcoord,1)[:,2]

fig = plt.figure(figsize=(9,5),constrained_layout=True)
spec2 = gridspec.GridSpec(ncols=2, nrows=1, left=0.05, right=0.98, wspace=0.05, figure=fig)
f2_ax1 = fig.add_subplot(spec2[0])
f2_ax1.scatter(trac_bc1[:,0],zcnt,  alpha=0.6,cmap='jet')

el_node2 = tet_node[trac_el2[:,0]-1,:]
tcoord2 = coord[el_node2-1,:]
zcnt2 = np.mean(tcoord2,1)[:,2]

f2_ax2 = fig.add_subplot(spec2[1])
f2_ax2.scatter(trac_bc2[:,1],zcnt2,  alpha=0.6,cmap='jet')


if dbl_trac or load_vtk:
    fltt_node = tet_node[(abs_bc1[:,0].astype(int)-1),:]
    tcoord = coord[fltt_node-1,:]
    zcnt = np.mean(tcoord,1)[:,2]
    f2_ax1.scatter(trac_bc4[:,0],zcnt,alpha=0.6,cmap='jet')
    
    fltt_node = tet_node[(abs_bc2[:,0].astype(int)-1),:]
    tcoord = coord[fltt_node-1,:]
    zcnt = np.mean(tcoord,1)[:,2]
    f2_ax2.scatter(trac_bc5[:,1],zcnt,alpha=0.6,cmap='jet')


# f2_ax2.scatter(st_init[:,0]/st_init[:,1],coord_fltele,  alpha=0.6,cmap='jet')
# f2_ax2.set_xlim(e6, 5e6)
# f2_ax2.set_ylim( -5.0, -2.6)
plt.savefig("trac2.png")


# %%


# absorbing bc (el,side)
abs_bc1 = np.hstack((abs_bc1, np.ones((len(abs_bc1),1))))
abs_bc2 = np.hstack((abs_bc2, 2*np.ones((len(abs_bc2),1))))
abs_bc3 = np.hstack((abs_bc3, 3*np.ones((len(abs_bc3),1))))
abs_bc4 = np.hstack((trac_el1, np.ones((len(trac_el1),1))))
abs_bc5 = np.hstack((trac_el2, 2*np.ones((len(trac_el2),1))))
abs_bc6 = np.hstack((trac_el3, 3*np.ones((len(trac_el3),1))))
# abs_bc6 can be free surface 
abs_bc = np.vstack((abs_bc1, abs_bc2, abs_bc3, abs_bc4, abs_bc5, abs_bc6))


# %%


# # nodal force/source input

fnode_bc = np.empty((0,7))


# %%


# Total length of constraint function
NCF_s2m = np.array([])
neqNCF=(dim+1)*len(NCF_s2m)
if poro:
    neqFT=dim*nfnd + sum(perm)
else:
    neqFT=dim*nfnd
neq = neqNCF+neqFT 
print ('%d NCF and %d fault constraint equations.' %(neqNCF,neqFT))


# %%


# Export to Defmod .inp file input
fout = name_out
print ('Write to ' + fout + '...')
if os.path.isfile(fout): os.remove(fout)
f = open(fout, 'a')
neqNCF=0 # zero nonconformal nodes
neqPIX=0 # zero fixed pressure
nfnode=len(fnode_bc) # zero nodal force/flux
nvsrc=0  # zero volume source
line2=np.array([len(tet_node),int(nnd),len(mat),int(neq),int(nfnode),len(trac_el),int(nvsrc),len(abs_bc),int(nfnd),len(ogrid),int(neqNCF),int(neqPIX)]).reshape(1,12)

np.savetxt(f, line1, fmt='%s')
np.savetxt(f, line2, delimiter=' ', fmt='%d '*12)
np.savetxt(f, line3, delimiter=' ', fmt='%g %g %d %d')
np.savetxt(f, line4, delimiter=' ', fmt='%g %g %d %g %d %d %d %d %d %d %d %d')
if rsf==0:
    np.savetxt(f, line5, delimiter=' ', fmt='%d %d')
else:
    np.savetxt(f, line5, delimiter=' ', fmt='%d %d %g')

if conv_Xflt: nxfnd = 0

np.savetxt(f, np.hstack((line6,[[nxfnd]])), delimiter=' ', fmt='%g %g %d')
np.savetxt(f, np.column_stack((tet_node, mat_typ)), delimiter=' ', fmt='%d '*5)
np.savetxt(f, np.column_stack((coord, bc_typ, bcc_typ)) , delimiter = ' ', fmt='%g '*3+ '%d '*(3+poro) + '%d '* ((3+poro)))

np.savetxt(f, mat, delimiter=' ', fmt = '%g '*12)
# HF transmissbility
if poro: np.savetxt(f, hhf, delimiter=' ', fmt = '%g '*9)


# %%


# fault slip: strike, dip and open, zero for hybrid model
slip = np.array([0.0, 0.0, 0.0]).reshape(3,1) # zero means locked
n=[2]
j=0
vecf = np.empty(shape=(0,11))
xfnd = np.empty(shape=(0,3))
for node_pos, node_neg in zip(ft_pos_nodes, ft_neg_nodes):
    if node_pos != node_neg:
        vec1  = [[1,  0,  0,  0, node_pos], # fourth dimension for poro 
                 [-1, 0,  0,  0, node_neg]]
        vec2  = [[0,  1,  0,  0, node_pos],
                 [0, -1,  0,  0, node_neg]]
        vec3  = [[0,  0,  1,  0, node_pos],
                 [0,  0, -1,  0, node_neg]]
        mat_ft = np.hstack((vec_fs[j,:].reshape(3,1), vec_fd[j,:].reshape(3,1), vec_fn[j,:].reshape(3,1)))
        mat_f = np.matrix.transpose(mat_ft).reshape(1,9)
        val = np.dot(mat_ft,slip)
        cval1 = np.hstack((val[0], [0.,0.])).reshape(1,3)
        cval2 = np.hstack((val[1], [0.,0.])).reshape(1,3)
        cval3 = np.hstack((val[2], [0.,0.])).reshape(1,3)
        np.savetxt(f, n, fmt = '%d')
        np.savetxt(f, vec1, delimiter = ' ', fmt = '%g %g %g %g %d')
        np.savetxt(f, cval1, delimiter = ' ', fmt = "%1.2E %g %g")
        np.savetxt(f, n, fmt = '%d')
        np.savetxt(f, vec2, delimiter = ' ', fmt = '%g %g %g %g %d')
        np.savetxt(f, cval2, delimiter = ' ', fmt = "%1.2E %g %g")
        np.savetxt(f, n, fmt = '%d')
        np.savetxt(f, vec3, delimiter = ' ', fmt = '%g %g %g %g %d')
        np.savetxt(f, cval3, delimiter = ' ', fmt = "%1.2E %g %g")
        vecf = np.vstack((vecf,np.hstack(([[node_pos, node_neg]], mat_f))))
        xfnd = np.vstack((xfnd, coord[node_pos-1,:]))
        if (poro) and perm[j] > 0:  # edit: d
            vec4 = [[0, 0, 0, 1, node_pos], 
                    [0, 0, 0,-1, node_neg]]
            cval4 =[[0, 0, 0]]
            np.savetxt(f, n, fmt = '%d')
            np.savetxt(f, vec4, delimiter = ' ', fmt = '%g %g %g %g %d')
            np.savetxt(f, cval4, delimiter = ' ', fmt = "%1.2E %g %g")
        j+=1
# Cross-link nodes
if Xflt>1:
    # vecx_fs[:,0] = -1.; vecx_fs[:,1] = 0.; vecx_fs[:,2] = 0.;
    # vecx_fd[:,0] = 0.; vecx_fd[:,1] = 0.; vecx_fd[:,2] = -1.;
    # vecx_fn[:,0] = 0.; vecx_fn[:,1] = 1.; vecx_fn[:,2] = 0.;
    ft_x_pos = np.hstack((ft_x_nodes[:,0],ft_x_nodes[:,1])) # postive means the postive side of the main fault ++(0) +-(1)
    ft_x_neg = np.hstack((ft_x_nodes[:,2],ft_x_nodes[:,3])) # negative --(2) -+(3)
    print (j)
    i=0
    vecxf=np.empty(shape=(nxfnd,10))
    stx_init=np.zeros(shape=(nxfnd,3))
    for node_pos,node_neg in zip(ft_x_pos,ft_x_neg):
        vec1  = [[1, 0, 0, 0, node_pos],
                 [-1, 0, 0,0, node_neg]]
        vec2  = [[0, 1, 0, 0, node_pos],
                 [0, -1, 0,0, node_neg]]
        vec3  = [[0, 0, 1, 0, node_pos],
                 [0, 0, -1,0, node_neg]]
        cval = np.array([[0.,0.,0.]])
        np.savetxt(f, n, fmt = '%d')
        np.savetxt(f, vec1, delimiter = ' ', fmt = '%g %g %g %g %d')
        np.savetxt(f, cval, delimiter = ' ', fmt = "%g %g %g")
        np.savetxt(f, n, fmt = '%d')
        np.savetxt(f, vec2, delimiter = ' ', fmt = '%g %g %g %g %d')
        np.savetxt(f, cval, delimiter = ' ', fmt = "%g %g %g")
        np.savetxt(f, n, fmt = '%d')
        np.savetxt(f, vec3, delimiter = ' ', fmt = '%g %g %g %g %d')
        np.savetxt(f, cval, delimiter = ' ', fmt = "%g %g %g")
        # if (poro) and perm[j] > 0: 
        #     vec4 = [[0, 0, 0, 1, node_pos], 
        #             [0, 0, 0,-1, node_neg]]
        #     cval4 =[[0, 0, 0]]
        #     np.savetxt(f, n, fmt = '%d')
        #     np.savetxt(f, vec4, delimiter = ' ', fmt = '%g %g %g %g %d')
        #     np.savetxt(f, cval4, delimiter = ' ', fmt = "%1.2E %g %g")
        mat_f = np.hstack((vec_fs[j,:], vec_fd[j,:], vec_fn[j,:])).reshape(1,9)
        vecf = np.vstack((vecf,np.hstack(([[node_pos, node_neg]], mat_f))))
        xfnd = np.vstack((xfnd, coord[node_pos-1,:]))
        j+=1 # Fortran 1 based nfnd index
        # Auxiliary fault
        if i<len(ft_x_nodes[:,0]):
            vecxf[i,:] = np.hstack((j,vecx_fs[i],vecx_fd[i],vecx_fn[i]))
            # vecxf[i,:] = np.hstack((j,[-1,0,0],[0,0,-1],[0,1,0]))
        else: # Reversely linked node pairs [3<->1]
            vecxf[i,:] = np.hstack((j,-vecx_fs[i],-vecx_fd[i],-vecx_fn[i]))
            # vecxf[i,:] = np.hstack((j,[1,0,0],[0,0,1],[0,-1,0]))
        i+=1
    if load_init_Xflt:
        stx_init[:len(ft_x_nodes),:] = stx_init_main[:,:]
        stx_init[len(ft_x_nodes):,:] = stx_init_main[:,:]
        
# Conventional intersection
if conv_Xflt>0:
    if node_pos != node_neg:
        ft_x_pos = ft_x_nodes[:,0] # postive means the postive side of the main fault ++(0) +-(1)
        ft_x_neg = ft_x_nodes[:,2] # negative --(2) -+(3)
        print (j)
        i=0
        for node_pos,node_neg in zip(ft_x_pos,ft_x_neg):
            vec1  = [[1, 0, 0, 0, node_pos],
                     [-1, 0, 0,0, node_neg]]
            vec2  = [[0, 1, 0, 0, node_pos],
                     [0, -1, 0,0, node_neg]]
            vec3  = [[0, 0, 1, 0, node_pos],
                     [0, 0, -1,0, node_neg]]
            cval = np.array([[0.,0.,0.]])
            np.savetxt(f, n, fmt = '%d')
            np.savetxt(f, vec1, delimiter = ' ', fmt = '%g %g %g %g %d')
            np.savetxt(f, cval, delimiter = ' ', fmt = "%g %g %g")
            np.savetxt(f, n, fmt = '%d')
            np.savetxt(f, vec2, delimiter = ' ', fmt = '%g %g %g %g %d')
            np.savetxt(f, cval, delimiter = ' ', fmt = "%g %g %g")
            np.savetxt(f, n, fmt = '%d')
            np.savetxt(f, vec3, delimiter = ' ', fmt = '%g %g %g %g %d')
            np.savetxt(f, cval, delimiter = ' ', fmt = "%g %g %g")
            if (poro) and perm[j] > 0: 
                vec4 = [[0, 0, 0, 1, node_pos], 
                        [0, 0, 0,-1, node_neg]]
                cval4 =[[0, 0, 0]]
                np.savetxt(f, n, fmt = '%d')
                np.savetxt(f, vec4, delimiter = ' ', fmt = '%g %g %g %g %d')
                np.savetxt(f, cval4, delimiter = ' ', fmt = "%1.2E %g %g")
            mat_f = np.hstack((vec_fs[j,:], vec_fd[j,:], vec_fn[j,:])).reshape(1,9)
            vecf = np.vstack((vecf,np.hstack(([[node_pos, node_neg]], mat_f))))
            xfnd = np.vstack((xfnd, coord[node_pos-1,:]))
            j+=1 # Fortran 1 based nfnd index
            i+=1


# %%


# add a header for reading input (p sync for pos 0/neg side 1, 2 average)
p_sync = 2*np.ones((nfnd,1),dtype=np.int8)
i=0
if poro: 
    for node_pos in ft_pos_nodes:
        y = coord[node_pos - 1,dim-1]
        if y > -3.000:
            p_sync[i] = 1
        elif y <= -3.000:
            p_sync[i] = 0
        i+=1


# %%


# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# coord_fltele = coord[ft_pos_nodes - 1,dim-1]

# fig = plt.figure(figsize=(4,5),constrained_layout=True)
# spec2 = gridspec.GridSpec(ncols=1, nrows=1, left=0.05, right=0.48, wspace=0.05, figure=fig)
# f2_ax2 = fig.add_subplot(spec2[0])
# f2_ax2.scatter(p_sync,coord_fltele,  alpha=0.6,cmap='jet')
# # f2_ax2.scatter(st_init[:,0]/st_init[:,1],coord_fltele,  alpha=0.6,cmap='jet')
# # f2_ax2.set_xlim(e6, 5e6)
# f2_ax2.set_ylim( -3.4, -2.6)


# %%


if rsf: # with biot & perm
    np.savetxt(f, np.hstack((vecf, b0, V0, dtau0, a, b, L, theta_init, perm, st_init, xfnd, frc, coh, dcoh, p_sync)), delimiter = ' ',       fmt = '%d '*2+'%g '*23+'%d '+'%g '*2 + '%d')
else:
    np.savetxt(f,np.hstack((vecf,fc,fcd,dc,perm,st_init,xfnd,frc,coh,dcoh,biot,p_sync)),delimiter=' ',           fmt='%d '*2 + '%g '*9 + '%g '*3 + '%d ' + '%g '*3 + '%g '*3 + '%d '+ '%g '*3 + '%d')

# Auxiliary fault
if Xflt>1:
    np.savetxt(f,np.hstack((vecxf,stx_init)),delimiter=' ', fmt='%d '+'%g '*12)
    
#  point force/source
if nfnode>0:
    np.savetxt(f, fnode_bc, delimiter=' ',            fmt ='%d %1.2E %1.2E %1.2E %1.2E %g %g')
# Boundary traction
np.savetxt(f, np.column_stack((trac_el, trac_bc)), delimiter=' ',        fmt ='%d %d %1.2E %1.2E %1.2E %1.2E %g %g')
# Observation grid
np.savetxt(f,ogrid,delimiter=' ',fmt='%g '*3)
# Absorbing boundariesm
np.savetxt(f,abs_bc,delimiter=' ',fmt='%d %d %d')
f.close(); # necessary for successful save
print ('Defmod file '+fout+' created')


# %%


print (abs_bc[-1])
print (len(abs_bc))
print (len(trac_el))
print (np.column_stack((trac_el,trac_bc)).shape)
print (neq)
print (j*4)
print (nfnd)
print (np.hstack((vecf,fc,fcd,dc,st_init,xfnd,frc,coh,dcoh)).shape)


# %%


# import shutil
# shutil.copy(name_out, '/home/jingmingruan/Desktop/3Dexample/Zeerijp/')


# %%




