#!/usr/bin/env python
# ## NOTICE! the node No. start with 0 in msh. Need to +1 for PFLOTRAN!!! 
# 
# 28-09-2020: implementing fault boundary elements
# 25-10-2021: updated as stand alone script
# 04-02-2022: minor fix and comment

import numpy as np
import argparse
import meshio # compatilble with version 3.3
import h5py

ap=argparse.ArgumentParser()
ap.add_argument('-m') # mesh file
ap.add_argument('-e2') # 2d element type
ap.add_argument('-e3') # 3d element type
ap.add_argument('-mo') # local coordinates
ap.add_argument('-km') # coordinate unit
ap.add_argument('-ro') # coordinate unit

name = ap.parse_args().m
etype_2D = ap.parse_args().e2
etype_3D = ap.parse_args().e3
coord_zero = int(ap.parse_args().mo)
unit_km = int(ap.parse_args().km)
rotate_angle = float(ap.parse_args().ro)

print ("Exporting FE mesh to hdf5 file")
print ("Reading form " + name + ".msh ....")
print ("element type is " + etype_2D + "/" + etype_3D)
if coord_zero:
    print ("Use min(coord) as origin...")

if ((meshio.__version__)[0]) != str(3):
    print ("Your meshio version is " + meshio.__version__)
    raise ValueError("Only used with meshio 3.3 installed !!")

# prosessing 2D mesh
mesh = meshio.read(name + ".msh") 
fout = name + '.h5'
print (mesh)


print (mesh.cells[etype_3D].shape)
print (mesh.cell_data[etype_3D]['gmsh:physical'].shape)


coord = mesh.points
# coord = np.round(coord, 8)
if unit_km==0:
    print ("Converting m to km...")
    coord = coord / 1000

print (coord.shape)
# rotate the axis
if rotate_angle>0:
    print ("Rotating axis with angle %0.3f degree..." %rotate_angle)
    new_yaxis = np.array([np.sin(rotate_angle*np.pi/180), np.cos(rotate_angle*np.pi/180), 0])
    new_zaxis = np.array([0, 0, 1])
    new_xaxis = np.cross(new_yaxis, new_zaxis)
    new_xaxis = new_xaxis / np.sqrt(np.sum(new_xaxis**2))
    new_yaxis = new_yaxis / np.sqrt(np.sum(new_yaxis**2))
    new_zaxis = new_zaxis / np.sqrt(np.sum(new_zaxis**2))
    new_axes  = np.array([new_xaxis, new_yaxis, new_zaxis])
    old_axes = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float).reshape(3, -1)
    rotation_matrix = np.inner(new_axes,old_axes)
    coord_rt = np.zeros_like(coord)
    for i in range(coord.shape[0]):
        coord_rt[i,:] = np.inner(coord[i,:],rotation_matrix)
    coord = (coord_rt)
# print (coord)
# print (coord_rt)

xmin, ymin, zmax = np.min(coord[:,0]),np.min(coord[:,1]),np.max(coord[:,2])
print ("Model origin(zmax): " + str(xmin) + "," +  str(ymin) + "," + str(0) + "...")
if coord_zero:
    coord[:,0] = coord[:,0] - xmin
    coord[:,1] = coord[:,1] - ymin
    # coord[:,2] = coord[:,2] - zmax
    xmin, ymin, zmax = np.min(coord[:,0]),np.min(coord[:,1]),np.max(coord[:,2])
    xmax, ymax, zmin = np.max(coord[:,0]),np.max(coord[:,1]),np.min(coord[:,2])
    print ("New model origin(zmax): " + str(xmin) + "," + str(ymin) + "," + str(zmax) + "...")
    print ("New model max(zmin): " + str(xmax) + "," + str(ymax) + "," + str(zmin) + "...")
lin_node = mesh.cells['line']+1
tet_node = mesh.cells[etype_3D]+1
tri_node = mesh.cells[etype_2D]+1
id_lin = np.array(range(len(lin_node)))+1 # line elements (fault boundary)
id_tet = np.array(range(len(tet_node)))+1 # tet elements
id_tri = np.array(range(len(tri_node)))+1 # tri elements (boundary)
col_lin = 2*np.ones((len(lin_node),1),dtype=np.int32)
col_tet = 4*np.ones((len(tet_node),1),dtype=np.int32)
col_tri = 3*np.ones((len(tri_node),1),dtype=np.int32)
phy_lin = np.abs(mesh.cell_data['line']['gmsh:physical']) # using boundary in gmsh will give a negative value, here we use abs
phy_tet = mesh.cell_data[etype_3D]['gmsh:physical']
phy_tri = mesh.cell_data[etype_2D]['gmsh:physical']
lins = np.hstack((col_lin,lin_node))
tets = np.hstack((col_tet,tet_node))
tris = np.hstack((col_tri,tri_node))


print (coord.shape)
print (len(lin_node)+len(tet_node)+len(tri_node))

phy_range = ([phy_lin.min(),phy_lin.max(),phy_tet.min(),phy_tet.max(),phy_tri.max(),phy_tri.min()])
print (min(phy_range),min(phy_range))
index = []

Vmap = np.hstack((id_tet.reshape(len(tet_node),1), phy_tet.reshape(len(tet_node),1)))

f = h5py.File(fout, "w")
print ('Outputting FE mesh file ' + fout + '...')
print (Vmap[:,1].min(),Vmap[:,1].max())
grp = f.create_group('Domain')
grp.create_dataset('Cells', data=np.hstack((col_tet,tet_node)), dtype=np.int32)
grp.create_dataset('Vertices', data=coord, dtype=np.float64) # input (m)
grp_reg = f.create_group('Regions')
grp = grp_reg.create_group('All')
grp.create_dataset('Cell Ids', data=Vmap[:,0], dtype=np.int32)

for i in np.unique(phy_tet):
    index.append(np.where(Vmap[:,1]==i))
    grp = grp_reg.create_group("group%d" % (i))
    grp.create_dataset('Cell Ids', data=np.squeeze(Vmap[index[i-1],0]), dtype=np.int32) # Pbug: number not continus
    print (len(index))



# index_tri=[]
Smap = np.hstack((id_tri.reshape(len(tris),1), phy_tri.reshape(len(tris),1)))
print (Smap[:,1].min(),Smap[:,1].max())

for i in np.unique(phy_tri):
    index.append(np.where(Smap[:,1]==i))
    grp = grp_reg.create_group("group%d" % (i))
    grp.create_dataset('Vertex Ids', data=np.squeeze(tris[Smap[index[i-1],0]-1,:]), dtype=np.int32) # Pbug: number not continus
    print (len(index))


Lmap = np.hstack((id_lin.reshape(len(lins),1), phy_lin.reshape(len(lins),1)))
print (Lmap[:,1].min(),Lmap[:,1].max())

for i in np.unique(phy_lin):
    index.append(np.where(Lmap[:,1]==(i)))
    grp = grp_reg.create_group("group%d" % ((i)))
    grp.create_dataset('Vertex Ids', data=np.squeeze(lins[Lmap[index[i-1],0]-1,:]), dtype=np.int32) # Pbug: number not continus
    print (len(index))

f.close()
print (fout + ' created.')





