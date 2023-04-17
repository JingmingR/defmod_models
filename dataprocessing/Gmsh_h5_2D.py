#!/usr/bin/env python
# ## NOTICE! the node No. start with 0 in msh. Need to +1 for PFLOTRAN!!! 
# 
# 28-09-2020: implementing fault boundary elements
# 25-10-2021: updated as stand alone script
# 04-02-2022: minor fix and comment
# 10-03-2022: modify for 2D version: include physical group for points

import numpy as np
import argparse
import meshio # compatilble with version 3.3
import h5py

ap=argparse.ArgumentParser()
ap.add_argument('-m') # mesh file
ap.add_argument('-e2') # 2d element type
ap.add_argument('-mo') # local coordinates
ap.add_argument('-km') # coordinate unit

name = ap.parse_args().m
etype_2D = ap.parse_args().e2
coord_zero = int(ap.parse_args().mo)
unit_km = int(ap.parse_args().km)

print ("Exporting FE mesh to hdf5 file")
print ("Reading form " + name + ".msh ....")
print ("element type is " + etype_2D + "..." )
if coord_zero:
    print ("Use min(coord) as origin...")

if ((meshio.__version__)[0]) != str(3):
    print ("Your meshio version is " + meshio.__version__)
    raise ValueError("Only used with meshio 3.3 installed !!")

# prosessing 2D mesh
mesh = meshio.read(name + ".msh") 
fout = name + '.h5'
print (mesh)

print (mesh.cells[etype_2D].shape)
print (mesh.cell_data[etype_2D]['gmsh:physical'].shape)

coord = mesh.points
# print (mesh.points)
# coord = np.round(coord, 6)
if unit_km==0:
    print ("Converting m to km...")
    coord = coord / 1000
xmin, ymin = np.min(coord[:,0]),np.min(coord[:,1])
xmax, ymax = np.max(coord[:,0]),np.max(coord[:,1])
print ("Model domain: " + str(xmin) + "," +  str(ymin) + "/" + str(xmax) + "," +  str(ymax) + "...")
if coord_zero:
    coord[:,0] = coord[:,0] - xmin
    coord[:,1] = coord[:,1] - ymin

    xmin, ymin = np.min(coord[:,0]),np.min(coord[:,1])
    xmax, ymax = np.max(coord[:,0]),np.max(coord[:,1])
    print ("New model domain: " + str(xmax) + "," + str(ymax) +
           "/" + str(xmin) + "," + str(ymin) + "...")
vtx_node = mesh.cells['vertex']+1
lin_node = mesh.cells['line']+1
tri_node = mesh.cells[etype_2D]+1
id_vtx = np.array(range(len(vtx_node)))+1 # point (fault boundary)
id_lin = np.array(range(len(lin_node)))+1 # line elements (fault boundary)
id_tri = np.array(range(len(tri_node)))+1 # tri elements (boundary)
col_vtx = 1*np.ones((len(vtx_node),1),dtype=np.int32)
col_lin = 2*np.ones((len(lin_node),1),dtype=np.int32)
col_tri = 3*np.ones((len(tri_node),1),dtype=np.int32)
phy_vtx = np.abs(mesh.cell_data['vertex']['gmsh:physical']) 
phy_lin = np.abs(mesh.cell_data['line']['gmsh:physical']) # using boundary in gmsh will give a negative value, here we use abs
phy_tri = mesh.cell_data[etype_2D]['gmsh:physical']
vtxs = np.hstack((col_vtx,vtx_node))
lins = np.hstack((col_lin,lin_node))
tris = np.hstack((col_tri,tri_node))


print (coord.shape)
print (len(lin_node)+len(tri_node)+len(vtx_node))

phy_range = ([phy_vtx.min(),phy_vtx.max(),phy_lin.min(),phy_lin.max(),phy_tri.max(),phy_tri.min()])
print (min(phy_range),min(phy_range))
index = []

# Vmap = np.hstack((id_tet.reshape(len(tet_node),1), phy_tet.reshape(len(tet_node),1)))

Smap = np.hstack((id_tri.reshape(len(tris),1), phy_tri.reshape(len(tris),1)))
Lmap = np.hstack((id_lin.reshape(len(lins),1), phy_lin.reshape(len(lins),1)))
Vtmap = np.hstack((id_vtx.reshape(len(vtxs),1), phy_vtx.reshape(len(vtxs),1)))


f = h5py.File(fout, "w")
print ('Outputting FE mesh file ' + fout + '...')
grp = f.create_group('Domain')
grp.create_dataset('Cells', data=np.hstack((col_tri,tri_node)), dtype=np.int32)
grp.create_dataset('Vertices', data=coord, dtype=np.float64) # input (m)
grp_reg = f.create_group('Regions')
grp = grp_reg.create_group('All')
grp.create_dataset('Cell Ids', data=Smap[:,0], dtype=np.int32)

# print (Vmap[:,1].min(),Vmap[:,1].max())

# for i in np.unique(phy_tet):
#     index.append(np.where(Vmap[:,1]==i))
#     grp = grp_reg.create_group("group%d" % (i))
#     grp.create_dataset('Cell Ids', data=np.squeeze(Vmap[index[i-1],0]), dtype=np.int32) # Pbug: number not continus
#     print (len(index))

print (Smap[:,1].min(),Smap[:,1].max())

for i in np.unique(phy_tri):
    index.append(np.where(Smap[:,1]==i))
    grp = grp_reg.create_group("group%d" % (i))
    grp.create_dataset('Cell Ids', data=np.squeeze([Smap[index[i-1],0]]), dtype=np.int32)
    print (len(index))


print (Lmap[:,1].min(),Lmap[:,1].max())

for i in np.unique(phy_lin):
    index.append(np.where(Lmap[:,1]==(i)))
    grp = grp_reg.create_group("group%d" % ((i)))
    grp.create_dataset('Vertex Ids', data=np.squeeze(lins[Lmap[index[i-1],0]-1,:]), dtype=np.int32) # Pbug: number not continus
    print (len(index))

print (Vtmap[:,1].min(),Vtmap[:,1].max())

for i in np.unique(phy_vtx):
    index.append(np.where(Vtmap[:,1]==(i)))
    grp = grp_reg.create_group("group%d" % ((i)))
    grp.create_dataset('Vertex Ids', data=np.squeeze(vtxs[Vtmap[index[i-1],0]-1,:]), dtype=np.int32) # Pbug: number not continus
    print (len(index))
    
f.close()
print (fout + ' created.')





