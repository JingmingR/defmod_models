#!/usr/bin/env python
# coding: utf-8
# %%

# 08-06-2021: need to add methods to choose postive direction
# 06-08-2021: fix Vecfsn2D,Vecfsn3D for non-intersecting model
# 09-02-2022: fix Crack3D, averaged fault normal on fault nodes
import numpy as np
from tqdm import tqdm

# only for unstructrued triangle mesh running with the order

def Crack2D(fe_node,coord,flt_sel,ft_pos_nodes): 
    id_neg = 1
    crd_flt_n_seg = []

    cell_map = np.zeros((len(flt_sel),2),dtype=np.int32)
    cells_sum = np.zeros((len(fe_node),len(flt_sel)),dtype=np.int32)

    nnd = len(coord)
    for i in range(len(ft_pos_nodes)):
        replace = 0
        work_node = ft_pos_nodes[i]
        mask = np.in1d(flt_sel, work_node)
        mask_rsh = mask.reshape((-1,2))
        work_sum = np.sum(mask_rsh,axis=1)
        tri_loc = np.where(work_sum==1)
        work_sel = flt_sel[tri_loc]

        mask = np.in1d(fe_node, work_node)
        mask_rsh = mask.reshape((-1,3))
        work_sum = np.sum(mask_rsh,axis=1)
        tet_loc = np.where(work_sum==1)
        work_el = fe_node[tet_loc]

        vec1 = coord[work_sel[0][0]-1]-coord[work_sel[0][1]-1]
        vec2 = coord[work_sel[1][0]-1]-coord[work_sel[1][1]-1]
        vec = (vec1+vec2)/2 # average over two line element, might be wrong if extreme
        vec = -vec*np.sign(vec[0]) # input if horizontal fault
        for j in range(len(work_el)):
            el_node = work_el[j]
            el_coord = coord[el_node-1,:]
            cnt = np.mean(el_coord,axis=0)
            vecc = cnt - coord[work_node-1]
            if np.cross(vec,vecc) < 0: 
    #             tet_p = work_el[j]
                error = 0
            elif np.cross(vec,vecc) > 0: # replace postive? does it matter? it does
    #             tet_n = work_el[j]
                node_loc = fe_node[tet_loc[0][j]] == work_node
                neg_node = nnd + id_neg
                fe_node[tet_loc[0][j]][node_loc] = neg_node
                if replace==0:
                    coord = np.vstack((coord,coord[work_node-1]))
                    replace = 1 
            else: 
                print ("np.cross(vec,vecc)=")
                print (np.cross(vec,vecc))
                print ("bad orientation!!!")
        id_neg += 1

    ft_neg_nodes = np.arange(nnd+1, nnd+len(ft_pos_nodes)+1)
    return fe_node, coord, ft_neg_nodes


def SearchFltele(fe_node,flt_sel):
    flt_el = np.zeros((len(flt_sel),3),dtype=np.uint32)
    el_id=np.arange((len(fe_node)),dtype=np.int32)
    for i in tqdm(range(len(flt_sel))):
        tet_sum_p = np.sum(np.isin(fe_node,flt_sel[i]), axis=1)
        loc_tet = el_id[tet_sum_p==2]
    #     if len(loc_p)>1: # debug
    #         print (loc_p)
#         print (loc_tet,flt_sel[i])
        flt_el[i]=fe_node[loc_tet,:]
        
    return flt_el


def Xflt2D(fe_node,coord,flt_sel,flt_sel_seg,ft_xinit_nodes):
    nnd_fltx = len(ft_xinit_nodes)
    print (str(nnd_fltx) + " intersecting nodes expected, " + str(nnd_fltx*3) + " nodes to be replaced..")
    ft_x_nodes = np.zeros((len(ft_xinit_nodes),4),dtype=np.uint32)
    # replace intersecting faults based on normal vector
    id_xflt = 0
    nnd = len(coord)
    for i in tqdm(range(len(ft_xinit_nodes))):
        vecxfn = np.zeros((4,2))
        ft_xinit_node = ft_xinit_nodes[i]
        ft_x_nodes[i,0] = ft_xinit_node
        crd_flt_x = coord[ft_xinit_node-1,:]
        for j in range(4-1):
            ft_x_nodes[i,j+1] = nnd + id_xflt +1
            id_xflt += 1
            coord = np.vstack((coord,crd_flt_x))

        work_sel = np.in1d(flt_sel,ft_xinit_node).reshape(-1,2)
        work_sel = np.sum(work_sel,axis=1)
        work_sel = np.where(work_sel==1)
#         print (work_sel[0])
        for j in range(len(work_sel[0])):
            xtri = flt_sel[work_sel[0][j]] # debug: seperate to (2/4)faults in flt_sel?
            segid_check = 0
            for k in range(len(flt_sel_seg)):
                work_sum = (np.sum(np.isin(flt_sel_seg[k],xtri),axis=1).max())
                if work_sum==2 and segid_check==0: 
                    id_seg = k
                    segid_check = 1
                elif work_sum == 1 or work_sum == 0:
                    pass
                else:
                    print(work_sum)
                    print("error1!")
            vec = coord[xtri-1]
            vec = vec[0,:] - vec[1,:]
            vecfn = np.array([vec[1],-vec[0]])
            vecfn = vecfn*np.sign(vecfn[1]) # debug this can't be zero!
            if segid_check: 
                vecxfn[id_seg,:] = vecfn # input positive direction
            else:
                print("error2!")
                
        if len(flt_sel_seg) == 3:
            vecxfn[3,:] = vecxfn[2,:]
        print (vecxfn)
            
        work_sum = np.sum(fe_node == ft_xinit_node,axis = 1)
        loc_tet = np.where(work_sum==1)
        xflt_fe_node = fe_node[loc_tet[0],:]
        for j in range(len(xflt_fe_node)):
            mask = fe_node[loc_tet[0][j],:] == ft_xinit_node    
            work_dot = np.zeros((4))
            el_node = xflt_fe_node[j,:]
            el_coord = coord[el_node-1,:]
            cnt = np.mean(el_coord,axis=0)
            vecc = cnt - coord[ft_xinit_node-1]
            for k in range(len(vecxfn)):
                work_dot[k] = np.dot(vecc,vecxfn[k])
#             print ((np.sign(work_dot)))
            # input based on number of the fault branches
            if np.sum(np.sign(work_dot)) == 4:  # positive-postive  debug! branched fault or unsym us max or min?
                error=0
            elif np.sum(np.sign(work_dot)) == 0:  # NP or PN
                if np.sign(work_dot[0])>0:
                    fe_node[loc_tet[0][j],:][mask] = ft_x_nodes[i,1]
                elif np.sign(work_dot[0])<0:
                    fe_node[loc_tet[0][j],:][mask] = ft_x_nodes[i,3]
                else:
#                     print (vecxfn)
                    print ("error3!!!")
            elif np.sum(np.sign(work_dot)) == -4: # NN
                fe_node[loc_tet[0][j],:][mask] = ft_x_nodes[i,2]
            else:
#                 print (vecxfn)
                print ("error4!!")

    print (str(len(coord)-nnd)+ " nodes replaced")
    
    return fe_node,coord,ft_x_nodes

# need to define postive vector for each fault branches input
def Vecfsn2D(coord,flt_sel,flt_el,flt_sel_index,ft_x_nodes,ft_pos_nodes,index_lastmain):
    nnd = len(coord)
    nxfnd = len(ft_x_nodes) * 2
    nfnd = len(ft_pos_nodes) + nxfnd
    print ('Forming fault constraints...')
    check_pos = np.zeros((len(ft_pos_nodes)), dtype=np.uint32)
    vec_fn = np.zeros(((nfnd), 2), dtype=float) # ft_pos_nodes doesnt contain edge nodes
    vec_fs = np.zeros(((nfnd), 2), dtype=float) # should use flt_node? or just ft_pos_nodes?
    k = 0
    for tri,tet in zip(flt_sel,flt_el): # need to fix for intersecting nodes?
        off_node = np.in1d(tet,tri,invert=True)
        off_node = tet[off_node]
        row = np.empty((2,), dtype=np.uint32)
        for node, i in zip(tri, [0, 1]): 
            if sum(ft_pos_nodes==node)==1:
                row[i] = np.where(ft_pos_nodes == node)[0][0]
                check_pos[row[i]] = 1 
            else:
    #             print ("Boundary node " + str(node))
                row[i] = nfnd -1  # holder just for running
        v = coord[tri-1]
        voff = coord[off_node-1][0]
        v1 = v[0]
        v2 = v[1]
        vec1 = v2 - v1 # tangent 
        if flt_sel_index[k] < index_lastmain+1: # input for main fault
            vec1 = -vec1*np.sign(vec1[0]) # make sure negative 
            vec2 = np.array([vec1[1], -vec1[0]]) # normal
            vec2 = vec2*np.sign(vec2[1]) # make sure positive
        elif flt_sel_index[k] > index_lastmain: # input for aux fault
            vec1 = vec1*np.sign(vec1[0])  # negative
            vec2 = np.array([vec1[1], -vec1[0]])
            vec2 = vec2*np.sign(vec2[1]) # negative !!!
        else: 
            print ("error!!!")
        vec_fs[row,:] += vec1 
        vec_fn[row,:] += vec2
        k += 1
        
    # reset the holder
    if nxfnd>0:
        vec_fs[-1,:] = 0
        vec_fn[-1,:] = 0
        
    vec_fs = np.round(vec_fs,6)
    vec_fn = np.round(vec_fn,6)
    return vec_fs,vec_fn

# ft_x_nodes [f1PP f2PN f1NP f2NN ]
# need to define postive vector for MAIN fault branches input
def Vecfsn2D_X(coord,flt_el,flt_sel,flt_sel_main,ft_x_nodes,ft_pos_nodes,vec_fs,vec_fn):
    j= 0 + len(ft_pos_nodes)
    # for node_pos,node_neg in zip(ft_x_west,ft_x_east):
    for node_pos,node_neg in zip(ft_x_nodes[:,0],ft_x_nodes[:,2]):
        row = np.where(flt_sel[:len(flt_sel_main)-1,:] == node_pos)[0] # only choose the main fault 
    #     st_init[j,:]=[0.,0.] # initial stress need to be redone later (check)
    #     st_init[j+len(ft_x_nodes),:]=[0.,0.]
    #     frc[j]=1 # maybe partially lock the fault
    #     frc[j+len(ft_x_nodes)]=1
        for i in range(len(row)):
            tri = flt_sel[row[i],:]
            tet = flt_el[row[i],:]
            off_node = np.in1d(tet,tri,invert=True)
            off_node = tet[off_node]
            v = coord[tri-1]
            voff = coord[off_node-1][0]
            v1 = v[0]
            v2 = v[1]
            vec1 = v2 - v1 # tangent 
            vec1 = -vec1*np.sign(vec1[0]) # input
            vec2 = np.array([vec1[1], -vec1[0]]) # normal
            vec2 = vec2*np.sign(vec2[1]) # make sure positive
    #         if np.linalg.norm(vec)==0: print (tri,off_node,vecpos, vec_old)
            vec_fs[j,:] += vec1 
            vec_fs[j+len(ft_x_nodes),:] += vec1
            vec_fn[j,:] += vec2
            vec_fn[j+len(ft_x_nodes),:] += vec2
        j += 1
    return vec_fs,vec_fn

# ft_x_nodes [f1PP f2PN f1NP f2NN ]
# need to define postive vector for AUX fault branches input
def VecXfsn2D_X(coord,flt_sel_aux,ft_x_nodes):
    vecx_fn = np.zeros((len(ft_x_nodes)*2, 2), dtype=float) # ft_pos_nodes doesnt contain edge nodes
    vecx_fs = np.zeros((len(ft_x_nodes)*2, 2), dtype=float) # should use flt_node? or just ft_pos_nodes?
    j= 0
    # for node_pos,node_neg in zip(ft_x_west,ft_x_east):
    for node_pos,node_neg in zip(ft_x_nodes[:,0],ft_x_nodes[:,3]):
        row = np.where(flt_sel_aux == node_pos)[0] # only choose the main fault 
    #     st_init[j,:]=[0.,0.] # initial stress need to be redone later (check)
    #     st_init[j+len(ft_x_nodes),:]=[0.,0.]
    #     frc[j]=1 # maybe partially lock the fault
    #     frc[j+len(ft_x_nodes)]=1
        for i in range(len(row)):
            tri = flt_sel_aux[row[i],:]
    #         tet = flt_el[row[i],:]
    #         off_node = np.in1d(tet,tri,invert=True)
    #         off_node = tet[off_node]
            v = coord[tri-1]
    #         voff = coord[off_node-1][0]
            v1 = v[0]
            v2 = v[1]
            vec1 = v2 - v1 # tangent 
            vec1 = vec1*np.sign(vec1[0]) 
            vec2 = np.array([vec1[1], -vec1[0]]) # normal
            vec2 =  vec2*np.sign(vec2[1]) # make sure positive input
    #         if np.linalg.norm(vec)==0: print (tri,off_node,vecpos, vec_old)
            vecx_fs[j,:] += vec1 
            vecx_fn[j,:] += vec2
            vecx_fs[j+len(ft_x_nodes),:] += vec1 
            vecx_fn[j+len(ft_x_nodes),:] += vec2
        j += 1
        
    return vecx_fs, vecx_fn

# Exodus Numbering of the side
def BCeleside(fe_node,bnd_el):
    cell_map_bc = []
    index_side = np.array([0, 1, 2])
    index_smap = np.array([0, 1, 3, 2]) 
    for i in range(len(bnd_el)):
        bc_cell = np.zeros((len(bnd_el[i]),2),dtype=np.int32)
        for j in range(len(bnd_el[i])):
            narray = bnd_el[i][j]
            mask = np.in1d(fe_node, narray)
            mask_rsh = mask.reshape((-1,3))
            mask_sum = (np.sum(mask_rsh,axis=1))
            loc = np.where(mask_sum==2)        
            bc_cell[j,0] = loc[0]+1
            ssn = sum(index_side[mask_rsh[loc[0]][0]])
            bc_cell[j,1] = index_smap[ssn]
        cell_map_bc.append(bc_cell)
    return cell_map_bc


# %%
# 3D function

def Crack3D(fe_node,coord,flt_sel,ft_pos_nodes,el_type): 
    """
    Generating crack in Gmsh mesh, by splitting nodes on the fault planes and 
    repleced the nagative side of the fault plane with the new node.
    Roadmap: need auto determination of postive direction (based on shear/normal direction?
    """
    
    if el_type == "tet":
        elnnd = 4 # numbder of nodes per element
        elnnd_flt = 3 # numbder of nodes per element of the 2D fault element
    elif el_type == "hex":
        elnnd = 8
        elnnd_flt = 4
    else: 
        raise ValueError('Not supported element type!')
    id_neg = 1
    crd_flt_n_seg = []

    # cells_sum = np.zeros((len(fe_node),len(flt_sel)),dtype=np.int32)

    nnd = len(coord)
    for i in tqdm(range(len(ft_pos_nodes))):
        replace = 0
        replace_times = 0
        work_node = ft_pos_nodes[i]
        
        # find out surface element contain the fault node
        mask = np.in1d(flt_sel, work_node)
        mask_rsh = mask.reshape((-1,elnnd_flt))
        work_sum = np.sum(mask_rsh,axis=1)
        tri_loc = np.where(work_sum==1)
        work_sel = flt_sel[tri_loc]

        # find out volume element contain the fault node
        mask = np.in1d(fe_node, work_node)
        mask_rsh = mask.reshape((-1,elnnd))
        work_sum = np.sum(mask_rsh,axis=1)
        tet_loc = np.where(work_sum==1)[0] # np.where return with tuple
        work_el = fe_node[tet_loc]
        
        # calculate fault normal on fault nodes
        pos_dir = 0    # x y z
        vec_ave = np.array([]).reshape(0,3)
        for j in range(len(work_sel)):
            v = coord[work_sel[j]-1]
            vec = np.cross(v[1] - v[0], v[2] - v[0]) # debug: here pick only one sel for vec
            vec /=(np.ones((3))*np.linalg.norm(vec)).T # normalize normal vector
            vec = vec*np.sign(vec[pos_dir]) # input if horizontal fault (debug) *(postive diredction)
            vec_ave = np.vstack((vec_ave,vec))
        vec_ave = np.mean(vec_ave,axis=0)

        if np.sign(vec_ave[pos_dir])==0:
            raise ValueError('Check your positive direction')

        for j in range(len(work_el)):
            el_node = work_el[j]            
            el_coord = coord[el_node-1,:]            
            cnt = np.mean(el_coord,axis=0)            
            vecc = cnt - coord[work_node-1] # debug: reliable? extreme?
            vecc /=(np.ones((3))*np.linalg.norm(vecc)).T # normalize normal vector
            # if (abs(np.dot(vec,vecc))<0.1): 
            #     print ("Possibly ill-shaped mesh:\n dot product=")
            #     print (np.dot(vec,vecc))
            if np.dot(vec_ave,vecc) > 0: 
    #             tet_p = work_el[j]
                error = 0
            elif np.dot(vec_ave,vecc) < 0: # replace negative side? does it matter? it does
    #             tet_n = work_el[j]
                node_loc = fe_node[tet_loc[j]] == work_node
                neg_node = nnd + id_neg
                fe_node[tet_loc[j]][node_loc] = neg_node
                if replace==0:
                    coord = np.vstack((coord,coord[work_node-1]))
                    replace = 1
                replace_times += 1
            else: 
                print ("np.cross(vec,vecc)=")
                print (np.cross(vec,vecc))
                print (vec)
                print (vecc)
                print ("bad orientation!!!")
        if replace==0: print ("not replaced")
        id_neg += 1

    ft_neg_nodes = np.arange(nnd+1, nnd+len(ft_pos_nodes)+1)
    return fe_node, coord, ft_neg_nodes


def SearchFltele3D(fe_node,flt_sel,coord,el_type): # debug: conner tri will result in two loc_tet
    """
    Searching 2D fault surface element in 3D mesh.
    """
    
    if el_type == "tet":
        elnnd = 4 # numbder of nodes per element
        elnnd_flt = 3 # numbder of nodes per element of the 2D fault element
    elif el_type == "hex":
        elnnd = 8
        elnnd_flt = 4
    else: 
        raise ValueError('Not supported element type!')
    flt_el = np.zeros((len(flt_sel),elnnd),dtype=np.uint32)
    el_id=np.arange((len(fe_node)),dtype=np.int32)
    for i in tqdm(range(len(flt_sel))):
        tet_sum_p = np.sum(np.isin(fe_node,flt_sel[i]), axis=1)
        loc_tet = el_id[tet_sum_p==elnnd_flt]
        if len(loc_tet)>1: 
            raise ValueError("Crack not generated properly!!")
            # Pos = 0
            # print (flt_sel[i],loc_tet)
            # work_sel = flt_sel[i]
            # v = coord[work_sel]
            # vec = np.cross(v[1] - v[0], v[2] - v[0])
            # cntri = np.mean(v,axis=0)
            # vec = vec*np.sign(vec[2]) # input if horizontal fault (debug) 
            # for j in range(len(loc_tet)):
            #     el_node = fe_node[loc_tet[j],:]
            #     el_coord = coord[el_node-1,:]
            #     cnt = np.mean(el_coord,axis=0)
            #     vecc = cnt - cntri
            #     if np.dot(vec,vecc) > 0: 
            #         Pos = j
            # loc_tet = loc_tet[Pos]
        elif len(loc_tet)<1: 
            print (flt_sel[i])
            raise ValueError("error! fault element not found")        
        flt_el[i]=fe_node[loc_tet,:]
        
    return flt_el


def Xflt3D(fe_node,coord,flt_sel,flt_sel_seg,ft_xinit_nodes):
    nnd_fltx = len(ft_xinit_nodes)
    print (str(nnd_fltx) + " intersecting nodes expected, " + str(nnd_fltx*3) + " nodes to be replaced..")
    ft_x_nodes = np.zeros((len(ft_xinit_nodes),4),dtype=np.uint32)
    # replace intersecting faults based on normal vector
    id_xflt = 0
    nnd = len(coord)
    for i in tqdm(range(len(ft_xinit_nodes))):
        vecxfn = np.zeros((4,3))
        ft_xinit_node = ft_xinit_nodes[i]
        ft_x_nodes[i,0] = ft_xinit_node
        crd_flt_x = coord[ft_xinit_node-1,:]
        for j in range(4-1):
            ft_x_nodes[i,j+1] = nnd + id_xflt +1
            id_xflt += 1
            coord = np.vstack((coord,crd_flt_x))

        work_sel = np.in1d(flt_sel,ft_xinit_node).reshape(-1,3)
        work_sel = np.sum(work_sel,axis=1)
        work_sel = np.where(work_sel==1)
#         print (work_sel[0])
        for j in range(len(work_sel[0])):
            xtri = flt_sel[work_sel[0][j]] # debug: seperate to (2/4)faults in flt_sel?
            segid_check = 0
            for k in range(4):
                work_sum = (np.sum(np.isin(flt_sel_seg[k],xtri),axis=1).max())
                if work_sum==3 and segid_check==0: 
                    id_seg = k
                    segid_check = 1
                elif work_sum == 2 or work_sum == 1 or work_sum == 0:
                    pass
                else:
                    print(work_sum)
                    print("error1!")
            v = coord[xtri-1]
            cntri = np.mean(v,axis=0)
            vec = np.cross(v[1] - v[0], v[2] - v[0])
            if id_seg < 2: # main fault
                vecfn = vec*np.sign(vec[2]) # debug!!: auto set positive direction
            elif id_seg > 1: # aux fault
                vecfn = vec*np.sign(vec[2]) # debug!!: auto set positive direction
            if segid_check: 
                vecxfn[id_seg,:] = vecfn 
            else:
                print("error2!")
        vecxfn /= (np.ones((3,1))*np.linalg.norm(vecxfn, axis=1)).T
        print (ft_xinit_node)
        print (vecxfn)

        work_sum = np.sum(np.isin(fe_node,ft_xinit_node),axis=1)
        loc_tet = np.where(work_sum==1)
        xflt_fe_node = fe_node[loc_tet[0],:]
        for j in range(len(xflt_fe_node)):
            mask = fe_node[loc_tet[0][j],:] == ft_xinit_node    
            work_dot = np.zeros((4))
            el_node = xflt_fe_node[j,:]
            el_coord = coord[el_node-1,:]
            cnt = np.mean(el_coord,axis=0)
            vecc = cnt - coord[ft_xinit_node-1]
            for k in range(len(vecxfn)):
                work_dot[k] = np.dot(vecc,vecxfn[k])
#             print ((np.sign(work_dot)))
            # input based on number of the fault branches
            if np.sum(np.sign(work_dot)) == 4:  # positive-postive  debug! branched fault or unsym us max or min?
                error=0
            elif np.sum(np.sign(work_dot)) == 0:  # NP or PN
                if np.sign(work_dot[0])>0:
                    fe_node[loc_tet[0][j],:][mask] = ft_x_nodes[i,1]
                elif np.sign(work_dot[0])<0:
                    fe_node[loc_tet[0][j],:][mask] = ft_x_nodes[i,3]
                else:
#                     print ((np.sign(work_dot)))
                    print ("error3!!!")
            elif np.sum(np.sign(work_dot)) == -4: # NN
                fe_node[loc_tet[0][j],:][mask] = ft_x_nodes[i,2]
            else:
#                 print ((np.sign(work_dot)))
                print ("error4!!")

    print (str(len(coord)-nnd)+ " nodes replaced")
    
    return fe_node,coord,ft_x_nodes

# need to define postive vector for each fault branches input
def Vecfsn3D(coord,flt_sel,flt_el,flt_sel_index,ft_x_nodes,ft_pos_nodes,el_type):
    """
    Calculate vector for 2D fault element.
    """    
    if el_type == "tet":
        elnnd = 4 # numbder of nodes per element
        elnnd_flt = 3 # numbder of nodes per element of the 2D fault element
        snode_list = [0,1,2]
    elif el_type == "hex":
        elnnd = 8
        elnnd_flt = 4
        snode_list = [0,1,2,3]
    else: 
        raise ValueError('Not supported element type!')
        
    nnd = len(coord)
    nxfnd = len(ft_x_nodes) * 2
    nfnd = len(ft_pos_nodes) + nxfnd
    print ('Forming fault constraints...')
    check_pos = np.zeros((len(ft_pos_nodes)), dtype=np.uint32)
    vec_fn = np.zeros(((nfnd), 3), dtype=float)
    vec_fs = np.zeros(((nfnd), 3), dtype=float)
    vec_fd = np.zeros(((nfnd), 3), dtype=float)
    k = 0
    for i in tqdm(range(len(flt_sel))): # need to fix for intersecting nodes?
        sel,el = flt_sel[i],flt_el[i]
        off_node = np.in1d(el,sel,invert=True)
        off_node = el[off_node]
        row = np.empty((elnnd_flt,), dtype=np.uint32)
        for node, i in zip(sel, snode_list): 
            if sum(ft_pos_nodes==node)==1:
                row[i] = np.where(ft_pos_nodes == node)[0][0]
                check_pos[row[i]] = 1 
            else:
    #             print ("Boundary node " + str(node))
                row[i] = nfnd -1  # place holder just for running
        v = coord[sel-1]
        voff = coord[off_node-1][0] # should work with hex as well
        vec = np.cross(v[1] - v[0], v[2] - v[0])
        vecpos = voff-(v[0]+v[1]+v[2])/3. 
        vec=vec*np.sign(np.dot(vec,vecpos)) # debug: auto-determine positive direction
        vecs = np.cross(vec, [0, 0, 1]) # debug: direction
        vecd = np.cross(vec, vecs)  # debug: direction
#         if flt_sel_index[k] < 8: # debug:input for main fault
#             vecs = -vecs*np.sign(vecs[0])
#             vec = -vec*np.sign(vec[0])
#             vecd = vecd*np.sign(vecd[1])
#         elif flt_sel_index[k] > 7: # input for aux fault
#             vecs = -vecs*np.sign(vecs[1])
#             vec = -vec*np.sign(vec[1])
#             vecd = vecd*np.sign(vecd[1])
#         else: 
#             print ("error!!!")
        vec_fs[row,:] += vecs 
        vec_fn[row,:] += vec
        vec_fd[row,:] += vecd
        k += 1
        
    # reset the holder
    if nxfnd>0: 
        vec_fs[-1,:] = 0
        vec_fn[-1,:] = 0
        vec_fd[-1,:] = 0
    
    return vec_fs,vec_fn,vec_fd

# ft_x_nodes [f1PP f2PN f1NP f2NN ]
# need to define postive vector for MAIN fault branches input
def Vecfsn3D_X(coord,flt_el,flt_sel,flt_sel_main,ft_x_nodes,ft_pos_nodes,vec_fs,vec_fn,vec_fd):
    j= 0 + len(ft_pos_nodes)
    # for node_pos,node_neg in zip(ft_x_west,ft_x_east):
    for node_pos,node_neg in zip(ft_x_nodes[:,0],ft_x_nodes[:,2]):
        row = np.where(flt_sel[:len(flt_sel_main)-1,:] == node_pos)[0] # only choose the main fault 
    #     st_init[j,:]=[0.,0.] # initial stress need to be redone later (check)
    #     st_init[j+len(ft_x_nodes),:]=[0.,0.]
    #     frc[j]=1 # maybe partially lock the fault
    #     frc[j+len(ft_x_nodes)]=1
        for i in range(len(row)):
            tri = flt_sel[row[i],:]
            tet = flt_el[row[i],:]
            off_node = np.in1d(tet,tri,invert=True)
            off_node = tet[off_node]
            v = coord[tri-1]
            voff = coord[off_node-1][0]
            vec = np.cross(v[1] - v[0], v[2] - v[0])
            vecpos = voff-(v[0]+v[1]+v[2])/3. 
            vec=vec*np.sign(np.dot(vec,vecpos)) # debug: auto-determine positive direction
            vecs = np.cross(vec, [0, 0, 1]) # debug: direction
            vecd = np.cross(vec, vecs)  # debug: direction
    #         if np.linalg.norm(vec)==0: print (tri,off_node,vecpos, vec_old)
            vec_fs[j,:] += vecs 
            vec_fs[j+len(ft_x_nodes),:] += vecs
            vec_fn[j,:] += vec
            vec_fn[j+len(ft_x_nodes),:] += vec
            vec_fd[j,:] += vecd
            vec_fd[j+len(ft_x_nodes),:] += vecd
        j += 1
    return vec_fs,vec_fn,vec_fd

# ft_x_nodes [f1PP f2PN f1NP f2NN ]
# need to define postive vector for MAIN fault branches input
def VecXfsn3D_X(coord,flt_sel_aux,ft_x_nodes):
    vecx_fn = np.zeros((len(ft_x_nodes)*2, 3), dtype=float) # ft_pos_nodes doesnt contain edge nodes
    vecx_fs = np.zeros((len(ft_x_nodes)*2, 3), dtype=float) # should use flt_node? or just ft_pos_nodes?
    vecx_fd = np.zeros((len(ft_x_nodes)*2, 3), dtype=float)
    j= 0
    # for node_pos,node_neg in zip(ft_x_west,ft_x_east):
    for node_pos,node_neg in zip(ft_x_nodes[:,0],ft_x_nodes[:,3]):
        row = np.where(flt_sel_aux == node_pos)[0] # only choose the main fault 
    #     st_init[j,:]=[0.,0.] # initial stress need to be redone later (check)
    #     st_init[j+len(ft_x_nodes),:]=[0.,0.]
    #     frc[j]=1 # maybe partially lock the fault
    #     frc[j+len(ft_x_nodes)]=1
        for i in range(len(row)):
            tri = flt_sel_aux[row[i],:]
    #         tet = flt_el[row[i],:]
    #         off_node = np.in1d(tet,tri,invert=True)
    #         off_node = tet[off_node]
            v = coord[tri-1]
    #         voff = coord[off_node-1][0]
            vec = np.cross(v[1] - v[0], v[2] - v[0])
            vecs = np.cross(vec, [0, 0, 1]) # debug: direction
            vecd = np.cross(vec, vecs)  # debug: direction
    #         if np.linalg.norm(vec)==0: print (tri,off_node,vecpos, vec_old)
            vecx_fs[j,:] += vecs 
            vecx_fn[j,:] += vec
            vecx_fd[j,:] += vecd
            vecx_fs[j+len(ft_x_nodes),:] += vecs 
            vecx_fn[j+len(ft_x_nodes),:] += vec
            vecx_fd[j+len(ft_x_nodes),:] += vecd
        j += 1
        
    return vecx_fs, vecx_fn,vecx_fd

# Exodus Numbering of the side
def BCeleside3D(fe_node,bnd_el,el_type):
    """
    Searching for boundary elements and sides for applying boundary conditions.
    """
    if el_type == "tet":
        options = {7 : 1,
                   9 : 2,
                   8 : 3,
                   6 : 4}
        index_side = np.array([1, 2, 3, 4])
        elnnd = 4
        elnnd_side = 3
    elif el_type == "hex":
        options = {14 : 1,
                   22 : 3,
                   10 : 5,
                   26 : 6}
        index_side = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        elnnd = 8
        elnnd_side = 4
    else: 
        raise ValueError('Not supported element type!')    
        
    cell_map_bc = []    
    for i in (range(len(bnd_el))):
        bc_cell = np.zeros((len(bnd_el[i]),2),dtype=np.int32)
        for j in tqdm(range(len(bnd_el[i]))):
            narray = bnd_el[i][j]
            mask = np.in1d(fe_node, narray)
            mask_rsh = mask.reshape((-1,elnnd))
            mask_sum = (np.sum(mask_rsh,axis=1))
            loc = np.where(mask_sum==elnnd_side)
            bc_cell[j,0] = loc[0]+1
#             print (index_side[mask_rsh[loc[0]][0]])
            ssn = sum(index_side[mask_rsh[loc[0]][0]])
            if ssn == 18:
#                 print (sum(np.in1d(index_side[mask_rsh[loc[0]][0]],2)))
                if  sum(np.in1d(index_side[mask_rsh[loc[0]][0]],2))>0:
                    bc_cell[j,1] = 2
                else:
                    bc_cell[j,1] = 4
            else:    
                bc_cell[j,1] = options[ssn]
        cell_map_bc.append(bc_cell)
    return cell_map_bc
