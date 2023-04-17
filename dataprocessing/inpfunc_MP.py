import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import h5py
import os, sys
import meshio
import time

class inpclass: 
    def __init__(self, fe_node,coord,flt_sel,ft_pos_nodes,el_type,pos_dir=0):
        """
        Run all below function with order! (not the *_func, which are multiprocessing
        implementation for the corresponding base functions)
        """
        
        self.fe_node = fe_node
        self.coord = coord
        self.flt_sel = flt_sel
        self.ft_pos_nodes = ft_pos_nodes
        self.pos_dir = pos_dir # x y z
        self.el_type = el_type

        # if flt_sel==0:
        #     self.flt_sel = np.empty((1,3), dtype=np.uint32)
        #     self.ft_pos_nodes = np.empty((1,), dtype=np.uint32)

        if self.el_type == "tet":
            self.elnnd = 4 # numbder of nodes per element
            self.elnnd_flt = 3 # numbder of nodes per element of the 2D fault element
            self.snode_list = [0,1,2]

        elif self.el_type == "hex":
            self.elnnd = 8
            self.elnnd_flt = 4
            self.snode_list = [0,1,2,3]
        else: 
            raise ValueError('Not supported element type!')
        
        # place holder (to be make by the functions)
        self.ft_neg_nodes = None
        self.flt_el = None
        self.vec_fs = None
        self.vec_fn = None
        self.vec_fd = None
        self.cell_map_bc = None
        self.ft_x_nodes = None
        self.loc_tet = None

    def Crack3D_MP(self,mp): 
        """
        Implementing Crack3D with multiprocessing
        ! debug: careful about the pos_dir if the element shape if wierd
        """

        nnd = len(self.coord)
        print ("Making crack in the mesh...")
        # start MP        
        start = time. time()
        print ("pool init with np=" + str(mp))
        # if __name__ == '__main__':
        with Pool(mp) as p:
            mp_out = p.map(self.Crack3D_MP_func, self.ft_pos_nodes)
        p.close()
        print ("pool done")
        end = time. time()
        print ("Time elapsed:", end - start)
        # End MP

        # could be parallized
        id_neg = 1
        for i in tqdm(range(len(self.ft_pos_nodes))):
            tet_loc   = mp_out[i][0]
            # print (tet_loc)
            node_loc  = mp_out[i][1]
            # print (node_loc)
            work_node = self.ft_pos_nodes[i]
            neg_node = nnd + id_neg

            # mask_pos = np.in1d(self.fe_node[tet_loc[j]][node_loc[j]],ft_pos_nodes)
            # pos_sum = np.sum(mask_pos,axis=1)
            # if pos_sum == 0:
            # else:
            #     print ("replaced node on another fault!!")

            for j in range(len(tet_loc)):
                if node_loc[j]>-1:
                    self.fe_node[tet_loc[j]][node_loc[j]] = neg_node
            self.coord = np.vstack((self.coord,self.coord[work_node-1]))
            id_neg += 1

        self.ft_neg_nodes = np.arange(nnd+1, nnd+len(self.ft_pos_nodes)+1)

    # nested function for MP
    # debug for the intersecting fault 
    def Crack3D_MP_func(self,ft_pos_nodes):
        replace = 0
        work_node = ft_pos_nodes
        
        flt_sel = self.flt_sel
        elnnd_flt = self.elnnd_flt
        elnnd = self.elnnd
        fe_node = self.fe_node
        coord = self.coord

        # find out surface element contain the fault node
        mask = np.in1d(flt_sel, work_node)
        mask_rsh = mask.reshape((-1,elnnd_flt))
        work_sum = np.sum(mask_rsh,axis=1)
        tri_loc = np.where(work_sum==1)
        work_sel = flt_sel[tri_loc]
        if len(work_sel)==0: raise ValueError('no 2d element found')

        # find out volume element contain the fault node
        mask = np.in1d(fe_node, work_node)
        mask_rsh = mask.reshape((-1,elnnd))
        work_sum = np.sum(mask_rsh,axis=1)
        tet_loc = np.where(work_sum==1)[0] # np.where return with tuple
        work_el = fe_node[tet_loc]
        # calculate fault normal on fault nodes
        if len(work_el)==0: raise ValueError('no 3d element found')

        # calculate fault normal on fault nodes
        # pos_dir = self.pos_dir    # x y z
        indexx = np.where(self.ft_pos_nodes==work_node)[0][0]
        pos_dir = self.pos_dir[indexx]
        vec_ave = np.array([]).reshape(0,3)
        for j in range(len(work_sel)):
            v = coord[work_sel[j]-1]
            vec = np.cross(v[1] - v[0], v[2] - v[0]) # debug: here pick only one sel for vec
            vec /=(np.ones((3))*np.linalg.norm(vec)).T # normalize normal vector
            vec = np.around(vec,8) # 0 != 0 without round off (not reliable!!s)
            # if np.sign(vec[pos_dir])==0:
            #     pos_dir = 1    # x y z
            # if np.sign(vec[pos_dir])==0:
            #     pos_dir = 2    # x y z
            if np.sign(vec[pos_dir])==0:
                print (vec[:])
                print (pos_dir)
                raise ValueError('Check your positive direction')
                # pos_dir = 1    # x y z
            vec = vec*np.sign(vec[pos_dir]) # input if horizontal fault (debug) *(postive diredction)
            vec_ave = np.vstack((vec_ave,vec))
        vec_ave = np.mean(vec_ave,axis=0)

        if np.sign(vec_ave[pos_dir])==0:
            raise ValueError('Check your positive direction')

        rep_loc = np.zeros_like(tet_loc)-1
        for j in range(len(work_el)):
            el_node = work_el[j]            
            el_coord = coord[el_node-1,:]            
            cnt = np.mean(el_coord,axis=0)            
            vecc = cnt - coord[work_node-1] # debug: reliable? extreme?
            vecc /=(np.ones((3))*np.linalg.norm(vecc)).T # normalize normal vector
            if np.dot(vec_ave,vecc) > 0: 
                error = 0
            elif np.dot(vec_ave,vecc) < 0: # replace negative side? does it matter? it does
                node_loc = fe_node[tet_loc[j]] == work_node
                rep_loc[j] = np.arange(elnnd)[node_loc]
                if rep_loc[j]<0:
                    raise ValueError('node not found')
                if replace==0:
                    replace = 1
            else: 
                print ("np.cross(vec,vecc)=")
                print (np.cross(vec,vecc))
                print (vec)
                print (vecc)
                print ("bad orientation!!!")
        if replace==0: raise ValueError ("not replaced")
        return tet_loc, rep_loc 

    def rotate_model(self,rotate_angle):
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
            coord_rt = np.zeros_like(self.coord)
            for i in range(self.coord.shape[0]):
                coord_rt[i,:] = np.inner(self.coord[i,:],rotation_matrix)
            self.coord = np.round(coord_rt,10) # round is good or not?

            self.coord[:,0] = self.coord[:,0]-self.coord[:,0].min()
            self.coord[:,1] = self.coord[:,1]-self.coord[:,1].min()
            
            coord = self.coord
            nfnd = len(self.ft_pos_nodes)

            print (coord[:,0].min(),coord[:,0].max(),coord[:,0].min()-coord[:,0].max())
            print (coord[:,1].min(),coord[:,1].max(),coord[:,1].min()-coord[:,1].max())
            print (coord[:,2].min(),coord[:,2].max(),coord[:,2].min()-coord[:,2].max())
            if coord.shape[0]-np.unique(coord,axis=0).shape[0]-nfnd>0: raise ValueError("Round off error!")
        else:
            raise ValueError("Postive and in degree!")
            
    # debug: slow and not really necessary
    def SearchFltele3D_MP(self,mp): # debug: conner tri will result in two loc_tet 
        """
        Searching 2D fault surface element in 3D mesh.
        """
        flt_sel = self.flt_sel
        elnnd_flt = self.elnnd_flt
        elnnd = self.elnnd
        fe_node = self.fe_node
        coord = self.coord
        print ("Looking for fault surface elements...")
            
        flt_el = np.zeros((len(flt_sel),elnnd),dtype=np.uint32)
        flt_sel_id=np.arange((len(flt_sel)),dtype=np.int32)
        self.el_id=np.arange((len(fe_node)),dtype=np.int32)
        
        if mp==0: 
            print ("mp=0 single core")
            for i in tqdm(flt_sel_id):
                a = self.SearchFltele3D_MP_func(i)
        else:
            start = time.time()
            print ("pool init with np=" + str(mp))
            # if __name__ == '__main__':
            with Pool(mp) as p:
                loc_tet = p.map(self.SearchFltele3D_MP_func, flt_sel_id)
            p.close()
            print ("pool done")
            end = time.time()
            print ("Time elapsed:", end - start)

        self.loc_tet = loc_tet
        self.flt_el = fe_node[loc_tet,:][:,0,:] # list to array problem
        # self.flt_el = np.asarray(flt_el)[:,0,:]
        self.el_id = None

    
    def SearchFltele3D_MP_func(self,flt_sel_id): # debug: conner tri will result in two loc_tet
        """
        Searching 2D fault surface element in 3D mesh. ! debug: intersection where tet_sum_p != 3 (=2 or 1)
        """
        elnnd_flt = self.elnnd_flt
        elnnd = self.elnnd
        fe_node = self.fe_node
        coord = self.coord

        flt_sel_mp = self.flt_sel[flt_sel_id]
        tet_sum_p = np.sum(np.isin(fe_node,flt_sel_mp), axis=1)
        loc_tet = self.el_id[tet_sum_p==elnnd_flt]
        if len(loc_tet)>1: 
            raise ValueError("Crack not generated properly!!")
        elif len(loc_tet)<1: 
            print ("id:")
            print (flt_sel_id)
            print ("flt_sel_mp:")
            print (flt_sel_mp)
            print (coord[flt_sel_mp-1,:])
            print (tet_sum_p.max())
            raise ValueError("error! fault element not found")        
        # flt_el_mp=fe_node[loc_tet,:]
        return loc_tet
    
    
    # need to define postive vector for each fault branches input
    # def Vecfsn3D(coord,flt_sel,flt_el,flt_sel_index,ft_x_nodes,ft_pos_nodes,el_type):
    # # debug: slow and not really necessary
    def Vecfsn3D_MP(self,mp): # ft_x_nodes could be dropped
        """
        Calculate vector for 2D fault element.
        """    
        self.nnd = len(self.coord)
        self.nxfnd = len(self.ft_x_nodes) * 2 # cross-link constraint is double the equations than the normal
        self.nfnd = len(self.ft_pos_nodes) + self.nxfnd
        
        print ('Forming fault constraints...')
        self.vec_fn = np.zeros(((self.nfnd), 3), dtype=float)
        self.vec_fs = np.zeros(((self.nfnd), 3), dtype=float)
        self.vec_fd = np.zeros(((self.nfnd), 3), dtype=float)
        
        flt_el_id=np.arange((len(self.flt_el)),dtype=np.int32)

        start = time. time()
        print ("pool init with np=" + str(mp))
        # if __name__ == '__main__':
        with Pool(mp) as p:
            vec_global = p.map(self.Vecfsn3D_MP_func, flt_el_id)
        p.close()
        print ("pool done")
        end = time. time()
        print ("Time elapsed (pool):", end - start)
        
        for vec in tqdm(vec_global):
            row = vec[0]
            vec_fs = vec[1]
            vec_fn = vec[2]
            vec_fd = vec[3]
            self.vec_fs[row,:] += vec_fs 
            self.vec_fn[row,:] += vec_fn
            self.vec_fd[row,:] += vec_fd

        # reset the holder (normalize zero vector will have warnings.)
        if self.nxfnd>0: 
            self.vec_fs[-1,:] = 0
            self.vec_fn[-1,:] = 0
            self.vec_fd[-1,:] = 0
            
        self.vec_fs /= (np.ones((3,1))*np.linalg.norm(self.vec_fs, axis=1)).T # normalized
        self.vec_fd /= (np.ones((3,1))*np.linalg.norm(self.vec_fd, axis=1)).T
        self.vec_fn /= (np.ones((3,1))*np.linalg.norm(self.vec_fn, axis=1)).T

    def Vecfsn3D_MP_func(self,flt_el_id): # define positive direction
        """
        Calculate vector for 2D fault element.
        """
        nnd = self.nnd
        nfnd = self.nfnd
        ft_pos_nodes = self.ft_pos_nodes
        coord = self.coord
        sel = self.flt_sel[flt_el_id]
        el = self.flt_el[flt_el_id]
        
        off_node = np.in1d(el,sel,invert=True)
        off_node = el[off_node]
        row = np.empty((self.elnnd_flt,), dtype=np.uint32)
        for node, i in zip(sel, self.snode_list): 
            if sum(ft_pos_nodes==node)==1:
                row[i] = np.where(ft_pos_nodes == node)[0][0]
            else:
                row[i] = nfnd -1  # place holder just for running
        v = coord[sel-1]
        voff = coord[off_node-1][0] # should work with hex as well
        vec = np.cross(v[1] - v[0], v[2] - v[0])
        vecpos = voff-(v[0]+v[1]+v[2])/3. 
        vec=vec*np.sign(np.dot(vec,vecpos)) # debug: auto-determine positive direction
        vecs = np.cross(vec, [0, 0, 1]) # debug: direction
        vecd = np.cross(vec, vecs)  # debug: direction
        vec_fs_local = vecs 
        vec_fn_local = vec
        vec_fd_local = vecd
        return row,vec_fs_local,vec_fn_local,vec_fd_local
        
    # Exodus Numbering of the side
    def BCeleside3D_MP(self,bnd_el,mp):
        """
        Searching for boundary elements and sides for applying boundary conditions.
        """
        self.cell_map_bc = []    
        for i in (range(len(bnd_el))):
            bc_cell = np.zeros((len(bnd_el[i]),2),dtype=np.int32)
            
            start = time. time()
            print ("looking for boundary element...")
            print ("pool init with np=" + str(mp))
            # if __name__ == '__main__':
            with Pool(mp) as p:
                bc_cell_global = p.map(self.BCeleside3D_MP_func, bnd_el[i])
            p.close()
            print ("pool done")
            end = time. time()
            print ("Time elapsed:", end - start)
            
            row = 0            
            for bc in tqdm(bc_cell_global):
                bc_cell[row,0] = bc[0]
                bc_cell[row,1] = bc[1]
                row += 1
          
            self.cell_map_bc.append(bc_cell)
    
    def BCeleside3D_MP_func(self,sel):
        """
        Searching for boundary elements and sides for applying boundary conditions.
        """
        
        if self.el_type == "tet":
            options = {7 : 1,
                       9 : 2,
                       8 : 3,
                       6 : 4}
            index_side = np.array([1, 2, 3, 4])
            elnnd = 4
            elnnd_side = 3
        elif self.el_type == "hex":
            options = {14 : 1,
                       22 : 3,
                       10 : 5,
                       26 : 6}
            index_side = np.array([1, 2, 3, 4, 5, 6, 7, 8])
            elnnd = 8
            elnnd_side = 4
        else: 
            raise ValueError('Not supported element type!')    
            
        fe_node = self.fe_node
        narray = sel
        
        mask = np.in1d(fe_node, narray)
        mask_rsh = mask.reshape((-1,elnnd))
        mask_sum = (np.sum(mask_rsh,axis=1))
        loc = np.where(mask_sum==elnnd_side)
        bc_cell_local0 = loc[0]+1
        ssn = sum(index_side[mask_rsh[loc[0]][0]])
        if ssn == 18:
            if  sum(np.in1d(index_side[mask_rsh[loc[0]][0]],2))>0:
                bc_cell_local1 = 2
            else:
                bc_cell_local1 = 4
        else:    
            bc_cell_local1 = options[ssn]
        return bc_cell_local0, bc_cell_local1
    
    # find the replace the intersecting nodes based on the normal vectors of the intersecting faults
    def Xflt3D(self,flt_sel_ex,flt_sel_seg,ft_xinit_nodes):
        fe_node = self.fe_node
        coord = self.coord
        flt_sel = flt_sel_ex
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
            crd_flt_x = coord[ft_xinit_node-1,:] # the coord of the node at the intersection
            for j in range(4-1):
                ft_x_nodes[i,j+1] = nnd + id_xflt +1 # new id for the replaced interescting node
                id_xflt += 1
                coord = np.vstack((coord,crd_flt_x)) # duplicate the coord for the split node

            work_sel = np.in1d(flt_sel,ft_xinit_node).reshape(-1,3) # finding the surface elements that include the intersecting node
            work_sel = np.sum(work_sel,axis=1) # 
            work_sel = np.where(work_sel==1)

            # check the intersecting nodes on the sel, and calculate the normal vector for the faults.
            for j in range(len(work_sel[0])): # loop over the elements containing the interesecting node
                xtri = flt_sel[work_sel[0][j]] # debug: seperate to (2/4)faults in flt_sel?
                segid_check = 0
                for k in range(4): # loop over all the segment
                    work_sum = (np.sum(np.isin(flt_sel_seg[k],xtri),axis=1).max())
                    if work_sum==3 and segid_check==0: # if sel belongs to one of the segments
                        id_seg = k # k being the seg id
                        segid_check = 1
                    elif work_sum == 2 or work_sum == 1 or work_sum == 0:
                        pass
                    else:
                        print(work_sum)
                        print("error1! Cannot find the sel on the flt_tri_seg_all!")
                v = coord[xtri-1]
                cntri = np.mean(v,axis=0)
                vec = np.cross(v[1] - v[0], v[2] - v[0])
                pos_dir_xflt = 0 # x y z
                if np.sign(vec[pos_dir_xflt])==0: 
                    pos_dir_xflt = 1
                # if np.sign(vec[pos_dir_xflt])==0: 
                #     pos_dir_xflt = 2
                if np.sign(vec[pos_dir_xflt])==0: 
                    print ('0 value at the postive direction!!')
                if id_seg < 2: # main fault
                    vecfn = vec*np.sign(vec[pos_dir_xflt]) # debug!!: auto set positive direction (potstive X)
                elif id_seg > 1: # aux fault
                    vecfn = vec*np.sign(vec[pos_dir_xflt]) # debug!!: auto set positive direction
                if segid_check: 
                    vecxfn[id_seg,:] = vecfn 
                else:
                    print("error2!")
            vecxfn /= (np.ones((3,1))*np.linalg.norm(vecxfn, axis=1)).T
            print (ft_xinit_node)
            print (vecxfn)

            # check the intersecting nodes on the tet elements...
            work_sum = np.sum(np.isin(fe_node,ft_xinit_node),axis=1)
            loc_tet = np.where(work_sum==1)
            xflt_fe_node = fe_node[loc_tet[0],:]
            for j in range(len(xflt_fe_node)): # loop over all flt_xele (tet elements containing the xnode)
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
                    if np.sign(work_dot[0])>0: # if postive from the main fault, then id = 1
                        fe_node[loc_tet[0][j],:][mask] = ft_x_nodes[i,1]
                    elif np.sign(work_dot[0])<0: # if postive from the second fault, then id = 3
                        fe_node[loc_tet[0][j],:][mask] = ft_x_nodes[i,3]
                    else:
    #                     print ((np.sign(work_dot)))
                        print ("error3!!!")
                elif np.sum(np.sign(work_dot)) == -4: # NN
                    fe_node[loc_tet[0][j],:][mask] = ft_x_nodes[i,2] # 2 (NN) is opposite with the 0 (PP)
                else:
    #                 print ((np.sign(work_dot)))
                    print ("error4!!")

        print (str(len(coord)-nnd)+ " nodes replaced")
        
        self.fe_node = fe_node
        self.coord = coord
        self.ft_x_nodes = ft_x_nodes

    # def saveclass():
    #     pass
    # def load():
    #     pass

    # ft_x_nodes [f1PP f2PN f1NP f2NN ]
    # need to define postive vector for MAIN fault branches input
    # def VecXfsn3D_X(coord,flt_sel_aux,ft_x_nodes):

    # calculate for fault vector at the intersection
    def Vecfsn3D_X(self,flt_sel_main):
        ft_x_nodes = self.ft_x_nodes
        ft_pos_nodes = self.ft_pos_nodes
        coord = self.coord
        self.vec_fs[len(ft_pos_nodes):,:] = 0 # nan can not be changed by +=
        self.vec_fn[len(ft_pos_nodes):,:] = 0
        self.vec_fd[len(ft_pos_nodes):,:] = 0
        vec_fs = self.vec_fs
        vec_fd = self.vec_fd
        vec_fn = self.vec_fn
        j= 0 + len(ft_pos_nodes)
        # for node_pos,node_neg in zip(ft_x_west,ft_x_east):
        for node_pos,node_neg in zip(ft_x_nodes[:,0],ft_x_nodes[:,2]):
            row = np.where(flt_sel_main == node_pos)[0] # only choose the main fault 
            if len(row) ==0: print ("error! intersecting node not found")
            for i in range(len(row)):
                tri = flt_sel_main[row[i],:]
                v = coord[tri-1]
        #         voff = coord[off_node-1][0]
                vec = np.cross(v[1] - v[0], v[2] - v[0])
                vecs = np.cross(vec, [0, 0, 1]) # debug: postive direction!! (regarding the fault vector)
                vecd = np.cross(vec, vecs)  # debug: direction
                vec_fs[j,:] += vecs 
                vec_fn[j,:] += vec
                vec_fd[j,:] += vecd
                vec_fs[j+len(ft_x_nodes),:] += vecs 
                vec_fn[j+len(ft_x_nodes),:] += vec
                vec_fd[j+len(ft_x_nodes),:] += vecd
            j += 1

        vec_fs /= (np.ones((3,1))*np.linalg.norm(vec_fs, axis=1)).T # normalized
        vec_fd /= (np.ones((3,1))*np.linalg.norm(vec_fd, axis=1)).T
        vec_fn /= (np.ones((3,1))*np.linalg.norm(vec_fn, axis=1)).T

        self.vec_fs = vec_fs
        self.vec_fd = vec_fd
        self.vec_fn = vec_fn

    def VecXfsn3D_X(self,flt_sel_aux):
        print ('forming cross-link constraints...')
        ft_x_nodes = self.ft_x_nodes
        coord = self.coord
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
                vecs = np.cross(vec, [0, 0, 1]) # debug: postive direction!! (regarding the fault vector)
                vecd = np.cross(vec, vecs)  # debug: direction
        #         if np.linalg.norm(vec)==0: print (tri,off_node,vecpos, vec_old)
                vecx_fs[j,:] += vecs 
                vecx_fn[j,:] += vec
                vecx_fd[j,:] += vecd
                vecx_fs[j+len(ft_x_nodes),:] += vecs 
                vecx_fn[j+len(ft_x_nodes),:] += vec
                vecx_fd[j+len(ft_x_nodes),:] += vecd
            j += 1

        vecx_fs /= (np.ones((3,1))*np.linalg.norm(vecx_fs, axis=1)).T # normalized
        vecx_fd /= (np.ones((3,1))*np.linalg.norm(vecx_fd, axis=1)).T
        vecx_fn /= (np.ones((3,1))*np.linalg.norm(vecx_fn, axis=1)).T

        self.vecx_fs = vecx_fs
        self.vecx_fd = vecx_fd
        self.vecx_fn = vecx_fn


    def Xflt3D_conv(self,flt_sel_ex,flt_sel_seg,ft_xinit_nodes):
        print ("Generating Xflt using conventional method...")
        fe_node = self.fe_node
        coord = self.coord
        flt_sel = flt_sel_ex
        nnd_fltx = len(ft_xinit_nodes)
        print (str(nnd_fltx) + " intersecting nodes expected, " + str(nnd_fltx*1) + " nodes to be replaced..")
        ft_x_nodes = np.zeros((len(ft_xinit_nodes),4),dtype=np.uint32)
        # replace intersecting faults based on normal vector
        id_xflt = 0
        nnd = len(coord)
        for i in tqdm(range(len(ft_xinit_nodes))):
            vecxfn = np.zeros((4,3))
            ft_xinit_node = ft_xinit_nodes[i]
            ft_x_nodes[i,0] = ft_xinit_node
            crd_flt_x = coord[ft_xinit_node-1,:] # the coord of the node at the intersection

            ft_x_nodes[i,1:] = nnd + id_xflt +1 # new id for the replaced interescting node
            id_xflt += 1
            coord = np.vstack((coord,crd_flt_x)) # duplicate the coord for the split node

            work_sel = np.in1d(flt_sel,ft_xinit_node).reshape(-1,3) # finding the surface elements that include the intersecting node
            work_sel = np.sum(work_sel,axis=1) # 
            work_sel = np.where(work_sel==1)
            # print (ft_x_nodes[i,:])

            # check the intersecting nodes on the sel, and calculate the normal vector for the faults.
            for j in range(len(work_sel[0])): # loop over the elements containing the interesecting node
                xtri = flt_sel[work_sel[0][j]] # debug: seperate to (2/4)faults in flt_sel?
                segid_check = 0
                for k in range(4): # loop over all the segment
                    work_sum = (np.sum(np.isin(flt_sel_seg[k],xtri),axis=1).max())
                    if work_sum==3 and segid_check==0: # if sel belongs to one of the segments
                        id_seg = k # k being the seg id
                        segid_check = 1
                    elif work_sum == 2 or work_sum == 1 or work_sum == 0:
                        pass
                    else:
                        print(work_sum)
                        print("error1! Cannot find the sel on the flt_tri_seg_all!")
                v = coord[xtri-1]
                cntri = np.mean(v,axis=0)
                vec = np.cross(v[1] - v[0], v[2] - v[0])
                pos_dir_xflt = 0 # x y z
                if np.sign(vec[pos_dir_xflt])==0: 
                    pos_dir_xflt = 1
                # if np.sign(vec[pos_dir_xflt])==0: 
                #     pos_dir_xflt = 2
                if np.sign(vec[pos_dir_xflt])==0: 
                    print ('0 value at the postive direction!!')
                if id_seg < 2: # main fault
                    vecfn = vec*np.sign(vec[pos_dir_xflt]) # debug!!: auto set positive direction (potstive X)
                elif id_seg > 1: # aux fault
                    vecfn = vec*np.sign(vec[pos_dir_xflt]) # debug!!: auto set positive direction
                if segid_check: 
                    vecxfn[id_seg,:] = vecfn 
                else:
                    print("error2!")
            vecxfn /= (np.ones((3,1))*np.linalg.norm(vecxfn, axis=1)).T
            # print (ft_xinit_node)
            # print (vecxfn)

            # check the intersecting nodes on the tet elements...
            work_sum = np.sum(np.isin(fe_node,ft_xinit_node),axis=1)
            loc_tet = np.where(work_sum==1)
            xflt_fe_node = fe_node[loc_tet[0],:]
            for j in range(len(xflt_fe_node)): # loop over all flt_xele (tet elements containing the xnode)
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
                    if np.sign(work_dot[0])>0: # if postive from the main fault, then id = 1
                        # fe_node[loc_tet[0][j],:][mask] = ft_x_nodes[i,1]
                        error=0
                    elif np.sign(work_dot[0])<0: # if postive from the second fault, then id = 3
                        fe_node[loc_tet[0][j],:][mask] = ft_x_nodes[i,3]
                    else:
    #                     print ((np.sign(work_dot)))
                        print ("error3!!!")
                elif np.sum(np.sign(work_dot)) == -4: # NN
                    fe_node[loc_tet[0][j],:][mask] = ft_x_nodes[i,3] # 2 (NN) is opposite with the 0 (PP)
                else:
    #                 print ((np.sign(work_dot)))
                    print ("error4!!")

        print (str(len(coord)-nnd)+ " nodes replaced")
        
        self.fe_node = fe_node
        self.coord = coord
        self.ft_x_nodes = ft_x_nodes

    # def Crack3D_sub_MP(self,pos_nodes,flt_tri_sub,mp): 
    #     """
    #     Implementing Crack3D with multiprocessing
    #     with subcequent order for Xflt conv
    #     ! debug: careful about the pos_dir if the element shape if wierd
    #     """
    #     self.ft_pos_nodes = np.hstack((self.ft_pos_nodes,pos_nodes))
    #     self.flt_sel = np.vstack((self.flt_sel, flt_tri_sub))

    #     nnd = len(self.coord)
    #     print ("Making crack in the mesh...")
    #     # start MP        
    #     start = time. time()
    #     print ("pool init with np=" + str(mp))
    #     # if __name__ == '__main__':
    #     with Pool(mp) as p:
    #         mp_out = p.map(self.Crack3D_MP_func, pos_nodes)
    #     p.close()
    #     print ("pool done")
    #     end = time. time()
    #     print ("Time elapsed:", end - start)
    #     # End MP

    #     # could be parallized
    #     id_neg = 1
    #     for i in tqdm(range(len(pos_nodes))):
    #         tet_loc   = mp_out[i][0]
    #         # print (tet_loc)
    #         node_loc  = mp_out[i][1]
    #         # print (node_loc)
    #         work_node = pos_nodes[i]
    #         neg_node = nnd + id_neg

    #         # mask_pos = np.in1d(self.fe_node[tet_loc[j]][node_loc[j]],ft_pos_nodes)
    #         # pos_sum = np.sum(mask_pos,axis=1)
    #         # if pos_sum == 0:
    #         # else:
    #         #     print ("replaced node on another fault!!")

    #         for j in range(len(tet_loc)):
    #             if node_loc[j]>-1:
    #                 self.fe_node[tet_loc[j]][node_loc[j]] = neg_node
    #         self.coord = np.vstack((self.coord,self.coord[work_node-1]))
    #         id_neg += 1

    #     ft_neg_nodes_sub = np.arange(nnd+1, nnd+len(pos_nodes)+1)
    #     self.ft_neg_nodes = np.hstack((self.ft_neg_nodes,ft_neg_nodes_sub))

    # def Xflt3D_conv(self,flt_sel_ex,ft_xinit_nodes):
    #     print ("Generating Xflt using conventional method...")
    #     fe_node = self.fe_node
    #     coord = self.coord
    #     ft_pos_nodes = self.ft_pos_nodes
    #     ft_neg_nodes = self.ft_neg_nodes
    #     elnnd = self.elnnd

    #     nnd = len(coord)
    #     flt_sel = flt_sel_ex
    #     nnd_fltx = len(ft_xinit_nodes)
    #     print (str(nnd_fltx) + " intersecting nodes expected, " + str(nnd_fltx) + " nodes to be replaced..")
    #     ft_x_nodes = np.zeros((len(ft_xinit_nodes),4),dtype=np.uint32)
    #     # replace intersecting faults based on normal vector
    #     id_xflt = 1
    #     for i in tqdm(range(len(ft_xinit_nodes))):
    #         replace = 0
    #         ft_xinit_node = ft_xinit_nodes[i]
    #         ft_pos_nodes = np.hstack((ft_pos_nodes,np.array([ft_xinit_node])))

    #         ft_neg_node_Xflt = nnd + id_xflt # 
    #         ft_neg_nodes = np.hstack((ft_neg_nodes,np.array([ft_neg_node_Xflt])))

    #         work_sel = np.in1d(flt_sel,ft_xinit_node).reshape(-1,3) # finding the surface elements that include the intersecting node
    #         work_sel = np.sum(work_sel,axis=1) # 
    #         work_sel = np.where(work_sel==1)
    #         if len(work_sel)==0: raise ValueError('no 2d element found')

    #         # find out volume element contain the fault node
    #         work_node = ft_xinit_node
    #         mask = np.in1d(fe_node, work_node)
    #         mask_rsh = mask.reshape((-1,elnnd))
    #         work_sum = np.sum(mask_rsh,axis=1)
    #         tet_loc = np.where(work_sum==1)[0] # np.where return with tuple
    #         work_el = fe_node[tet_loc]
    #         # calculate fault normal on fault nodes
    #         if len(work_el)==0: raise ValueError('no 3d element found')

    #         # calculate fault normal on fault nodes
    #         pos_dir = self.pos_dir    # x y z
    #         vec_ave = np.array([]).reshape(0,3)
    #         for j in range(len(work_sel)):
    #             v = coord[work_sel[j]-1]
    #             vec = np.cross(v[1] - v[0], v[2] - v[0]) # debug: here pick only one sel for vec
    #             vec /=(np.ones((3))*np.linalg.norm(vec)).T # normalize normal vector
    #             vec = np.around(vec,8) # 0 != 0 without round off (not reliable!!s)
    #             # if np.sign(vec[pos_dir])==0:
    #             #     pos_dir = 1    # x y z
    #             # if np.sign(vec[pos_dir])==0:
    #             #     pos_dir = 2    # x y z
    #             if np.sign(vec[pos_dir])==0:
    #                 raise ValueError('Check your positive direction')
    #             vec = vec*np.sign(vec[pos_dir]) # input if horizontal fault (debug) *(postive diredction)
    #             vec_ave = np.vstack((vec_ave,vec))
    #         vec_ave = np.mean(vec_ave,axis=0)

    #         rep_loc = np.zeros_like(tet_loc)-1
    #         for j in range(len(work_el)):
    #             el_node = work_el[j]            
    #             el_coord = coord[el_node-1,:]            
    #             cnt = np.mean(el_coord,axis=0)            
    #             vecc = cnt - coord[work_node-1] # debug: reliable? extreme?
    #             vecc /=(np.ones((3))*np.linalg.norm(vecc)).T # normalize normal vector
    #             if np.dot(vec_ave,vecc) > 0: 
    #                 error = 0
    #             elif np.dot(vec_ave,vecc) < 0: # replace negative side? does it matter? it does
    #                 node_loc = fe_node[tet_loc[j]] == work_node
    #                 rep_loc[j] = np.arange(elnnd)[node_loc]
    #                 if rep_loc[j]<0:
    #                     raise ValueError('node not found')
    #                 if replace==0:
    #                     replace = 1
    #             else: 
    #                 print ("np.cross(vec,vecc)=")
    #                 print (np.cross(vec,vecc))
    #                 print (vec)
    #                 print (vecc)
    #                 print ("bad orientation!!!")
    #         if replace==0: raise ValueError ("not replaced")


    #         for j in range(len(tet_loc)):
    #             if rep_loc[j]>-1:
    #                 self.fe_node[tet_loc[j]][rep_loc[j]] = ft_neg_node_Xflt
    #         self.coord = np.vstack((self.coord,self.coord[work_node-1]))
    #         id_xflt += 1

    #     self.ft_neg_nodes = ft_neg_nodes
    #     self.ft_pos_nodes = ft_pos_nodes

