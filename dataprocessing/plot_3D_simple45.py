import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
from scipy.signal import butter, lfilter, freqz
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.tri as tri
from matplotlib.colors import Normalize
import scipy.interpolate
import matplotlib.animation
from pylab import get_cmap
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter

class plot3D:
    def __init__(self, file, dyn,seis):
        self.file = file
        self.dyn = int(dyn)
        self.seis = int(seis)
        
        self.mdict = self.load_dict_from_hdf5(self.file)
    
        self.fcoord = self.mdict['crd_flt']
        self.dat_trac_sta = self.mdict['trac_sta']
        self.dat_slip_sta = self.mdict['slip_sta']

        if self.dyn>0:
            self.dt = self.mdict['dt']
            self.dt_dyn = self.mdict['dt_dyn']
            self.dat_log = self.mdict['log']
            self.dat_log_dyn = self.mdict['log_dyn']
            self.dat_trac_sort = self.mdict['trac_dyn']
            self.dat_slip_sort = self.mdict['slip_dyn']
            
        if self.seis:
            self.crd_obs = self.mdict['crd_obs']
            self.dat_seis_sort = self.mdict['obs_dyn']

        del self.mdict

    def recursively_load_dict_contents_from_group(self,h5file, path):
        """
        ....
        """
        ans = {}
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = item[()]
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = self.recursively_load_dict_contents_from_group(h5file,
                                                                     path + key + '/')
        return ans

    def load_dict_from_hdf5(self,filename):
        """
        ....
        """
        with h5py.File(filename, 'r') as h5file:
            return self.recursively_load_dict_contents_from_group(h5file, '/')      
        
    def plot_static(self,tstep_list,axis,delta=False,projection=False,
                    scatter=False,mask=0,xyaxis=0,vline=False,vdegree=27, azimuth=-110,zlim0=False,zlim1=False):
        
        font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

        matplotlib.rc('font', **font)
        
        z = self.fcoord[:,2][:]*1e0
        x = self.fcoord[:,0][:]*1e0
        y = self.fcoord[:,1][:]*1e0
        # xi = np.linspace(x.max(),x.min(), 2000)
        # yi = np.linspace(y.max(),y.min(), 2000)
        xi = np.linspace(x.min(),x.max(), 2000)
        yi = np.linspace(y.min(),y.max(), 2000)
        zi = np.linspace(z.max(),z.min(), 2000)
        
        points = np.array(list(zip(x,y,z)))
        grid_ = np.array(list(zip(xi,yi,zi)))

        if not projection:
            fig = plt.figure(figsize=(12,8))
            ax0 = fig.add_subplot(1, 1, 1, projection='3d')

            i = tstep_list[0]
            if axis < 4:
                z1 = (self.dat_trac_sta[:,axis,i-1])/1e6
                if delta:
                    z1 = z1 - (self.dat_trac_sta[:,axis,1])/1e6
                    print (np.unique(z1))

            elif axis == 4:
                z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2)/self.dat_trac_sta[:,2,i-1]
                z1 = np.abs(z1)
                if mask:
                    mask_ = z1>=mask
                    z1[~mask_] = 0

            elif axis == 5: # stress path ratio
                z0 = np.sqrt(self.dat_trac_sta[:,0,1]**2+self.dat_trac_sta[:,1,1]**2)
                z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2)
                zn0 = self.dat_trac_sta[:,2,i-1]-self.dat_trac_sta[:,2,1]
                z1 = np.abs((z1-z0)/zn0)
                # z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2)
                # z2 = (self.dat_trac_sta[:,2,i-1])-(self.dat_trac_sta[:,3,i-1])
                # z1 = np.abs(z1/z2)
                # z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2)-np.sqrt(self.dat_trac_sta[:,0,1]**2+self.dat_trac_sta[:,1,1]**2)
                # z2 = (self.dat_trac_sta[:,2,i-1])-(self.dat_trac_sta[:,2,1])
                # z1 = np.abs(z1/z2)
                    
            # si = griddata(points, z1, grid_)
            # pt0 = ax0.scatter3D(x, yi, zi, c=si, cmap='jet');
            pt0 = ax0.scatter3D(x, y, z, c=z1);

            ax0.set_xlabel('x(km)')
            ax0.set_ylabel('y(km)')
            ax0.set_zlabel('z(km)')
            ax0.view_init(vdegree, azimuth)

            cbar = fig.colorbar(pt0)
            if axis == 0:
                cbar.set_label('Shear stress (Strike) (MPa)')
            elif axis == 1:
                cbar.set_label('Shear stress (dip) (MPa)')
            elif axis == 2:
                cbar.set_label('Effective normal stress (MPa)')
            elif axis == 3:
                cbar.set_label('Pressure (MPa)')
            elif axis == 4:
                cbar.set_label(r'$ \tau / \sigma_{n} $')
            elif axis == 5:
                cbar.set_label(r'$ Stress path ratio$')

                
        else: # projection in x/y direction
            fig = plt.figure(figsize=(10,10))
            ax0 = fig.add_subplot(2, 1, 1)
            i = tstep_list[0]
            if axis < 4:
                z1 = (self.dat_trac_sta[:,axis,i-1])/1e6
                if delta:
                    z1 = z1 - (self.dat_trac_sta[:,axis,1])/1e6

            elif axis == 4:
                z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2)/self.dat_trac_sta[:,2,i-1]
                z1 = np.abs(z1)
                if mask:
                    mask_ = z1>=mask
                    z1[~mask_] = 0
                    
            elif axis == 5: # stress path ratio
                z0 = np.sqrt(self.dat_trac_sta[:,0,1]**2+self.dat_trac_sta[:,1,1]**2)
                z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2)
                zn0 = self.dat_trac_sta[:,2,i-1]-self.dat_trac_sta[:,2,1]
                z1 = np.abs((z1-z0)/zn0)
                mask_ = z1>=mask
                z1[mask_] = 0
                    
            # total normal stress
            elif axis == 6:
                z1 = (self.dat_trac_sta[:,2,i-1]-self.dat_trac_sta[:,3,i-1])/1e6
                if delta:
                    z1 = z1 - (self.dat_trac_sta[:,2,2]+self.dat_trac_sta[:,3,2])/1e6

            # stress path coefficient
            # elif axis == 6:
            #     z1 = (self.dat_trac_sta[:,axis,i-1])/1e6
            #     if delta:
            #         z1 = z1 - (self.dat_trac_sta[:,axis,1])/1e6
            if xyaxis == 0:
                xi, zi = np.meshgrid(xi, zi)
                si = griddata((x, z), z1, (xi, zi), method='linear')
                pt0 = ax0.imshow(si,extent=(x.min(),x.max(),z.min(),z.max()))
                # pt0 = ax0.imshow(si,cmap="jet")
            else:
                yi, zi = np.meshgrid(yi, zi)
                si = griddata((y, z), z1, (yi, zi), method='linear')
                pt0 = ax0.imshow(si,extent=(y.min(),y.max(),z.min(),z.max()))  
                # pt0 = ax0.imshow(si,cmap="jet")  

            cbar = fig.colorbar(pt0,fraction=0.046, pad=0.04)
            if axis == 0:
                cbar.set_label('Shear stress (Strike) (MPa)')
            elif axis == 1:
                cbar.set_label('Shear stress (dip) (MPa)')
            elif axis == 2:
                cbar.set_label('Effective normal stress (MPa)')
            elif axis == 3:
                cbar.set_label('Pressure (MPa)')
            elif axis == 4:
                cbar.set_label(r'$ \tau / \sigma_{n} $')
            elif axis == 5:
                cbar.set_label(r'$Stress path ratio$')
            elif axis == 6:
                cbar.set_label('Total normal stress (MPa)')

            if scatter:
                if xyaxis == 0:
                    pt1 = ax0.scatter(x,z,c=z1,cmap="gray")
                if xyaxis == 1:
                    pt1 = ax0.scatter(y,z,c=z1,cmap="gray")

            # if hline:
            #     ax0.axhline(y=hline)
            if vline:
                ax0.axvline(x=vline)
                ax1 = fig.add_subplot(2, 1, 2)
                if xyaxis == 0:
                        si = griddata((x, z), z1, ([vline], zi), method='linear')
                if xyaxis == 1:
                        si = griddata((y, z), z1, ([vline], zi), method='linear')
                ax1.plot(si,zi,'r-')
                ax1.set_ylabel('Z (km)')
                if axis == 0:
                    ax1.set_xlabel('Shear stress (Strike) (MPa)')
                elif axis == 1:
                    ax1.set_xlabel('Shear stress (dip) (MPa)')
                elif axis == 2:
                    ax1.set_xlabel('Effective normal stress (MPa)')
                elif axis == 3:
                    ax1.set_xlabel('Pressure (MPa)')
                elif axis == 4:
                    ax1.set_xlabel(r'$ \tau / \sigma_{n} $')
                elif axis == 5:
                    cbar.set_label('Total normal stress (MPa)')


                if si.max()>0:
                    ax1.invert_xaxis()
            if zlim0:        
                ax0.set_ylim(zlim0,zlim1)

            ax0.set_xlabel('X (km)')
            ax0.set_ylabel('Z (km)')

        fig.tight_layout()
        plt.show()
        
    def plot_static_list(self,xyaxis,vline,tstep_list,axis,scatter=False,delta=False,
                         mask=0,zlim0=False,zlim1=False):
        z = self.fcoord[:,2][:]*1e0
        x = self.fcoord[:,0][:]*1e0
        y = self.fcoord[:,1][:]*1e0
        xi = np.linspace(x.min(),x.max(), 2000)
        yi = np.linspace(y.min(),y.max(), 2000)
        zi = np.linspace(z.max(),z.min(), 2000)
        points = np.array(list(zip(x,y,z)))
        grid_ = np.array(list(zip(xi,yi,zi)))

        fig = plt.figure(figsize=(6,8))
        ax1 = fig.add_subplot(1, 1, 1)
        for i in (tstep_list):
            if axis < 4:
                z1 = (self.dat_trac_sta[:,axis,i-1])/1e6
                if delta:
                    z1 = z1 - (self.dat_trac_sta[:,axis,1])/1e6

            elif axis == 4:
                z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2)/self.dat_trac_sta[:,2,i-1]
                z1 = np.abs(z1)
                if mask:
                    mask_ = z1>=mask
                    z1[~mask_] = 0
                    
            elif axis == 5: # stress path ratio, shear/total normal
                if i > 4:
                    z0 = np.sqrt(self.dat_trac_sta[:,0,1]**2+self.dat_trac_sta[:,1,1]**2)
                    z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2)
                    zn0 = (self.dat_trac_sta[:,2,2]-self.dat_trac_sta[:,3,2])
                    zn1 = (self.dat_trac_sta[:,2,i-1]-self.dat_trac_sta[:,3,i-1])
                    z1 = np.abs((z1-z0)/(zn0-zn1))
                    mask_ = z1>=mask
                    z1[mask_] = 0
                else:
                    z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2)
                    z1[:] = 0
                    
            elif axis == 6:
                z1 = (self.dat_trac_sta[:,2,i-1]-self.dat_trac_sta[:,3,i-1])/1e6
                if delta:
                    z1 = z1 - (self.dat_trac_sta[:,2,2]-self.dat_trac_sta[:,3,2])/1e6
            if xyaxis == 0:        
                si = griddata((x, z), z1, ([vline], zi), method='linear')
            if xyaxis == 1:        
                si = griddata((y, z), z1, ([vline], zi), method='linear')
            ax1.plot(si,zi,'--',label=("Step" + str(i)))
            if scatter:
                ax1.plot(si,zi,'o', mfc='none')
        ax1.legend(loc='upper left')
        ax1.set_ylabel('Z (km)')
        
        if zlim0:
            ax1.set_ylim(zlim0,zlim1)
        if axis == 0:
            ax1.set_xlabel('Shear stress (Strike) (MPa)')
        elif axis == 1:
            ax1.set_xlabel('Shear stress (dip) (MPa)')
        elif axis == 2:
            ax1.set_xlabel('Effective normal stress (MPa)')
        elif axis == 3:
            ax1.set_xlabel('Pressure (MPa)')
        elif axis == 4:
            ax1.set_xlabel(r'$ \tau / \sigma_{n} $')
        elif axis == 5:
            ax1.set_xlabel('Stress path ratio')
        elif axis == 6:
            ax1.set_xlabel('Total normal stress')
        fig.tight_layout()
        
        plt.show()

    # DIY plotting
    def getz1(axis,tstep_list,delta=False,mask=False):
        for i in tstep_list:
            if axis < 4:
                z1 = (self.dat_trac_sta[:,axis,i-1])/1e6
                if delta:
                    z1 = z1 - (self.dat_trac_sta[:,axis,1])/1e6

            elif axis == 4:
                z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2)/self.dat_trac_sta[:,2,i-1]
                z1 = np.abs(z1)
                if mask:
                    mask_ = z1>=mask
                    z1[~mask_] = 0
            # total normal stress
            elif axis == 5:
                z1 = (self.dat_trac_sta[:,2,i-1]-self.dat_trac_sta[:,3,i-1])/1e6
                if delta:
                    z1 = z1 - (self.dat_trac_sta[:,2,2]+self.dat_trac_sta[:,3,2])/1e6
        return z1
        
    def verplot(z1,xaxis=False,yaxis=False):
        z = self.fcoord[:,2][:]*1e0
        x = self.fcoord[:,0][:]*1e0
        y = self.fcoord[:,1][:]*1e0
        xi = np.linspace(x.min(),x.max(), 2000)
        yi = np.linspace(y.min(),y.max(), 2000)
        zi = np.linspace(z.max(),z.min(), 2000)
        points = np.array(list(zip(x,y,z)))
        grid_ = np.array(list(zip(xi,yi,zi)))
        
        fig = plt.figure(figsize=(5,6))
        ax1 = fig.add_subplot(1, 1, 1)
        
        si = griddata((x, z), z1, ([vline], zi), method='linear')
        ax1.plot(si,zi,'--',label=("Step" + str(i)))
        
    def plot_static_tri_group(self,filename,tstep_list,delta=False,
                            scatter=False,mask=0,xyaxis=0,vline=False,
                            edgecolor=False,xlim0=False,xlim1=False,zlim0=False,zlim1=False):
        fig = plt.figure(figsize=(22,20),tight_layout=True)
        gs = gridspec.GridSpec(3, 2)
        titleax = { 0 : 'a',
                     1 : 'b',
                     2 : 'c',
                     3 : 'd',
                     4 : 'e',
                     5 : 'f',}
        for i in range(5):
            ax1 = fig.add_subplot(gs[i])
            triang, z1, cbar_label = self.plot_static_tri_group_func(filename=filename,tstep_list=tstep_list,
                                                                     axis=i,delta=delta,xyaxis=xyaxis)
            NbLevels = 256        
            if edgecolor:
                pt0=ax1.tripcolor(triang, z1, NbLevels, edgecolor="black") # cmap=plt.cm.jet,
            else:
                pt0=ax1.tripcolor(triang, z1, NbLevels ) # cmap=plt.cm.jet,

            cbar = fig.colorbar(pt0)
            cbar.set_label(cbar_label)
            if zlim0 or zlim1:
                ax1.set_ylim(zlim0,zlim1)
            if xlim0 or xlim1:
                ax1.set_xlim(xlim0,xlim1)
            # ax1.set_xlabel('Strike (km)')
            ax1.set_xlabel('X (km)')
            ax1.set_ylabel('Depth (km)')

            ax1.set_title(titleax[i], y=0.96,x=0.04, pad=-14)

        z = self.fcoord[:,2][:]*1e0
        x = self.fcoord[:,0][:]*1e0
        y = self.fcoord[:,1][:]*1e0

        # show the difference between the datas
        ax1 = fig.add_subplot(gs[5])
        # sid = self.diff_data_func(filename="/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/Buijze3D_fe.h5",filename_tri=False,xyaxis=xyaxis,axis=2,
        #           step=[23],delta=1,friction=0.6,xlim0=xlim0,xlim1=xlim1,zlim0=zlim0,zlim1=zlim1)
        sid = self.diff_data_func(filename="/media/jingmingruan/BACKUP/TUD/texel/3D_models/Zeerijp3D_anhydrite_4m/02/Zeerijp_fe.h5",filename_tri=False,xyaxis=xyaxis,axis=2,
                  step=[19],delta=1,friction=0.6,xlim0=xlim0,xlim1=xlim1,zlim0=zlim0,zlim1=zlim1)
        if xyaxis == 0:
            pt0 = ax1.imshow(sid,extent=(x.min(),x.max(),z.min(),z.max()),aspect='auto')  
        else:
            pt0 = ax1.imshow(sid,extent=(y.min(),y.max(),z.min(),z.max()),aspect='auto')  

        # p0 = np.array([1.922,-2.800])
        # p1 = np.array([2.000,-3.100])
        # points = np.vstack((p0,p1))
        points = []
        if len(points)>0:
            for i in range(points.shape[0]-1):
                ax1.plot(points[i:i+2,0], points[i:i+2,1], c='g',linestyle='--',alpha=0.95,linewidth=3)
        p0 = np.array([1.768,-2.800])
        p1 = np.array([2.000,-3.100])
        points = np.vstack((p0,p1))
        if len(points)>0:
            for i in range(points.shape[0]-1):
                ax1.plot(points[i:i+2,0], points[i:i+2,1], c='c',linestyle='--',alpha=0.95,linewidth=3)
        if zlim0 or zlim1:
            ax1.set_ylim(zlim0,zlim1)
        if xlim0 or xlim1:
            ax1.set_xlim(xlim0,xlim1)
        cbar = fig.colorbar(pt0)
        # ax1.set_xlabel('Y (km)')
        ax1.set_xlabel('X (km)')
        ax1.set_ylabel('Depth (km)')
        ax1.set_title(titleax[5], y=0.96,x=0.04, pad=-14)
        cbar.set_label('Incremental Columb stress (MPa)')

    def plot_static_tri_group_func(self,filename,tstep_list,axis,delta=False,
                        scatter=False,mask=0,xyaxis=0,vline=False,
                        edgecolor=False,xlim0=False,xlim1=False,zlim0=False,zlim1=False):
        flt_tri = np.load(filename)
        flt_tri = flt_tri - 1
        ft_pos_nodes = np.unique(flt_tri)
        a = np.arange(ft_pos_nodes.max(),dtype=np.int32)
        b = np.zeros(ft_pos_nodes.max(),dtype=np.int32)
        c = np.vstack((a,b)).T
        d = np.arange(len(ft_pos_nodes),dtype=np.int32)
        c[ft_pos_nodes-1,1] = d
       
        z = self.fcoord[:,2][:]*1e0
        x = self.fcoord[:,0][:]*1e0
        y = self.fcoord[:,1][:]*1e0

        i = tstep_list[0]
        if axis < 4:
            z1 = (self.dat_trac_sta[:,axis,i-1])/1e6
            if delta:
                z1 = z1 - (self.dat_trac_sta[:,axis,1])/1e6

        elif axis == 4:
            z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2)/self.dat_trac_sta[:,2,i-1]
            z1 = np.abs(z1)
            if mask:
                mask_ = z1>=mask
                print (str(np.sum(mask_)) + "("+ str(np.sum(mask_)/len(z1)*100) + ")"+"nodes above the mask")
                z1[~mask_] = 0


        elif axis == 5:
            z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2)
            z2 = (self.dat_trac_sta[:,2,i-1])-(self.dat_trac_sta[:,3,i-1])
            z1 = np.abs(z1/z2)

        elif axis == 6:
            z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2)
            z1 = np.abs(z1)
            if mask:
                mask_ = z1>=mask
                print (str(np.sum(mask_)) + "("+ str(np.sum(mask_)/len(z1)*100) + ")"+"nodes above the mask")
                z1[~mask_] = 0
                
        if xyaxis == 0:
            triang = tri.Triangulation(x, z,c[flt_tri-1][:,:,1])
        else:
            triang = tri.Triangulation(y, z,c[flt_tri-1][:,:,1])

        if axis == 0:
            cbar_label = ('Shear stress (Strike) (MPa)')
        elif axis == 1:
            cbar_label = ('Shear stress (dip) (MPa)')
        elif axis == 2:
            cbar_label =('Effective normal stress (MPa)')
        elif axis == 3:
            cbar_label = ('Pore pressure (MPa)')
        elif axis == 4:
            cbar_label = (r'$ {\tau / \sigma_{n}}^{\prime} $')
        elif axis == 5:
            cbar_label = ('Total normal stress (MPa)')
        elif axis == 6:
            cbar_label = ('Total shear stress (MPa)')

        return triang, z1, cbar_label
        
    def plot_static_tri(self,filename,tstep_list,axis,delta=False,
                        scatter=False,mask=0,xyaxis=0,vline=False,
                        edgecolor=False,xlim0=False,xlim1=False,zlim0=False,zlim1=False):
        flt_tri = np.load(filename)
        flt_tri = flt_tri - 1
        ft_pos_nodes = np.unique(flt_tri)
        a = np.arange(ft_pos_nodes.max(),dtype=np.int32)
        b = np.zeros(ft_pos_nodes.max(),dtype=np.int32)
        c = np.vstack((a,b)).T
        d = np.arange(len(ft_pos_nodes),dtype=np.int32)
        c[ft_pos_nodes-1,1] = d
       
        z = self.fcoord[:,2][:]*1e0
        x = self.fcoord[:,0][:]*1e0
        y = self.fcoord[:,1][:]*1e0

        fig = plt.figure(figsize=(15,10))
        ax0 = fig.add_subplot(1, 1, 1)
        i = tstep_list[0]
        if axis < 4:
            z1 = (self.dat_trac_sta[:,axis,i-1])/1e6
            if delta:
                z1 = z1 - (self.dat_trac_sta[:,axis,1])/1e6
                print (np.abs(z1).max())

        elif axis == 4:
            z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2)/self.dat_trac_sta[:,2,i-1]
            z1 = np.abs(z1)
            if mask:
                mask_ = z1>=mask
                print (str(np.sum(mask_)) + "("+ str(np.sum(mask_)/len(z1)*100) + ")"+"nodes above the mask")
                z1[~mask_] = 0
        # total normal stress
        # elif axis == 5:
        #     z1 = (self.dat_trac_sta[:,2,i-1]-self.dat_trac_sta[:,3,i-1])/1e6
        #     if delta:
        #         z1 = z1 - (self.dat_trac_sta[:,2,2]+self.dat_trac_sta[:,3,2])/1e6

        elif axis == 5:
            z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2)
            z2 = (self.dat_trac_sta[:,2,i-1])-(self.dat_trac_sta[:,3,i-1])
            z1 = np.abs(z1/z2)
            # z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2) - np.sqrt(self.dat_trac_sta[:,0,1]**2+self.dat_trac_sta[:,1,1]**2)
            # z2 = (self.dat_trac_sta[:,2,i-1])-(self.dat_trac_sta[:,2,1])
            # z1 = np.abs(z1/z2)

        elif axis == 6:
            z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2)
            z1 = np.abs(z1)
            if mask:
                mask_ = z1>=mask
                print (str(np.sum(mask_)) + "("+ str(np.sum(mask_)/len(z1)*100) + ")"+"nodes above the mask")
                z1[~mask_] = 0
                
        if xyaxis == 0:
            triang = tri.Triangulation(x, z,c[flt_tri-1][:,:,1])
        else:
            triang = tri.Triangulation(y, z,c[flt_tri-1][:,:,1])
        NbLevels = 256        
        if edgecolor:
            pt0=ax0.tripcolor(triang, z1, NbLevels, edgecolor="black") # cmap=plt.cm.jet,
        else:
            pt0=ax0.tripcolor(triang, z1, NbLevels ) # cmap=plt.cm.jet,


        # ax0.triplot(triang, 'bo-', lw=1)
        # ax0.triplot(y, z, c[flt_tri-1][:,:,1], 'go-', lw=1.0)
        # ax0.set_title('triplot of Delaunay triangulation')

        cbar = fig.colorbar(pt0)
        if axis == 0:
            cbar.set_label('Shear stress (Strike) (MPa)')
        elif axis == 1:
            cbar.set_label('Shear stress (dip) (MPa)')
        elif axis == 2:
            cbar.set_label('Effective normal stress (MPa)')
        elif axis == 3:
            cbar.set_label('Pore pressure (MPa)')
        elif axis == 4:
            cbar.set_label(r'$ \tau / \sigma_{n} $')
        elif axis == 5:
            cbar.set_label('Total normal stress (MPa)')
        elif axis == 6:
            cbar.set_label('Total shear stress (MPa)')

        if zlim0 or zlim1:
            ax0.set_ylim(zlim0,zlim1)
        if xlim0 or xlim1:
            ax0.set_xlim(xlim0,xlim1)
        ax0.set_xlabel('Strike (km)')
        ax0.set_ylabel('Depth (km)')

    def plot_static_tri_3D(self,filename,tstep_list,axis,delta=False,
                        scatter=False,mask=0,xyaxis=0,vline=False, vdegree=27, azimuth=-110,
                        edgecolor=False):
        flt_tri = np.load(filename)
        flt_tri = flt_tri - 1
        ft_pos_nodes = np.unique(flt_tri)
        a = np.arange(ft_pos_nodes.max(),dtype=np.int32)
        b = np.zeros(ft_pos_nodes.max(),dtype=np.int32)
        c = np.vstack((a,b)).T
        d = np.arange(len(ft_pos_nodes),dtype=np.int32)
        c[ft_pos_nodes-1,1] = d
       
        z = self.fcoord[:,2][:]*1e0
        x = self.fcoord[:,0][:]*1e0
        y = self.fcoord[:,1][:]*1e0

        fig = plt.figure(figsize=(15,10))
        ax0 = fig.add_subplot(1, 1, 1,projection='3d')
        
        i = tstep_list[0]
        if axis < 4:
            z1 = (self.dat_trac_sta[:,axis,i-1])/1e6
            if delta:
                z1 = z1 - (self.dat_trac_sta[:,axis,1])/1e6

        elif axis == 4:
            z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2)/self.dat_trac_sta[:,2,i-1]
            z1 = np.abs(z1)
            if mask:
                mask_ = z1>=mask
                print (str(np.sum(mask_)) + "("+ str(np.sum(mask_)/len(z1)*100) + ")"+"nodes above the mask")
                z1[~mask_] = 0
        # total normal stress
        # elif axis == 5:
        #     z1 = (self.dat_trac_sta[:,2,i-1]-self.dat_trac_sta[:,3,i-1])/1e6
        #     if delta:
        #         z1 = z1 - (self.dat_trac_sta[:,2,2]+self.dat_trac_sta[:,3,2])/1e6

        elif axis == 5:
            z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2)
            z2 = (self.dat_trac_sta[:,2,i-1])-(self.dat_trac_sta[:,3,i-1])
            z1 = np.abs(z1/z2)
            # z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2) - np.sqrt(self.dat_trac_sta[:,0,1]**2+self.dat_trac_sta[:,1,1]**2)
            # z2 = (self.dat_trac_sta[:,2,i-1])-(self.dat_trac_sta[:,2,1])
            # z1 = np.abs(z1/z2)
                
        if xyaxis == 0:
            triang = tri.Triangulation(x, y,c[flt_tri-1][:,:,1])
        else:
            triang = tri.Triangulation(x, y,c[flt_tri-1][:,:,1])



        # triangle_vertices = np.array([np.array([[x[T[0]], y[T[0]], z[T[0]]],
        #                                         [x[T[1]], y[T[1]], z[T[1]]], 
        #                                         [x[T[2]], y[T[2]], z[T[2]]]]) for T in triang.triangles])

        # midpoints = np.average(triangle_vertices, axis = 1)
        # midx = midpoints[:, 0]
        # midy = midpoints[:, 1]

        # face_color_function = tri.LinearTriInterpolator(triang, z1)
        # face_color_index = face_color_function(midx, midy)
        # face_color_index[face_color_index < 0] = 0
        # face_color_index /= np.max(z1)

        # cmap = get_cmap('Spectral')

        colors = np.mean( [z1[triang.triangles[:,0]], z1[triang.triangles[:,1]], z1[triang.triangles[:,2]]], axis = 0);
        # print (colors.shape)
        # color_dimension = colors # change to desired fourth dimension
        # minn, maxx = color_dimension.min(), color_dimension.max()
        # norm = matplotlib.colors.Normalize(minn, maxx)
        # print (norm)
        # print (triang.triangles)
        # print (colors)

        # # fourth dimention - colormap
        # # create colormap according to x-value (can use any 50x50 array)
        # color_dimension = colors # change to desired fourth dimension
        # minn, maxx = color_dimension.min(), color_dimension.max()
        # norm = matplotlib.colors.Normalize(minn, maxx)
        # fcolors =  cm.jet(norm(color_dimension))
        # print (fcolors.shape)


        colors = make_colormap(colors)

        if edgecolor:
            pt0=ax0.plot_trisurf(x, y, z,triangles=triang.triangles,color=cmap(face_color_index), antialiased = True, edgecolor="white")
        else:
            pt0=ax0.plot_trisurf(x, y, z,triangles=triang.triangles, antialiased = True, cmap=colors )

        # ax0.triplot(triang, 'bo-', lw=1)
        # ax0.triplot(y, z, c[flt_tri-1][:,:,1], 'go-', lw=1.0)
        ax0.set_title('triplot of Delaunay triangulation')

        cbar = fig.colorbar(pt0)
        if axis == 0:
            cbar.set_label('Shear stress (Strike) (MPa)')
        elif axis == 1:
            cbar.set_label('Shear stress (dip) (MPa)')
        elif axis == 2:
            cbar.set_label('Effective normal stress (MPa)')
        elif axis == 3:
            cbar.set_label('Pressure (MPa)')
        elif axis == 4:
            cbar.set_label(r'$ \tau / \sigma_{n} $')
        elif axis == 5:
            cbar.set_label('Total normal stress (MPa)')

        # if hline:
        #     ax0.axhline(y=hline)
        if vline:
            ax0.axvline(x=vline)
            ax1 = fig.add_subplot(2, 1, 2)
            if xyaxis == 0:
                    si = griddata((x, z), z1, ([vline], zi), method='linear')
            if xyaxis == 1:
                    si = griddata((y, z), z1, ([vline], zi), method='linear')
            ax1.plot(si,zi,'r-')
            ax1.set_ylabel('Z (km)')
            if axis == 0:
                ax1.set_xlabel('Shear stress (Strike) (MPa)')
            elif axis == 1:
                ax1.set_xlabel('Shear stress (dip) (MPa)')
            elif axis == 2:
                ax1.set_xlabel('Effective normal stress (MPa)')
            elif axis == 3:
                ax1.set_xlabel('Pressure (MPa)')
            elif axis == 4:
                ax1.set_xlabel(r'$ \tau / \sigma_{n} $')
            elif axis == 5:
                cbar.set_label('Total normal stress (MPa)')


            if si.max()>0:
                ax1.invert_xaxis()

        ax0.set_xlabel('X (km)')
        ax0.set_ylabel('Y (km)')
        ax0.set_ylabel('Z (km)')
        ax0.view_init(vdegree, azimuth)

        fig.tight_layout()
        plt.show()

       
    def plot_dyn_trac_tri(self,filename,step,axis,frame,
                        mask=0,xyaxis=0,vline=False,
                        edgecolor=False):
        
        flt_tri = np.load(filename)
        flt_tri = flt_tri - 1
        ft_pos_nodes = np.unique(flt_tri)
        a = np.arange(ft_pos_nodes.max(),dtype=np.int32)
        b = np.zeros(ft_pos_nodes.max(),dtype=np.int32)
        c = np.vstack((a,b)).T
        d = np.arange(len(ft_pos_nodes),dtype=np.int32)
        c[ft_pos_nodes-1,1] = d
       
        z = self.fcoord[:,2][:]*1e0
        x = self.fcoord[:,0][:]*1e0
        y = self.fcoord[:,1][:]*1e0
        # t_axis = np.arange(tmin,tmax,dt)
        trac_data = self.dat_trac_sort[step]

        fig = plt.figure(figsize=(15,10))
        ax0 = fig.add_subplot(1, 1, 1)
        if axis < 4:
            z1 = (trac_data[:,axis,frame])/1e6 # MPa
            
        if xyaxis == 0:
            triang = tri.Triangulation(x, z,c[flt_tri-1][:,:,1])
        else:
            triang = tri.Triangulation(y, z,c[flt_tri-1][:,:,1])
        NbLevels = 256
        if edgecolor:
            pt0=ax0.tripcolor(triang, z1, NbLevels,  edgecolor="black")
        else:
            pt0=ax0.tripcolor(triang, z1, NbLevels, )

        ax0.set_title('triplot of Delaunay triangulation')

        cbar = fig.colorbar(pt0)
        if axis == 0:
            cbar.set_label('Shear stress (Strike) (MPa)')
        elif axis == 1:
            cbar.set_label('Shear stress (dip) (MPa)')
        elif axis == 2:
            cbar.set_label('Effective normal stress (MPa)')
        elif axis == 3:
            cbar.set_label('Friction (Strike) (Mpa)')
        elif axis == 4:
            cbar.set_label('Friction (Dip) (Mpa)')
        elif axis == 5:
            cbar.set_label('Normal force (MPa)')

        if xyaxis == 0:
            ax0.set_xlabel('X (km)')
        else:
            ax0.set_xlabel('Y (km)')
        ax0.set_ylabel('Z (km)')

        fig.tight_layout()
        plt.show()

    def plot_dyn_slip_tri_group(self,tri_file_1,frame_list,step,axis,vabs=0,
                        mask=0,xyaxis=0,vline=False,zlim0=False,zlim1=False,xlim0=False,xlim1=False,
                        edgecolor=False):
        print ("printing dynamic at " + step)
        fig = plt.figure(figsize=(22,24),tight_layout=True)
        heights = [1, 1, 1, 0.08]
        gs = gridspec.GridSpec(4, 2,height_ratios=heights)
        # frame_list = np.arange(0,100,25)
        # frame_list = [10,20,40]

        # fe_file_1  = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/30_lessdp2/Buijze3D_fe.h5"
        # tri_file_1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/30/flt_tri_nob.npy"

        # fe_file_1  = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60_lessdp/Buijze3D_fe.h5"
        # tri_file_1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"

        # fe_file_1  = "/media/jingmingruan/BACKUP/TUD/delftblue/3D/3D_mfvaryingoffset/intersection_angle/45/Buijze3D_fe_1mpa.h5"
        # tri_file_1 = "/media/jingmingruan/BACKUP/TUD/delftblue/3D/3D_mfvaryingoffset/intersection_angle/45/flt_tri_nob.npy"

        # fe_file_1  = "/media/jingmingruan/BACKUP/TUD/delftblue/3D/3D_mfvaryingoffset/intersection_angle/45_01mpa/Buijze3D_fe.h5"
        # tri_file_1 = "/media/jingmingruan/BACKUP/TUD/delftblue/3D/3D_mfvaryingoffset/intersection_angle/45/flt_tri_nob.npy"

        # fe_file_1  = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/Zeerijp3D_anhydrite_4m/01_musdc/Zeerijp_fe.h5"
        # tri_file_1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/Zeerijp3D_anhydrite_4m/01_mus/flt_tri_nob.npy" # "./Zeerijp/flt_tri_nob.npy" "./Buijze3D_offset/flt_tri_nob.npy"

        # self.mdict = self.load_dict_from_hdf5(fe_file_1)
        d1_fcoord = self.fcoord
        d1_dt_dyn = self.dt_dyn
        d1_dat_trac_sort = self.dat_trac_sort
        d1_dat_slip_sort = self.dat_slip_sort

        # d1_fcoord = self.mdict['crd_flt']
        # d1_dt_dyn = self.mdict['dt_dyn']
        # d1_dat_trac_sort = self.mdict['trac_dyn']
        # d1_dat_slip_sort = self.mdict['slip_dyn']
        ngsax = 0
        for i in frame_list:
            # data 1
            # 0 strike 1 dip 2 normal 3? strike (LM) 4? dip (LM) 5? norma (LM)
            triang, z1, vmin, vmax = self.plot_dyn_trac_tri_group_func(dat_trac_sort=d1_dat_trac_sort,fcoord=d1_fcoord,
                        filename=tri_file_1,step=step,axis=6,frame=i,vabs=0,
                        mask=0,xyaxis=xyaxis,vline=False,zlim0=zlim0,zlim1=zlim1,xlim0=xlim0,xlim1=xlim1+1,
                        edgecolor=0)
            ax0 = fig.add_subplot(gs[ngsax])
            NbLevels = 256
            if edgecolor:
                pt0=ax0.tripcolor(triang, z1, NbLevels, edgecolor="black",vmin = vmin, vmax = vmax)
            else:
                pt0=ax0.tripcolor(triang, z1, NbLevels)
            if zlim0 or zlim1:
                ax0.set_ylim(zlim0,zlim1)
            if xlim0 or xlim1:
                ax0.set_xlim(xlim0,xlim1)
            pt0.set_clim(0,0.55) 
            # cbar = fig.colorbar(pt0)
            # cbar.set_label(r'$ \tau / \sigma_{n} $')
            if i == frame_list[-1]:
                axes = plt.subplot(gs[3,0])
                cbar = plt.colorbar(pt0, cax=axes,orientation="horizontal", pad=0)
                # cbar.set_label(r'$ \tau / {\sigma_{n}}^{\prime} $')
                cbar.set_label('Shear stress/effective normal stress')
            ax0.set_xlabel('Y (km)')
            ax0.set_ylabel('Depth (km)')
            # ax0.set_title('t = ' + str(i*0.01) + ' s')


            # data 2
            # fe_file_1  = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/Buijze3D_fe.h5"
            # tri_file_1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/3D_mfvaryoffset/intersect_angle/run/60/flt_tri_nob.npy"
            # d1_mdict = self.load_dict_from_hdf5(fe_file_1)

            # step 23
            triang, z1, vmin, vmax = self.plot_dyn_slip_tri_group_func(dat_trac_sort=d1_dat_slip_sort,fcoord=d1_fcoord,
                        filename=tri_file_1,step=step,axis=3,frame=i,vabs=0,
                        mask=0,xyaxis=xyaxis,vline=False,zlim0=zlim0,zlim1=zlim1,xlim0=xlim0-1,xlim1=xlim1+1,
                        edgecolor=0)
            ax0 = fig.add_subplot(gs[ngsax+1])
            NbLevels = 256
            if edgecolor:
                pt0=ax0.tripcolor(triang, z1, NbLevels, edgecolor="black",vmin = vmin, vmax = vmax)
            else:
                pt0=ax0.tripcolor(triang, z1, NbLevels, vmin = vmin, vmax = vmax)

            if zlim0 or zlim1:
                ax0.set_ylim(zlim0,zlim1)
            if xlim0 or xlim1:
                ax0.set_xlim(xlim0,xlim1)
            if i == frame_list[-1]:
                axes = plt.subplot(gs[3,1])
                cbar = plt.colorbar(pt0, cax=axes,orientation="horizontal", pad=0.2)
                cbar.set_label('Relative slip  (m)')

            ax0.set_xlabel('Y (km)')
            ax0.set_ylabel('Depth (km)')
            # ax0.set_title('t = ' + str(i*0.01) + ' s')
            # pt0.set_clim(0,0.02)

            ngsax += 2

        fig.tight_layout()
        plt.show()

    def plot_dyn_trac_tri_group_func(self,dat_trac_sort,fcoord,filename,step,axis,frame,vabs=0,
                        mask=0,xyaxis=0,vline=False,zlim0=False,zlim1=False,xlim0=False,xlim1=False,
                        edgecolor=False):
        
        flt_tri = np.load(filename)
        flt_tri = flt_tri - 1
        ft_pos_nodes = np.unique(flt_tri)
        a = np.arange(ft_pos_nodes.max(),dtype=np.int32)
        b = np.zeros(ft_pos_nodes.max(),dtype=np.int32)
        c = np.vstack((a,b)).T
        d = np.arange(len(ft_pos_nodes),dtype=np.int32)
        c[ft_pos_nodes-1,1] = d
       
        z = fcoord[:,2][:]*1e0
        x = fcoord[:,0][:]*1e0
        y = fcoord[:,1][:]*1e0
        slip_data = dat_trac_sort[step]/1e6
        print (slip_data.shape)

        if axis == 6:
            z1 = np.abs(np.sqrt(slip_data[:,3,frame]**2+slip_data[:,4,frame]**2)/slip_data[:,5,frame])

        _min, _max = np.amin((slip_data[:,0,:])), np.amax((slip_data[:,0,:]))

        if xyaxis == 0:
            triang = tri.Triangulation(x, z,c[flt_tri-1][:,:,1])
        else:
            triang = tri.Triangulation(y, z,c[flt_tri-1][:,:,1])

        return triang, z1, _min, _max

    def plot_dyn_slip_tri_group_func(self,dat_trac_sort,fcoord,filename,step,axis,frame,vabs=0,
                        mask=0,xyaxis=0,vline=False,zlim0=False,zlim1=False,xlim0=False,xlim1=False,
                        edgecolor=False):
        
        flt_tri = np.load(filename)
        flt_tri = flt_tri - 1
        ft_pos_nodes = np.unique(flt_tri)
        a = np.arange(ft_pos_nodes.max(),dtype=np.int32)
        b = np.zeros(ft_pos_nodes.max(),dtype=np.int32)
        c = np.vstack((a,b)).T
        d = np.arange(len(ft_pos_nodes),dtype=np.int32)
        c[ft_pos_nodes-1,1] = d
       
        z = fcoord[:,2][:]*1e0
        x = fcoord[:,0][:]*1e0
        y = fcoord[:,1][:]*1e0
        slip_data = dat_trac_sort[step]
        print (slip_data.shape)

        if axis < 3:
            z1 = (slip_data[:,axis,frame]) # MPa
        elif axis == 3:
            z1 = np.sqrt(slip_data[:,0,frame]**2+slip_data[:,1,frame]**2)
        if vabs: z1 = np.abs(z1)
        if axis <3:
            _min, _max = np.amin((slip_data[:,axis,:])), np.amax((slip_data[:,axis,:]))
        if axis == 3:
            slip_amp = np.sqrt(slip_data[:,0,:]**2+slip_data[:,1,:]**2)
            _min, _max = np.amin(np.abs(slip_amp)), np.amax(np.abs(slip_amp))

        if xyaxis == 0:
            triang = tri.Triangulation(x, z,c[flt_tri-1][:,:,1])
        else:
            triang = tri.Triangulation(y, z,c[flt_tri-1][:,:,1])
        return triang, z1, _min, _max

        
    def plot_dyn_slip_tri(self,filename,step,axis,frame,vabs=0,
                        mask=0,xyaxis=0,vline=False,zlim0=False,zlim1=False,xlim0=False,xlim1=False,
                        edgecolor=False):
        
        flt_tri = np.load(filename)
        flt_tri = flt_tri - 1
        ft_pos_nodes = np.unique(flt_tri)
        a = np.arange(ft_pos_nodes.max(),dtype=np.int32)
        b = np.zeros(ft_pos_nodes.max(),dtype=np.int32)
        c = np.vstack((a,b)).T
        d = np.arange(len(ft_pos_nodes),dtype=np.int32)
        c[ft_pos_nodes-1,1] = d
       
        z = self.fcoord[:,2][:]*1e0
        x = self.fcoord[:,0][:]*1e0
        y = self.fcoord[:,1][:]*1e0
        # t_axis = np.arange(tmin,tmax,dt)
        slip_data = self.dat_slip_sort[step]
        print (slip_data.shape)

        fig = plt.figure(figsize=(15,10))
        ax0 = fig.add_subplot(1, 1, 1)
        if axis < 3:
            z1 = (slip_data[:,axis,frame]) # MPa
        elif axis == 3:
            z1 = np.sqrt(slip_data[:,0,frame]**2+slip_data[:,1,frame]**2)
        if vabs: z1 = np.abs(z1)
        if axis <3:
            _min, _max = np.amin((slip_data[:,axis,:])), np.amax((slip_data[:,axis,:]))
        if axis == 3:
            slip_amp = np.sqrt(slip_data[:,0,:]**2+slip_data[:,1,:]**2)
            _min, _max = np.amin(np.abs(slip_amp)), np.amax(np.abs(slip_amp))

        if xyaxis == 0:
            triang = tri.Triangulation(x, z,c[flt_tri-1][:,:,1])
        else:
            triang = tri.Triangulation(y, z,c[flt_tri-1][:,:,1])
        NbLevels = 256
        if edgecolor:
            pt0=ax0.tripcolor(triang, z1, NbLevels, edgecolor="black",vmin = _min, vmax = _max)
        else:
            pt0=ax0.tripcolor(triang, z1, NbLevels, vmin = _min, vmax = _max)

        # ax0.triplot(triang, 'bo-', lw=1)
        # ax0.triplot(y, z, c[flt_tri-1][:,:,1], 'go-', lw=1.0)
        # ax0.set_title('triplot of Delaunay triangulation')

        if zlim0 or zlim1:
            ax0.set_ylim(zlim0,zlim1)
        if xlim0 or xlim1:
            ax0.set_xlim(xlim0,xlim1)
        cbar = fig.colorbar(pt0)
        if axis == 0:
            cbar.set_label('Shear slip (Strike) (m)')
        elif axis == 1:
            cbar.set_label('Shear slip (dip) (m)')
        elif axis == 2:
            cbar.set_label('Normal displacement (m)')
        elif axis == 3:
            cbar.set_label('Shear slip amplitude (m)')
        # elif axis == 3:
        #     cbar.set_label('Friction (Strike) (Mpa)')
        # elif axis == 4:
        #     cbar.set_label('Friction (Dip) (Mpa)')
        # elif axis == 5:
        #     cbar.set_label('Normal force (MPa)')

        if xyaxis == 0:
            ax0.set_xlabel('Strike (km)')
        else:
            ax0.set_xlabel('Strike (km)')
        ax0.set_ylabel('Depth (km)')

        fig.tight_layout()
        plt.show()

    def plot_dyn_slip_ani(self,filename,step,axis,frame,vabs=0,
                        mask=0,xyaxis=0,vline=False,zlim0=False,zlim1=False,xlim0=False,xlim1=False,
                        edgecolor=False):
        # print (frame)
        flt_tri = np.load(filename)
        flt_tri = flt_tri - 1
        ft_pos_nodes = np.unique(flt_tri)
        a = np.arange(ft_pos_nodes.max(),dtype=np.int32)
        b = np.zeros(ft_pos_nodes.max(),dtype=np.int32)
        c = np.vstack((a,b)).T
        d = np.arange(len(ft_pos_nodes),dtype=np.int32)
        c[ft_pos_nodes-1,1] = d
       
        z = self.fcoord[:,2][:]*1e0
        x = self.fcoord[:,0][:]*1e0
        y = self.fcoord[:,1][:]*1e0
        # t_axis = np.arange(tmin,tmax,dt)
        slip_data = self.dat_slip_sort[step]
        # fig.clear()
        # ax = plt.subplot(1,1,1)


        if axis < 3:
            z1 = (slip_data[:,axis,frame]) # MPa
        if axis == 3:
            # z1 = np.sqrt((slip_data[:,0,:frame])**2 + (slip_data[:,1,:frame])**2)
            dat_int = np.sum(slip_data[:,:2,:frame],axis=2)*self.dt_dyn
            z1 =  np.linalg.norm(dat_int,axis=1)
        if vabs: z1 = np.abs(z1)
        _min, _max = np.amin((slip_data[:,axis,:])), np.amax((slip_data[:,axis,:]))

        if xyaxis == 0:
            triang = tri.Triangulation(x, z,c[flt_tri-1][:,:,1])
        else:
            triang = tri.Triangulation(y, z,c[flt_tri-1][:,:,1])
        NbLevels = 256
        if edgecolor:
            pt0 = plt.tripcolor(triang, z1, NbLevels, edgecolor="black",vmin = _min, vmax = _max)
        else:
            pt0 = plt.tripcolor(triang, z1, NbLevels, vmin = _min, vmax = _max)

        # ax0.triplot(triang, 'bo-', lw=1)
        # ax0.triplot(y, z, c[flt_tri-1][:,:,1], 'go-', lw=1.0)
        # plt.set_title('triplot of Delaunay triangulation')
        if zlim0 or zlim1:
            plt.ylim(zlim0,zlim1)
        if xlim0 or xlim1:
            plt.xlim(xlim0,xlim1)
        if frame==240:
            cbar = plt.colorbar(pt0)
            cbar.set_label('Slip (m)')

        # fig.tight_layout()
        # plt.show()

        
    def plot_dyn_slip_ct(self,step,ymin,ymax,zmin,zmax,dsp=0,
                        xyaxis=0,vline=False,
                        edgecolor=False):
        tol = 0
        dt = 0
        CropY=[0,0]; CropZ=[0,0]; ResY=1000; ResZ=1000; t_plt=16.
        ang_dip=np.pi/180*66
        fcoord   = self.fcoord
        dt_slip  = self.dt_dyn
        dat_slip = self.dat_slip_sort[step]

        
        len_tot = dat_slip.shape[-1]
        cnt = 1E9*np.ones(shape=dat_slip[:,0,0].shape)
        if dsp==1: dat = np.linalg.norm(dat_slip[:,:2,:],axis=1)
        for i in range(len_tot):
            t = i*dt_slip+dt
            if dsp==1:
                if h5:
                    slip = dat[:,i]
                else:
                    slip = dat[i,:]
            else:
                # Time integral for velocity output
                dat_int = np.sum(dat_slip[:,:2,:i+1],axis=2)*dt_slip
                slip = np.linalg.norm(dat_int,axis=1)
            idrup = np.squeeze(np.where(slip>tol))
            idfrn = np.squeeze(np.where(cnt<1E9))
            set_rup = np.setdiff1d(idrup,idfrn)
            

            
            cnt[set_rup] = t
            if i>int(len_tot*0.15) and set_rup.size==0:
                t_rup=t
                break
        
        yi = np.linspace(ymin, ymax, ResY)
        zi = np.linspace(zmin, zmax, ResZ)
        
        yi,zi = np.meshgrid(yi,zi)
        # dip=zi/np.sin(ang_dip)
        dip = zi
        dat_grid = griddata((np.squeeze(fcoord[:,1]), np.squeeze(fcoord[:,2])), cnt, (yi, zi), method='linear')
        # dat_grid_ES = griddata((dat_ES[:,0], dat_ES[:,1]), dat_ES[:,2], (yi*1E3, -dip*1E3), method='linear')
        # dat_grid_FM = griddata((dat_FM[:,0], dat_FM[:,1]), dat_FM[:,2], (yi*1E3, -dip*1E3), method='linear')
        # fig, ax = plt.subplots()
        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(1, 1, 1)
        #t_rup=2.
        dt_rup=0.2
        levels = np.arange(0.,t_rup+dt_rup,dt_rup)
        cs1=plt.contour(yi, dip, dat_grid,    colors='b', linestyles='--',linewidths=1.,levels=levels)

        ax.clabel(cs1, inline=1, fmt = '%1.1f',fontsize=10)

        lines = [cs1.collections[0]]

        labels = ['defmod']

        plt.legend(lines, labels, loc=4, fontsize=10.)
        plt.xlabel('strike [km]')
        plt.ylabel('dip [km]')
        plt.ylabel('Z [km]')
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.title('SCEC%d, dt = %0.1f [s]'%(idcase,dt_rup))
        # plt.tight_layout()
        # plt.savefig(name_sol+'_rup'+'.png')
        #plt.show()
        
        
        
    def plot_dyn_slip_ct_list(self,steps,ymin,ymax,zmin,zmax,dsp=0,
                        xyaxis=0,vline=False,
                        edgecolor=False):
        tol = 0.40
        dt = 0
        CropY=[0,0]; CropZ=[0,0]; ResY=1000; ResZ=1000; t_plt=16.
        ang_dip=np.pi/180*66
        fcoord   = self.fcoord
        dt_slip  = self.dt_dyn
        # for step in steps:
        dat_slip = np.sqrt(self.dat_trac_sta[:,0,:]**2+self.dat_trac_sta[:,1,:]**2)/self.dat_trac_sta[:,2,:]
        dat_slip = np.abs(dat_slip)
        cnt = 1E9*np.ones(shape=dat_slip[:,0].shape)
        dat = dat_slip
        for i in steps:
            t = i
            slip = dat_slip[:,i]
            # slip = np.linalg.norm(dat_slip[:,i],axis=1)
            
            idrup = np.squeeze(np.where(slip>tol))
            idfrn = np.squeeze(np.where(cnt<1E9))
            set_rup = np.setdiff1d(idrup,idfrn)
            cnt[set_rup] = t
            # if i>int(len_tot*0.15) and set_rup.size==0:
            #     t_rup=t
            #     break


        yi = np.linspace(ymin, ymax, ResY)
        zi = np.linspace(zmin, zmax, ResZ)

        yi,zi = np.meshgrid(yi,zi)
        # dip=zi/np.sin(ang_dip)
        dip = zi
        dat_grid = griddata((np.squeeze(fcoord[:,0]), np.squeeze(fcoord[:,2])), cnt, (yi, zi), method='linear')

        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(1, 1, 1)
        # t_rup=0
        # dt_rup=1
        levels = np.arange(0.,len(steps),1)
        cs1=plt.contour(yi, dip, dat_grid,    colors='b', linestyles='--',linewidths=1.,levels=levels)

        ax.clabel(cs1, inline=1, fmt = '%1.1f',fontsize=10)

        lines = [cs1.collections[0]]
        # labels = [str(step)]
        # plt.legend(lines, labels, loc=4, fontsize=10.)
        plt.xlabel('strike [km]')
        plt.ylabel('dip [km]')
        plt.ylabel('Z [km]')
            # plt.gca().set_aspect('equal', adjustable='box')
            # plt.title('SCEC%d, dt = %0.1f [s]'%(idcase,dt_rup))
            # plt.tight_layout()
            # plt.savefig(name_sol+'_rup'+'.png')
            #plt.show()
        
    def inter_tri_2D(self,filename,tstep_list,axis,delta=False,
                        mask=0,xyaxis=0):
        flt_tri = np.load(filename)
        flt_tri = flt_tri - 1
        ft_pos_nodes = np.unique(flt_tri)
        a = np.arange(ft_pos_nodes.max(),dtype=np.int32)
        b = np.zeros(ft_pos_nodes.max(),dtype=np.int32)
        c = np.vstack((a,b)).T
        d = np.arange(len(ft_pos_nodes),dtype=np.int32)
        c[ft_pos_nodes-1,1] = d
       
        x = self.fcoord[:,0][:]*1e0
        y = self.fcoord[:,1][:]*1e0
        z = self.fcoord[:,2][:]*1e0
        
        print (c[flt_tri-1].shape)
        print (c[flt_tri-1][:,:,1].shape)
        print (np.mean(x[c[flt_tri-1][:,:,1]],axis=1).shape)
        
        tri_cnt_x = np.mean(x[c[flt_tri-1][:,:,1]],axis=1)
        tri_cnt_y = np.mean(y[c[flt_tri-1][:,:,1]],axis=1)
        tri_cnt_z = np.mean(z[c[flt_tri-1][:,:,1]],axis=1)
        
        print (tri_cnt_x)
        print (tri_cnt_y)
        print (tri_cnt_z)
        
#         ResX = 100; ResY = 100; ResZ = 100;

#         xi = np.linspace(x.min, x.max, ResX)
#         yi = np.linspace(y.min, y.max, ResY)
#         zi = np.linspace(z.min, z.max, ResZ)
        
#         for xxi in xi:
#             for yyi in yi:
#                 coordi = np.array([xi,yi])


    # def make_colormap(seq):
    #     """Return a LinearSegmentedColormap
    #     seq: a sequence of floats and RGB-tuples. The floats should be increasing
    #     and in the interval (0,1).
    #     """
    #     #%
    #     cdict = {'red': [], 'green': [], 'blue': []}

    #     # make a lin_space with the number of records from seq.     
    #     x = np.linspace(0,1, len(seq))
    #     #%
    #     for i in range(len(seq)):
    #         segment = x[i]
    #         tone = seq[i]
    #         cdict['red'].append([segment, tone, tone])
    #         cdict['green'].append([segment, tone, tone])
    #         cdict['blue'].append([segment, tone, tone])
    #     #%
    #     return mcolors.LinearSegmentedColormap('CustomMap', cdict)

# Calculate the moment magnitude of the a seismic event
# get the final frame of the dynamic simulation
# calculate the maximum slip, and the slip of the element (using integration)
# using summation over the all the element within the slip patch, with the shear modulus
# Improve: calculate only using the seismci slip patch, and excluding all the aseismic slip patch afte the step0 
# Improve: using Gaussian quadrature integral.
    def getML(self,step,xyaxis=0,vline=False,zlim0=False,zlim1=False,xlim0=False,xlim1=False,edgecolor=False):
        for i in [100]:
            # data 1
            fe_file_1  = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/Zeerijp3D_anhydrite_4m/01_mus/Zeerijp_fe.h5"
            tri_file_1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/Zeerijp3D_anhydrite_4m/01_mus/flt_tri_nob.npy" # "./Zeerijp/flt_tri_nob.npy" "./Buijze3D_offset/flt_tri_nob.npy"

            d1_fcoord = self.mdict['crd_flt']
            d1_dt_dyn = self.mdict['dt_dyn']
            d1_dat_trac_sort = self.mdict['trac_dyn']
            d1_dat_slip_sort = self.mdict['slip_dyn']

            # step 23
            self.getML_func(dat_trac_sort=d1_dat_slip_sort,fcoord=d1_fcoord,
                        filename=tri_file_1,step=step,axis=3,frame=i,vabs=0,
                        mask=0,xyaxis=xyaxis,vline=False,zlim0=zlim0,zlim1=zlim1,xlim0=xlim0-1,xlim1=xlim1+1,
                        edgecolor=0)


    def getML_func(self,dat_trac_sort,fcoord,filename,step,axis,frame,vabs=0,
                        mask=0,xyaxis=0,vline=False,zlim0=False,zlim1=False,xlim0=False,xlim1=False,
                        edgecolor=False):
        
        flt_tri = np.load(filename)
        flt_tri = flt_tri - 1
        ft_pos_nodes = np.unique(flt_tri)
        a = np.arange(ft_pos_nodes.max(),dtype=np.int32)
        b = np.zeros(ft_pos_nodes.max(),dtype=np.int32)
        c = np.vstack((a,b)).T
        d = np.arange(len(ft_pos_nodes),dtype=np.int32)
        c[ft_pos_nodes-1,1] = d
       
        z = fcoord[:,2][:]*1e0
        x = fcoord[:,0][:]*1e0
        y = fcoord[:,1][:]*1e0

        slip_data = dat_trac_sort[step]
        z1 = np.sqrt(slip_data[:,0,frame]**2+slip_data[:,1,frame]**2)

        if axis == 3:
            slip_amp = np.sqrt(slip_data[:,0,:]**2+slip_data[:,1,:]**2)
            _min, _max = np.amin(np.abs(slip_amp)), np.amax(np.abs(slip_amp))

        tri_slip = slip_amp[c[flt_tri-1][:,:,1]][:,:,frame]
        tri_x = x[c[flt_tri-1][:,:,1]]*1000
        tri_y = y[c[flt_tri-1][:,:,1]]*1000
        tri_z = z[c[flt_tri-1][:,:,1]]*1000
        tri_x_el = np.mean(tri_x,axis=1)
        tri_y_el = np.mean(tri_y,axis=1)
        tri_z_el = np.mean(tri_z,axis=1)

        # gaussian quadrature integration intergration 
        # area is calculated using x z coordinates
        Area = np.abs(1/2*(tri_x[:,0]*(tri_z[:,1] - tri_z[:,2]) + tri_x[:,1]*(tri_z[:,2] \
            - tri_z[:,0]) + tri_x[:,2]*(tri_z[:,0] - tri_z[:,1])))

        # Gaussian integral (local coordinates?)
        mu = 6e9 # shear modulus 6 GPa
        weight = [1/3, 1/3, 1/3]
        slip_el = np.zeros(len(flt_tri))
        M0 = 0
        # summation
        for i in range(len(flt_tri)):
            slip_el[i] = np.dot(weight[:],tri_slip[i,:])
            M0 += slip_el[i]*Area[i]*mu*2 # Rick Wentick assume E_dyn = E * 2

        ML = (np.log10(M0)-9.1)/1.5

        print (slip_el.shape)
        print (slip_el)
        print (slip_el.max())
        print (slip_el.min())
        print (Area.max())
        print (Area.min())
        print (ML)


# calculate the initial stress has some problems. For example, the maximum shear stress direction is definitly not the same in
# the slip patch. Therefore, the slip patch length ccould only be approximated assumed one directionla vector.
# Another tehcnical issue is that, how do you know how long the slip patch?
# first, check the static slip distance of the fault. If the slip distance is 0 then the weaknening is not weaknening. If not, 
# then the critical nodes should have the decayed ratio between shear stress and the effective normal stress.
# Notice that, it is possible that inside one slip patch, there is a fully weakened (or not weakened node) that seperate the slip patch
# which results in two slip patch, and therefore the seismic slip is not happening. However, conisdering the result in the resevoir region
# where the upper and lower slip patch is spereated by the less critical region. So, this might just return to the scale problem.
# for example, if i have a mesh that is small enough to seperate the previous slip patch which is a whole in the rough resoutlion mesh.
# So this should be a scale problem. Or the analytical soution is just not working in this case.


    def getLnuc(self,step,xyaxis=0,vline=False,zlim0=False,zlim1=False,xlim0=False,xlim1=False,edgecolor=False):
        for i in [0]:
            # data 1
            fe_file_1  = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/Zeerijp3D_anhydrite_4m/01_musdc/Zeerijp_fe.h5"
            tri_file_1 = "/media/jingmingruan/BACKUP/TUD/texel/3D_models/Zeerijp3D_anhydrite_4m/01_mus/flt_tri_nob.npy" # "./Zeerijp/flt_tri_nob.npy" "./Buijze3D_offset/flt_tri_nob.npy"

            d1_fcoord = self.mdict['crd_flt']
            d1_dt_dyn = self.mdict['dt_dyn']
            d1_dat_trac_sort = self.mdict['trac_dyn']
            d1_dat_slip_sort = self.mdict['slip_dyn']

            for j in range(29):
                self.getLnuc_func(fcoord=d1_fcoord,
                            filename=tri_file_1,step=j,axis=3,frame=i,vabs=0,
                            mask=0,xyaxis=xyaxis,vline=False,zlim0=zlim0,zlim1=zlim1,xlim0=xlim0-1,xlim1=xlim1+1,
                            edgecolor=0)


# how to calculate the initial slip patch length?
# 1. check the first time step(maybe not the dynamic time step, because it is not accurate because of the zero-round-off dispalcement)
# 2. check the maximum shear stress location (take the average? or the maximum)
# 3. translate the slip patch length in the vector of the maximum shear stress
# 4. make the dot production, which is the projection. 
    def getLnuc_func(self,fcoord,filename,step,axis,frame,vabs=0,
                        mask=0,xyaxis=0,vline=False,zlim0=False,zlim1=False,xlim0=False,xlim1=False,
                        edgecolor=False):
        
        flt_tri = np.load(filename)
        flt_tri = flt_tri - 1
        ft_pos_nodes = np.unique(flt_tri)
        a = np.arange(ft_pos_nodes.max(),dtype=np.int32)
        b = np.zeros(ft_pos_nodes.max(),dtype=np.int32)
        c = np.vstack((a,b)).T
        d = np.arange(len(ft_pos_nodes),dtype=np.int32)
        c[ft_pos_nodes-1,1] = d
       
        z = fcoord[:,2][:]*1e0
        x = fcoord[:,0][:]*1e0
        y = fcoord[:,1][:]*1e0

        slip_data = self.mdict['slip_sta'][:,:,step] # should this be one time step earlier?
        trac_data = self.mdict['trac_sta'][:,:,step]
        # print (slip_data.shape)
        # print (slip_data[:,0].max()) # strike
        # print (slip_data[:,1].max()) # dip 
        # print (slip_data[:,2].max()) # normal

        slip_data = np.sqrt(slip_data[:,0]**2 + slip_data[:,1]**2)
        trac_max = np.sqrt(trac_data[:,0]**2+trac_data[:,1]**2)
        flt_fc = trac_max/np.abs(trac_data[:,2]) 

        # print (slip_data.shape) # all
        # print (slip_data[:].max()) # all
        # print (slip_data[:].min()) # all

        # print (flt_fc.shape) # all
        # print (flt_fc[:].max()) # all
        # print (flt_fc[:].min()) # all

        # print (trac_data.shape)
        # print (trac_data[0].max()/1e6) # strike # dip # normal
        # print (trac_data[0].min()/1e6) 
        # print (trac_data[1].max()/1e6) # strike # dip # normal
        # print (trac_data[1].min()/1e6) 
        # print (trac_data[2].max()/1e6) # strike # dip # normal
        # print (trac_data[2].min()/1e6) 
        # print (trac_data[3].max()/1e6) # strike # dip # normal
        # print (trac_data[3].min()/1e6) 

        mus = 0.55
        mud = 0.4
        fcd = 0.01 # there is a problem about the more slip that dc 
        slip_data = np.abs(slip_data)
        for i in range(len(slip_data)):
            if (slip_data[i]) > fcd:
                slip_data[i] = fcd
        mu_dyn = np.ones(len(slip_data)) * mus - slip_data/fcd * (mus-mud)

        count = 0
        count_dead = 0
        crt_node = np.zeros((len(slip_data)))
        for i in range(len(flt_fc)):
            if slip_data[i]>0 and mu_dyn[i]>mud: # not taking the new critical node here!!
                if flt_fc[i] - mu_dyn[i] >= 0:
                    crt_node[i] = 1
                    count += 1
            elif slip_data[i]==0 and mu_dyn[i]>=mus:
                if flt_fc[i] - mu_dyn[i] >= 0:
                    crt_node[i] = 1
                    count += 1
            elif slip_data[i]>0 and mu_dyn[i]==mud:
                if flt_fc[i] - mu_dyn[i] >= 0:
                    crt_node[i] = 1
                    count_dead += 1


        # print (mu_dyn.max(),mu_dyn.min())
        print ('============')
        print ('not dead>>')
        print (count,count/len(fcoord)*100)
        print ('dead >>')
        print (count_dead,count_dead/len(fcoord)*100)

        # trac_max = np.sqrt(trac_data[:,0]**2+trac_data[:,1]**2)

        # elif axis == 4:
        #     z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2)/self.dat_trac_sta[:,2,i-1]
        #     z1 = np.abs(z1)
        #     if mask:
        #         mask_ = z1>=mask
        #         print (str(np.sum(mask_)) + "("+ str(np.sum(mask_)/len(z1)*100) + ")"+"nodes above the mask")
        #         z1[~mask_] = 0

        # if axis == 3:
        #     slip_amp = np.sqrt(slip_data[:,0,:]**2+slip_data[:,1,:]**2)
        #     _min, _max = np.amin(np.abs(slip_amp)), np.amax(np.abs(slip_amp))

        # tri_slip = slip_amp[c[flt_tri-1][:,:,1]][:,:,frame]
        # tri_x = x[c[flt_tri-1][:,:,1]]*1000
        # tri_y = y[c[flt_tri-1][:,:,1]]*1000
        # tri_z = z[c[flt_tri-1][:,:,1]]*1000
        # tri_x_el = np.mean(tri_x,axis=1)
        # tri_y_el = np.mean(tri_y,axis=1)
        # tri_z_el = np.mean(tri_z,axis=1)

        # # gaussian quadrature integration intergration 
        # # area is calculated using x z coordinates
        # Area = np.abs(1/2*(tri_x[:,0]*(tri_z[:,1] - tri_z[:,2]) + tri_x[:,1]*(tri_z[:,2] \
        #     - tri_z[:,0]) + tri_x[:,2]*(tri_z[:,0] - tri_z[:,1])))

        # # Gaussian integral (local coordinates?)
        # weight = [1/3, 1/3, 1/3]
        # slip_el = np.zeros(len(flt_tri))
        # # summation
        # for i in range(len(flt_tri)):
        #     slip_el[i] = np.dot(weight[:],tri_slip[i,:])
        #     M0 += slip_el[i]*Area[i]*mu

        # print (Area.min()
        # print (ML)


    def plot_induced_static_tri_group(self,filename,tstep_list,delta=False,
                            scatter=False,mask=0,xyaxis=0,vline=False,
                            edgecolor=False,xlim0=False,xlim1=False,zlim0=False,zlim1=False,interp=False):
        fig = plt.figure(figsize=(20,14),tight_layout=True)
        gs = gridspec.GridSpec(2, 2)
        z = self.fcoord[:,2][:]*1e0
        x = self.fcoord[:,0][:]*1e0
        y = self.fcoord[:,1][:]*1e0
        if interp:
            xi, yi = np.meshgrid(np.linspace(x.min(), x.max(), 1000), np.linspace(z.min(), z.max(), 1000))
        for i in range(4):
            print ("plotting figure " + str(i))
            ax1 = fig.add_subplot(gs[i])
            triang, z1, cbar_label = self.plot_induced_static_tri_group_func(filename=filename,tstep_list=tstep_list,
                                                                     axis=i,delta=delta,xyaxis=xyaxis)

            # set mask for the triangles
            # triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
            #                          y[triang.triangles].mean(axis=1))
            #                 < min_radius)

            if interp and i==3:
                # interp_lin = tri.LinearTriInterpolator(triang, z1)
                # # interp_lin = tri.CubicTriInterpolator(triang, z1,kind='geom')
                # zi_lin = interp_lin(xi, yi)
                # pt0=ax1.contourf(xi, yi, zi_lin)
                # pt0=ax1.plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
                # pt0=ax1.plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
                # contour figure
                # levels = np.linspace(0,1.1,12) # levels = np.linspace(0,1.1,12)


                # levels = np.linspace(0,1,21)
                # levels = np.concatenate((levels,np.array([1.01])))
                # print (levels)
                # pt0 = ax1.tricontourf(triang, z1,levels=levels,vmin=0, vmax=1)

                pt0=ax1.tripcolor(triang, z1, shading='flat',cmap='Reds' )


                # contour plot with SCU=1 
                # levels = np.array([0.999,1.0]) # levels = np.array([0,0.2,0.3,0.9,1.0,1.1])
                levels = np.array([-1,-0.999,0.999,1.0]) # levels = np.array([0,0.2,0.3,0.9,1.0,1.1])
                ax1.tricontour(triang, z1,levels=levels, colors='k')
                #ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
            else:
                vmax = np.abs(z1).max()
                if edgecolor:
                    pt0=ax1.tripcolor(triang, z1, shading='flat', edgecolor="black",vmin=-1*vmax, vmax=vmax) # cmap=plt.cm.jet,
                else:
                    pt0=ax1.tripcolor(triang, z1, shading='flat',vmin=-1*vmax, vmax=vmax) # cmap=plt.cm.jet,

            cbar = fig.colorbar(pt0)
            cbar.set_label(cbar_label)
            if zlim0 or zlim1:
                ax1.set_ylim(zlim0,zlim1)
            if xlim0 or xlim1:
                ax1.set_xlim(xlim0,xlim1)
            # ax1.set_xlabel('Strike (km)')
            ax1.set_xlabel('X (km)')
            ax1.set_ylabel('Depth (km)')


    def plot_induced_static_tri_group_func(self,filename,tstep_list,axis,delta=False,
                        scatter=False,mask=0,xyaxis=0,vline=False,
                        edgecolor=False,xlim0=False,xlim1=False,zlim0=False,zlim1=False):
        flt_tri = np.load(filename)
        flt_tri = flt_tri - 1
        ft_pos_nodes = np.unique(flt_tri)
        a = np.arange(ft_pos_nodes.max(),dtype=np.int32)
        b = np.zeros(ft_pos_nodes.max(),dtype=np.int32)
        c = np.vstack((a,b)).T
        d = np.arange(len(ft_pos_nodes),dtype=np.int32)
        c[ft_pos_nodes-1,1] = d
       
        z = self.fcoord[:,2][:]*1e0
        x = self.fcoord[:,0][:]*1e0
        y = self.fcoord[:,1][:]*1e0

        i = tstep_list[0]
        print (np.abs(self.dat_trac_sta[:,3,i-1]-self.dat_trac_sta[:,3,1]).max())
        weak_done = 0

        if axis < 3:# pressure is 3
            z1 = (self.dat_trac_sta[:,axis,i-1])/1e6
            if delta:
                z1 = z1 - (self.dat_trac_sta[:,axis,1])/1e6
        # SCU 
        elif axis == 3:
            dc = 0.005
            mu_node = 0.6*np.ones(len(self.dat_slip_sta[:,1,i-1])) # friction coefficient 
            slip_node = np.sqrt(self.dat_slip_sta[:,0,i-1]**2+self.dat_slip_sta[:,1,i-1]**2) # total aseismic slip            
            for j in range(len(mu_node)):
                if slip_node[j]>dc:
                    mu_node[j] = 0.45
                    weak_done = 1
                else:
                    mu_node[j] = 0.6 - 0.15*(slip_node[j]/dc)

            z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2)/self.dat_trac_sta[:,2,i-1]/mu_node
            z1 = np.abs(z1)

            # find fully weakened node
            for j in range(len(mu_node)):
                if slip_node[j]>=dc:
                    z1[j] = -1

            z1[z1>1] = 1
            if weak_done: print ('completed weakening') 
        # difference on SCU
        elif axis == 5:
            dc = 0.005
            mu_node = 0.6*np.ones(len(self.dat_slip_sta[:,1,i-1])) # friction coefficient 
            slip_node = np.sqrt(self.dat_slip_sta[:,0,i-1]**2+self.dat_slip_sta[:,1,i-1]**2) # total aseismic slip

            
            for j in range(len(mu_node)):
                if slip_node[j]>dc:
                    mu_node[j] = 0.45
                    print ('good')
                    z1[j] = -1
                else:
                    mu_node[j] = 0.6 - 0.15*(slip_node[j]/dc)

            z1 = np.sqrt(self.dat_trac_sta[:,0,i-1]**2+self.dat_trac_sta[:,1,i-1]**2)/self.dat_trac_sta[:,2,i-1]/mu_node
            z1 = np.abs(z1)
                
        if xyaxis == 0:
            triang = tri.Triangulation(x, z,c[flt_tri-1][:,:,1])
        else:
            triang = tri.Triangulation(y, z,c[flt_tri-1][:,:,1])

        if delta:
            if axis == 0:
                cbar_label = ('Incremental shear stress (strike) (MPa)')
            elif axis == 1:
                cbar_label = ('Incremental shear stress (dip) (MPa)')
            elif axis == 2:
                cbar_label = ('Incremental effective normal stress (MPa)')
            elif axis == 3:
                cbar_label = ('SCU')
            elif axis == 4:
                cbar_label = (r'$ {\tau / \sigma_{n}}^{\prime} $')
            elif axis == 5:
                cbar_label = ('Total normal stress (MPa)')
            elif axis == 6:
                cbar_label = ('Total shear stress (MPa)')
        else:
            if axis == 0:
                cbar_label = ('Shear stress (strike) (MPa)')
            elif axis == 1:
                cbar_label = ('Shear stress (dip) (MPa)')
            elif axis == 2:
                cbar_label =('Effective normal stress (MPa)')
            elif axis == 3:
                cbar_label = ('SCU')
            elif axis == 4:
                cbar_label = (r'$ {\tau / \sigma_{n}}^{\prime} $')
            elif axis == 5:
                cbar_label = ('Total normal stress (MPa)')
            elif axis == 6:
                cbar_label = ('Total shear stress (MPa)')

        # if delta:
        #     if axis == 0:
        #         cbar_label = ('Incremental shear stress in strike direction (MPa)')
        #     elif axis == 1:
        #         cbar_label = ('Incremental shear stress in dip direction (MPa)')
        #     elif axis == 2:
        #         cbar_label = ( 'Incremental effective normal stress (MPa)')
        #     elif axis == 3:
        #         cbar_label = ('SCU')
        #     elif axis == 4:
        #         cbar_label = (r'$ {\tau / \sigma_{n}}^{\prime} $')
        #     elif axis == 5:
        #         cbar_label = ('Total normal stress (MPa)')
        #     elif axis == 6:
        #         cbar_label = ('Total shear stress (MPa)')
        # else:
        #     if axis == 0:
        #         cbar_label = ('Shear stress in strike direction (MPa)')
        #     elif axis == 1:
        #         cbar_label = ('Shear stress in dip direction (MPa)')
        #     elif axis == 2:
        #         cbar_label =('Effective normal stress (MPa)')
        #     elif axis == 3:
        #         cbar_label = ('SCU')
        #     elif axis == 4:
        #         cbar_label = (r'$ {\tau / \sigma_{n}}^{\prime} $')
        #     elif axis == 5:
        #         cbar_label = ('Total normal stress (MPa)')
        #     elif axis == 6:
        #         cbar_label = ('Total shear stress (MPa)')

        return triang, z1, cbar_label

    def diff_data_triint(self,filename,step_,step_cri,xyaxis,delta,filename_tri0,filename_tri1,
                        xlim0=False,xlim1=False,zlim0=False,zlim1=False,
                        points=[],hlines=[]):
        """
        calculate the difference between the results
        """
        d2_mdict = self.load_dict_from_hdf5(filename)
        d2_fcoord = d2_mdict['crd_flt']
        d2_dat_trac_sta = d2_mdict['trac_sta']
        d2_dat_slip_sta = d2_mdict['slip_sta']
        del d2_mdict
        fig = plt.figure(figsize=(40,7),tight_layout=True)
        gs = gridspec.GridSpec(1, 5)

        print (np.abs(self.dat_trac_sta[:,3,step_[0]-1]-self.dat_trac_sta[:,3,1]).max())
        print (np.abs(d2_dat_trac_sta[:,3,step_[1]-1]-d2_dat_trac_sta[:,3,1]).max())
        ax_list={}
        plot_axis = [0,1,2,3,4]

        for i in range(5):
            plot_axi = plot_axis[i]
            ax_list[i]=fig.add_subplot(gs[i])
            # step_cri = [20, 51] # 51 for 45, 20 for 30 23 for 60
            print ("plotting figure " + str(i))
            # ax1 = fig.add_subplot(gs[i])

            # interprete for different data set
            for j in range(2):
                step = step_[j]

                if j == 0:
                    z = self.fcoord[:,2][:]*1e0
                    x = self.fcoord[:,0][:]*1e0
                    y = self.fcoord[:,1][:]*1e0
                    flt_tri = np.load(filename_tri0)

                    if plot_axi < 2:
                        axis = plot_axi
                        z1 = (self.dat_trac_sta[:,axis,step-1])/1e6
                        if delta:
                            z1 = z1 - (self.dat_trac_sta[:,axis,1])/1e6
                    elif plot_axi == 3:
                        # incremental Coulomb stress
                        ss = np.sqrt((self.dat_trac_sta[:,0,step-1]-self.dat_trac_sta[:,0,1])**2
                             +(self.dat_trac_sta[:,1,step-1]-self.dat_trac_sta[:,1,1])**2)
                        sn = -1*(self.dat_trac_sta[:,2,step-1]-self.dat_trac_sta[:,2,1])
                        z1 = (ss - sn*0.6)/1e6
                    elif plot_axi == 2:
                        axis = plot_axi
                        # incremental effective normal stress
                        z1 = (self.dat_trac_sta[:,axis,step-1])/1e6
                        if delta:
                            z1 = z1 - (self.dat_trac_sta[:,axis,1])/1e6
                    # SCU 
                    elif plot_axi == 4:
                        step = step_cri[j]
                        dc = 0.005
                        mu_node = 0.6*np.ones(len(self.dat_slip_sta[:,1,step-1])) # friction coefficient 
                        slip_node = np.sqrt(self.dat_slip_sta[:,0,step-1]**2+self.dat_slip_sta[:,1,step-1]**2) # total aseismic slip            
                        for k in range(len(mu_node)):
                            if slip_node[k]>dc:
                                mu_node[k] = 0.45
                                print ('completed weakening')
                            else:
                                mu_node[k] = 0.6 - 0.15*(slip_node[k]/dc)

                        z1 = np.sqrt(self.dat_trac_sta[:,0,step-1]**2+self.dat_trac_sta[:,1,step-1]**2)/self.dat_trac_sta[:,2,step-1]/mu_node
                        z1 = np.abs(z1)

                if j == 1:
                    z = d2_fcoord[:,2][:]*1e0
                    x = d2_fcoord[:,0][:]*1e0
                    y = d2_fcoord[:,1][:]*1e0
                    flt_tri = np.load(filename_tri1)

                    if plot_axi < 2:
                        axis = plot_axi
                        z1 = (d2_dat_trac_sta[:,axis,step-1])/1e6
                        if delta:
                            z1 = z1 - (d2_dat_trac_sta[:,axis,1])/1e6

                    elif plot_axi == 3:
                        # incremental Coulomb stress
                        ss = np.sqrt((d2_dat_trac_sta[:,0,step-1]-d2_dat_trac_sta[:,0,1])**2
                             +(d2_dat_trac_sta[:,1,step-1]-d2_dat_trac_sta[:,1,1])**2)
                        sn = -1*(d2_dat_trac_sta[:,2,step-1]-d2_dat_trac_sta[:,2,1])
                        z1 = (ss - sn*0.6)/1e6
                    elif plot_axi == 2:
                        axis = plot_axi                        # incremental effective normal stress
                        z1 = (d2_dat_trac_sta[:,axis,step-1])/1e6
                        if delta:
                            z1 = z1 - (d2_dat_trac_sta[:,axis,1])/1e6
                    # SCU 
                    elif plot_axi == 4:
                        step = step_cri[j]
                        dc = 0.005
                        mu_node = 0.6*np.ones(len(d2_dat_slip_sta[:,1,step-1])) # friction coefficient 
                        slip_node = np.sqrt(d2_dat_slip_sta[:,0,step-1]**2+d2_dat_slip_sta[:,1,step-1]**2) # total aseismic slip            
                        for k in range(len(mu_node)):
                            if slip_node[k]>dc:
                                mu_node[k] = 0.45
                                print ('completed weakening')
                            else:
                                mu_node[k] = 0.6 - 0.15*(slip_node[k]/dc)

                        z1 = np.sqrt(d2_dat_trac_sta[:,0,step-1]**2+d2_dat_trac_sta[:,1,step-1]**2)/d2_dat_trac_sta[:,2,step-1]/mu_node
                        z1 = np.abs(z1)

                reso = 1000
                intx_lim0 = xlim0
                intx_lim1 = xlim1
                intz_lim0 = zlim0
                intz_lim1 = zlim1
                if xyaxis == 0:
                    # xi = np.linspace(x.min(),x.max(), 200)
                    # yi = np.linspace(z.min(),z.max(), 200)
                    # xi, yi = np.meshgrid(np.linspace(x.min(),x.max(), reso), np.linspace(z.min(),z.max(), reso))
                    xi, yi = np.meshgrid(np.linspace(intx_lim0,intx_lim1, reso), np.linspace(intz_lim0,intz_lim1, reso))
                else:
                    # xi = np.linspace(y.min(),y.max(), 200)
                    # yi = np.linspace(z.min(),z.max(), 200)
                    # xi, yi = np.meshgrid(np.linspace(y.min(),y.max(), reso), np.linspace(z.min(),z.max(), reso))
                    xi, yi = np.meshgrid(np.linspace(intx_lim0,intx_lim1, reso), np.linspace(intz_lim0,intz_lim1, reso))

                flt_tri = flt_tri - 1
                ft_pos_nodes = np.unique(flt_tri)
                a = np.arange(ft_pos_nodes.max(),dtype=np.int32)
                b = np.zeros(ft_pos_nodes.max(),dtype=np.int32)
                c = np.vstack((a,b)).T
                d = np.arange(len(ft_pos_nodes),dtype=np.int32)
                c[ft_pos_nodes-1,1] = d

                if plot_axi == 4: z1[z1>1] = 1
                if xyaxis == 0:
                    triang = tri.Triangulation(x, z,c[flt_tri-1][:,:,1])
                else:
                    triang = tri.Triangulation(y, z,c[flt_tri-1][:,:,1])
                    if plot_axi==4 and j==0:
                        triang00 = tri.Triangulation(y, z,c[flt_tri-1][:,:,1])
                        z100 = np.copy(z1)

                # triangular interpretation
                if plot_axi == 4: z1[z1<0.99999] = 0
                interp_lin = tri.LinearTriInterpolator(triang, z1)
                # interp_lin = tri.CubicTriInterpolator(triang, z1,kind='geom')
                if j==0:
                    zi_lin0 = interp_lin(xi, yi)
                if j==1:
                    zi_lin1 = interp_lin(xi, yi)

            zi_lin = (zi_lin0 - zi_lin1)
            # zi_lin_abs = np.abs(zi_lin0 - zi_lin1)
            # zi_lin[zi_lin_abs<1]=0
            # if save:
            #     np.save("inte30-60",zi_lin=zi_lin)
            # print (zi_lin0)
            # print (zi_lin0.shape)
            # pt0=ax1.contourf(xi, yi, zi_lin)

            if len(points)>0:
                colorhl = {'0' : 'c--',
                           '1' : 'g--',}
                for pi in range(points.shape[0]//2):
                    ax_list[i].plot(points[pi:pi+2,0], points[pi:pi+2,1], colorhl[str(pi)],alpha=0.95,linewidth=2)

            if xyaxis == 0:
                # pt0 = ax1.imshow(zi_lin,extent=(x.min(),x.max(),z.max(),z.min()),aspect='auto')  
                pt0 = ax_list[i].imshow(zi_lin,extent=(intx_lim0,intx_lim1,intz_lim1,intz_lim0),aspect='auto')  

            if xyaxis == 1:
                # pt0 = ax1.imshow(zi_lin,extent=(y.min(),y.max(),z.max(),z.min()),aspect='auto') 
                ori_vmax = np.abs(zi_lin).max()
                if plot_axi < 4:
                    zi_lin = gaussian_filter(zi_lin, sigma=10) # 10 15
                # else:
                #     zi_lin = gaussian_filter(zi_lin, sigma=1)
                vmax = np.abs(zi_lin).max()
                zi_lin = zi_lin/vmax
                zi_lin = zi_lin*ori_vmax
                vmax = np.abs(zi_lin).max()
                pt0 = ax_list[i].imshow(zi_lin,extent=(intx_lim0,intx_lim1,intz_lim1,intz_lim0),aspect='auto',vmin = -1*vmax, vmax = vmax)  
                # n_levels = 7
                # pt0 = ax1.contourf(zi_lin, n_levels,
                #                    extent=(intx_lim0,intx_lim1,intz_lim0,intz_lim1))
                if plot_axi == 4:
                    pt0 = ax_list[i].tripcolor(triang00, z100, shading='flat',cmap='Reds' ) 
                    n_levels_cf = np.array([-1,-0.99999,0.99999,1])
                    tcf=ax_list[i].contourf(zi_lin, n_levels_cf,
                                       colors="none",
                                       extent=(intx_lim0,intx_lim1,intz_lim0,intz_lim1),
                                       hatches=["\\\\", None,"///", "\\", None, "\\\\", "*"])

                    # ------------------------------
                    # New bit here that handles changing the color of hatches
                    colors = ['black','white', 'white']
                    # For each level, we set the color of its hatch 
                    for kk, collection in enumerate(tcf.collections):
                        collection.set_edgecolor(colors[kk % len(colors)])
                    # Doing this also colors in the box around each level
                    # We can remove the colored line around the levels by setting the linewidth to 0
                    for collection in tcf.collections:
                        collection.set_linewidth(0.)
                    # ------------------------------

                    n_levels = np.array([-1,-0.99])
                    ax_list[i].contour(zi_lin, n_levels, extent=(intx_lim0,intx_lim1,intz_lim0,intz_lim1), colors="black",)
                    for axi in range(4):
                        ax_list[axi].contour(zi_lin, n_levels, extent=(intx_lim0,intx_lim1,intz_lim0,intz_lim1), colors="black",)
                        n_levels_cf = np.array([-1,-0.99999])
                        # n_levels = 4
                        tcf=ax_list[axi].contourf(zi_lin, n_levels_cf,
                                           colors="none",
                                           extent=(intx_lim0,intx_lim1,intz_lim0,intz_lim1),
                                           hatches=["\\\\", None,"///", "\\", None, "\\\\", "*"])
                        colors = ['black','white', 'white']
                        # For each level, we set the color of its hatch 
                        for kk, collection in enumerate(tcf.collections):
                            collection.set_edgecolor(colors[kk % len(colors)])
                        # Doing this also colors in the box around each level
                        # We can remove the colored line around the levels by setting the linewidth to 0
                        for collection in tcf.collections:
                            collection.set_linewidth(0.)
                    n_levels = np.array([0.99,1])
                    ax_list[i].contour(zi_lin, n_levels, extent=(intx_lim0,intx_lim1,intz_lim0,intz_lim1), colors="white",)
                    for axi in range(4):
                        ax_list[axi].contour(zi_lin, n_levels, extent=(intx_lim0,intx_lim1,intz_lim0,intz_lim1), colors="white",)
                        n_levels_cf = np.array([0.99999,1])
                        # n_levels = 4
                        tcf=ax_list[axi].contourf(zi_lin, n_levels_cf,
                                           colors="none",
                                           extent=(intx_lim0,intx_lim1,intz_lim0,intz_lim1),
                                           hatches=["///", "\\", None, "\\\\", "*"])
                        colors = ['white','white', 'white']
                        # For each level, we set the color of its hatch 
                        for kk, collection in enumerate(tcf.collections):
                            collection.set_edgecolor(colors[kk % len(colors)])
                        # Doing this also colors in the box around each level
                        # We can remove the colored line around the levels by setting the linewidth to 0
                        for collection in tcf.collections:
                            collection.set_linewidth(0.)

                    # artists, labels = tcf.legend_elements(str_format="{:2.1f}".format)
                    # ax1.legend(artists, labels, handleheight=2, framealpha=1)
            # interpretation mesh 
            # pt0=ax1.plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
            # pt0=ax1.plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)

            # contour plot with SCU=1 
            # levels = np.array([0.99,1.0]) # levels = np.array([0,0.2,0.3,0.9,1.0,1.1])
            # ax1.contour([xi,yi],zi_lin,levels=levels, colors='k')

            if len(hlines)>0:
                colorhl = {'0' : 'c',
                           '1' : 'c',
                           '2' : 'g',
                           '3' : 'g',
                           '4' : 'y',
                           '5' : 'y'}
                hlid = 0
                for hline in (hlines):
                    ax_list[i].axhline(y=hline,color=colorhl[str(hlid)],alpha=0.95)

                    hlid += 1
            cbar = fig.colorbar(pt0,format='%.1f')
            if plot_axi == 0:
                cbar.set_label('Shear stress (Strike) (MPa)')
            elif plot_axi == 1:
                cbar.set_label('Shear stress (dip) (MPa)')
            elif plot_axi == 2:
                cbar.set_label('Effective normal stress (MPa)')
                # cbar.set_label('Coulomb stress (MPa)')
                # if delta>0: cbar.set_label('Effective normal stress (NO-dP) (MPa)')
                # if friction>0: cbar.set_label('Columb stress change (MPa)')
            elif plot_axi == 3:
                cbar.set_label('Coulomb stress (MPa)')
            elif plot_axi == 4:
                cbar.set_label('SCU')

            # elif plot_axi == 4:
            #     cbar.set_label(r'$ \tau / \sigma_{n} $')
            # elif plot_axi == 5:
            #     cbar.set_label(r'$ Stress path ratio$')

            if zlim0 or zlim1:
                ax_list[i].set_ylim(zlim0,zlim1)
            if xlim0 or xlim1:
                ax_list[i].set_xlim(xlim0,xlim1)
            ax_list[i].set_xlabel('Strike (km)')
            if i == 0: ax_list[i].set_ylabel('Depth (km)')
            if i > 0:
                ax_list[i].tick_params(
                    axis='y',          # changes apply to the x-axis
                    labelleft=False) # labels along the bottom edge are off

        fig.tight_layout()
        plt.show()

