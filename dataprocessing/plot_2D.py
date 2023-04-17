import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
from scipy.signal import butter, lfilter, freqz
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.tri as tri
from matplotlib.colors import Normalize
import scipy.interpolate
import scipy.io

class plot2D:
    def __init__(self, file, dyn, seis):
        self.file = file
        self.dyn = int(dyn)
        self.seis = int(seis)
        
        self.mdict = self.load_dict_from_hdf5(self.file)
    
        self.fcoord = self.mdict['crd_flt']
        self.fsort=np.argsort(self.fcoord[:,1][:]) # necessary for 2D data
        
        self.dat_trac_sta = self.mdict['trac_sta']
        self.dat_slip_sta = self.mdict['slip_sta']


        if self.dyn>0:
            self.dt = self.mdict['dt']
            self.dt_dyn = self.mdict['dt_dyn']
            self.dat_log = self.mdict['log']
            self.dat_log_dyn = self.mdict['log_dyn']
            self.dat_trac_sort = self.mdict['trac_dyn']
            self.dat_slip_sort = self.mdict['slip_dyn']
            
        if self.seis>0:
            self.crd_obs = self.mdict['crd_obs']
            self.dat_seis_sort = self.mdict['obs_dyn']

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



    def plot_static_group(self,benchmark=False,delta=False,
                         mask=0,zlim0=False,zlim1=False,
                         xlim0=False,xlim1=False,punique=False,hline0=False,hline1=False,hline2=False,):
        # 50m offset

        self.file = "./Buijze_benchmark_06m_fe.h5"
        self.dyn = 1
        
        self.mdict = self.load_dict_from_hdf5(self.file)
    
        self.fcoord = self.mdict['crd_flt']
        self.fsort=np.argsort(self.fcoord[:,1][:]) # necessary for 2D data
        
        self.dat_trac_sta = self.mdict['trac_sta']
        self.dat_slip_sta = self.mdict['slip_sta']


        if self.dyn>0:
            self.dt = self.mdict['dt']
            self.dt_dyn = self.mdict['dt_dyn']
            self.dat_log = self.mdict['log']
            self.dat_log_dyn = self.mdict['log_dyn']
            self.dat_trac_sort = self.mdict['trac_dyn']
            self.dat_slip_sort = self.mdict['slip_dyn']
        x = self.fcoord[:,0][:]*1e0
        y = self.fcoord[:,1][:]*1e0

        # 0m offset label
        mat_dict = { 4 : 0,
                     11 : 4,
                     18 : 8,
                     25 : 12,
                     30 : 15}
        mat_labl_defmod = { 4 : 0,
                     11 : 7,
                     18 : 14,
                     25 : 21,
                     30 : 26}
        mat_labl = { 4 : 0,
                     11 : 7,
                     18 : 14,
                     25 : 21,
                     30 : 26.25}


        fig = plt.figure(figsize=(22,16),tight_layout=True)
        gs = gridspec.GridSpec(2, 5)
        if benchmark: 
            mat = scipy.io.loadmat('./benchmark/basecase_0m/compaction_fault.mat')

            mat_slip = scipy.io.loadmat('./benchmark/basecase_0m/dva_fault.mat')
            mat_stress = scipy.io.loadmat('./benchmark/basecase_0m/stress_fault.mat')
            mat_trac_shear = -1*mat_stress['out'][0][0][5][:,:]/1e6
            mat_slip_shear = np.abs(mat_slip['out'][0][0][4][:,:])

            mat_z = mat['out'][0][0][17][:,0]/1e3
            mat_trac_normal = mat['out'][0][0][12]/1e6
            mat_trac_shear = -1*mat['out'][0][0][11]/1e6
            mat_pressure = mat['out'][0][0][13]/1e6
            mat_asei = np.abs(mat['out'][0][0][6]) # 4 5 6
            # print (mat_asei.max(),mat_asei.min())
            mat_SCU = -1*mat_trac_shear/(mat_trac_normal*(0.6-mat['out'][0][0][4]*10/0.05*(0.15)))
            mat_z_slip = mat_slip['out'][0][0][10][:,:][:,0]/1e3
            mat_z_stress = mat_stress['out'][0][0][13][:,:][:,0]/1e3
            mat_slip_shear = np.abs(mat_slip['out'][0][0][4][:,:])


        plot_axis = [0]
        kk = 0
        for i in range(len(plot_axis)):
            # dynamic 0m
            ax1 = fig.add_subplot(gs[0, 4])
            t0 = 0
            t1 = 50
            dt = 10
            step = 30
            tstep_list = (np.arange(t0,t1,dt,dtype=np.int32))
            for j in (tstep_list):
                xp,yp,xlabel,ylabel = self.plot_dyn_group_func(step=step,tstep_list=[j],axis=plot_axis[i]) # axis = slip and shear trac        

                if benchmark:
                    if plot_axis[i] == 4:
                        if (kk==0 or kk==len(tstep_list)-1):
                            pcheck, = ax1.plot(xp,yp,'-',label=(str(j*0.01)+" s"))
                        kk += 1
                    else:
                        pcheck, = ax1.plot(xp,yp,'-',label=(str(j*0.01)+" s"))
                    cbm = pcheck.get_color()
                    if plot_axis[i] == 0: 
                        if j == tstep_list[-1]:
                            ax1.plot(mat_slip_shear[:,j*5], mat_z_slip,'--',c=cbm,label=("Buijze et al.(2019)"))
                        else:
                            ax1.plot(mat_slip_shear[:,j*5], mat_z_slip,'--',c=cbm)
                    if plot_axis[i] == 4:
                        if kk==1: 
                            # print (mat_trac_shear[:,0].shape)
                            ax1.plot(mat_trac_shear[:,0],mat_z_stress,'--',c=cbm,label=("- " + str([j])+"step (DIANA)"))
                        if kk==len(tstep_list): 
                            # print (mat_trac_shear[:,-1].shape)
                            ax1.plot(mat_trac_shear[:,-1],mat_z_stress,'--',c=cbm,label=("- " + str([j])+"step (DIANA)"))
                    # if i == 3: ax1.plot(mat_trac_normal[:,mat_dict[j]],mat_z,'-',label=("Step0" + str(j)))

                else:
                    pcheck, = ax1.plot(xp,yp,'--',label=("step " + str(j)))

            ax1.set_xlabel(xlabel)
            # ax1.set_ylabel(ylabel)
            ax1.tick_params(
                axis='y',          # changes apply to the x-axis
                labelleft=False) # labels along the bottom edge are off
            if hline0:  ax1.axhline(y=hline0, color='r', linestyle='--')
            if hline1:  ax1.axhline(y=hline1, color='g', linestyle='--')
            if hline2:  ax1.axhline(y=hline2, color='b', linestyle='--')
            if zlim0:
                ax1.set_ylim(zlim0,zlim1)
            if xlim1 and i ==0:
                ax1.set_xlim(xlim0,xlim1)
            if plot_axis[i] == 0: ax1.legend(loc='lower left',prop={'size': 14}) # prop={'size': 12}
            if plot_axis[i] == 4: ax1.legend(loc='lower left',prop={'size': 12},ncol=2)
            ax1.axhspan(-2.8, -3.0, facecolor='gray', alpha=0.2)
            plt.grid()

        plot_axis = [0,1,3,4]
        for i in range(4):
            ax1 = fig.add_subplot(gs[0, i])
            tstep_list = [4,11,18,25,30]

            for j in (tstep_list):
                xp,yp,xlabel,ylabel = self.plot_static_group_func(tstep_list=[j],axis=plot_axis[i])
                if benchmark:
                    pcheck, = ax1.plot(xp,yp,'-',label=("- " + str(mat_labl_defmod[j])+" MPa"))
                    cbm = pcheck.get_color()
                    if plot_axis[i] == 0: ax1.plot(mat_trac_shear[:,mat_dict[j]],mat_z,'--',c=cbm,label=("- " + str(mat_labl[j])+" MPa (DIANA)"))
                    if plot_axis[i] == 1: ax1.plot(mat_trac_normal[:,mat_dict[j]],mat_z,'--',c=cbm,label=("- " + str(mat_labl[j])+" MPa (DIANA)"))
                    if plot_axis[i] == 2: ax1.plot(mat_pressure[:,mat_dict[j]],mat_z,'--',c=cbm,label=("- " + str(mat_labl[j])+" MPa (DIANA)"))
                    if plot_axis[i] == 3: ax1.plot(mat_SCU[:,mat_dict[j]],mat_z,'--',c=cbm,label=("- " + str(mat_labl[j])+" MPa (DIANA)"))
                    if plot_axis[i] == 4 :
                        if j == tstep_list[-1]:
                            ax1.plot(mat_asei[:,mat_dict[j]],mat_z,'--',c=cbm,label=("Buijze et al. (2019)"))
                        else:
                            ax1.plot(mat_asei[:,mat_dict[j]],mat_z,'--',c=cbm)
                    # if i == 3: ax1.plot(mat_trac_normal[:,mat_dict[j]],mat_z,'-',label=("Step0" + str(j)))
                else:
                    pcheck, = ax1.plot(xp,yp,'--',label=("step " + str(j)))

            ax1.set_xlabel(xlabel)
            if i ==0: 
                ax1.set_ylabel(ylabel)
            else:
                ax1.tick_params(
                    axis='y',          # changes apply to the x-axis
                labelleft=False) # labels along the bottom edge are off
            if hline0:  ax1.axhline(y=hline0, color='r', linestyle='--')
            if hline1:  ax1.axhline(y=hline1, color='g', linestyle='--')
            if hline2:  ax1.axhline(y=hline2, color='b', linestyle='--')
            if zlim0:
                ax1.set_ylim(zlim0,zlim1)
            if xlim1:
                ax1.set_xlim(xlim0,xlim1)
            if plot_axis[i] == 4: ax1.legend(loc='lower left',prop={'size': 14})
            # titleax = { 0 : 'a',
            #              1 : 'b',
            #              2 : 'c',
            #              3 : 'd',}
            # ax1.set_title(titleax[i], y=0.98,x=0.08, pad=-14)
            ax1.axhspan(-2.8, -3.0, facecolor='gray', alpha=0.2)
            plt.grid()



        # 50m offset

        self.file = "./Buijze_benchmark_50m_fe.h5"
        self.dyn = 1
        
        self.mdict = self.load_dict_from_hdf5(self.file)
    
        self.fcoord = self.mdict['crd_flt']
        self.fsort=np.argsort(self.fcoord[:,1][:]) # necessary for 2D data
        
        self.dat_trac_sta = self.mdict['trac_sta']
        self.dat_slip_sta = self.mdict['slip_sta']


        if self.dyn>0:
            self.dt = self.mdict['dt']
            self.dt_dyn = self.mdict['dt_dyn']
            self.dat_log = self.mdict['log']
            self.dat_log_dyn = self.mdict['log_dyn']
            self.dat_trac_sort = self.mdict['trac_dyn']
            self.dat_slip_sort = self.mdict['slip_dyn']
        x = self.fcoord[:,0][:]*1e0
        y = self.fcoord[:,1][:]*1e0
        # 50m offset label
        mat_dict = { 4 : 0,
                     11 : 4,
                     15: 7}
        mat_labl_defmod = { 4 : 0,
                     11 : 7,
                     15: 11}
        mat_labl = { 4 : 0,
                     11 : 7,
                     15: 10.73}
        if benchmark: 
            # mat = scipy.io.loadmat('./benchmark/compaction_fault_0m_noslip.mat')
            mat = scipy.io.loadmat('./benchmark/basecase_50m/compaction_fault.mat')
            mat_z = mat['out'][0][0][17][:,0]/1e3
            mat_trac_normal = mat['out'][0][0][12]/1e6
            mat_trac_shear = -1*mat['out'][0][0][11]/1e6
            mat_pressure = mat['out'][0][0][13]/1e6
            mat_asei = np.abs(mat['out'][0][0][6]) # 4 5 6
            # print (mat_asei.max(),mat_asei.min())
            mat_SCU = -1*mat_trac_shear/(mat_trac_normal*(0.6-mat['out'][0][0][4]*10/0.05*(0.15)))

            mat_slip = scipy.io.loadmat('./benchmark/basecase_50m/dva_fault.mat')
            mat_stress = scipy.io.loadmat('./benchmark/basecase_50m/stress_fault.mat')
            # mat_trac_shear = -1*mat_stress['out'][0][0][5][:,:]/1e6
            mat_slip_shear = np.abs(mat_slip['out'][0][0][4][:,:])

            mat_z_slip = mat_slip['out'][0][0][10][:,:][:,0]/1e3

        plot_axis = [0]
        kk = 0
        # dynamic 50m
        for i in range(len(plot_axis)):
            ax1 = fig.add_subplot(gs[1, 4])
            t0 = 0
            t1 = 50
            dt = 10
            step = 15
            tstep_list = (np.arange(t0,t1,dt,dtype=np.int32))
            for j in (tstep_list):
                xp,yp,xlabel,ylabel = self.plot_dyn_group_func(step=step,tstep_list=[j],axis=plot_axis[i]) # axis = slip and shear trac        

                if benchmark:
                    if plot_axis[i] == 4:
                        if (kk==0 or kk==len(tstep_list)-1):
                            pcheck, = ax1.plot(xp,yp,'-',label=("- " + str(j)+"step"))
                        kk += 1
                    else:
                        pcheck, = ax1.plot(xp,yp,'-',label=(str(j*0.01)+" s"))
                    cbm = pcheck.get_color()
                    if plot_axis[i] == 0: 
                        if j == tstep_list[-1]:
                            ax1.plot(mat_slip_shear[:,j*5], mat_z_slip,'--',c=cbm,label=("Buijze et al. (2019)"))
                        else:
                            ax1.plot(mat_slip_shear[:,j*5], mat_z_slip,'--',c=cbm)
                    if plot_axis[i] == 4:
                        if kk==1: 
                            # print (mat_trac_shear[:,0].shape)
                            ax1.plot(mat_trac_shear[:,0],mat_z_stress,'--',c=cbm,label=("- " + str([j])+"step (DIANA)"))
                        if kk==len(tstep_list): 
                            # print (mat_trac_shear[:,-1].shape)
                            ax1.plot(mat_trac_shear[:,-1],mat_z_stress,'--',c=cbm,label=("- " + str([j])+"step (DIANA)"))
                    # if i == 3: ax1.plot(mat_trac_normal[:,mat_dict[j]],mat_z,'-',label=("Step0" + str(j)))

                else:
                    pcheck, = ax1.plot(xp,yp,'--',label=("step " + str(j)))

            ax1.set_xlabel(xlabel)
            ax1.tick_params(
                axis='y',          # changes apply to the x-axis
            labelleft=False) # labels along the bottom edge are off
            # ax1.set_ylabel(ylabel)
            if hline0:  ax1.axhline(y=hline0, color='r', linestyle='--')
            if hline1:  ax1.axhline(y=hline1, color='g', linestyle='--')
            if hline2:  ax1.axhline(y=hline2, color='b', linestyle='--')
            if zlim0:
                ax1.set_ylim(zlim0,zlim1)
            if xlim1 and i ==0:
                ax1.set_xlim(xlim0,xlim1)
            if plot_axis[i] == 0: ax1.legend(loc='lower left',prop={'size': 14})
            if plot_axis[i] == 4: ax1.legend(loc='lower left',prop={'size': 12},ncol=2)
            # ax1.axhspan(-2.8, -3.0, facecolor='gray', alpha=0.2)
            x_lim_ax = ax1.get_xlim()
            x_lim_min = x_lim_ax[0]
            x_lim_max = x_lim_ax[1]
            x_lim_span = np.abs(x_lim_max - x_lim_min)/2
            polygon1 = Polygon([(x_lim_min,-2.85), (x_lim_min+x_lim_span,-2.85), (x_lim_min+x_lim_span*0.6,-3.05),(x_lim_min,-3.05)])
            patch = ax1.add_patch(polygon1)
            patch.set_color('gray')
            patch.set_alpha(0.2)
            polygon1 = Polygon([(x_lim_max,-2.8), (x_lim_max-x_lim_span*0.6,-2.8), (x_lim_max-x_lim_span,-3.0),(x_lim_max,-3.0)])
            patch = ax1.add_patch(polygon1)
            patch.set_color('gray')
            patch.set_alpha(0.2)
            plt.grid()

        plot_axis = [0,1,3,4]
        for i in range(4):
            ax1 = fig.add_subplot(gs[1, i])
            tstep_list = [4,11,15]

            for j in (tstep_list):
                xp,yp,xlabel,ylabel = self.plot_static_group_func(tstep_list=[j],axis=plot_axis[i])
                if benchmark:
                    pcheck, = ax1.plot(xp,yp,'-',label=("- " + str(mat_labl_defmod[j])+" MPa"))
                    cbm = pcheck.get_color()
                    if plot_axis[i] == 0: ax1.plot(mat_trac_shear[:,mat_dict[j]],mat_z,'--',c=cbm,label=("- " + str(mat_labl[j])+" MPa (DIANA)"))
                    if plot_axis[i] == 1: ax1.plot(mat_trac_normal[:,mat_dict[j]],mat_z,'--',c=cbm,label=("- " + str(mat_labl[j])+" MPa (DIANA)"))
                    if plot_axis[i] == 2: ax1.plot(mat_pressure[:,mat_dict[j]],mat_z,'--',c=cbm,label=("- " + str(mat_labl[j])+" MPa (DIANA)"))
                    if plot_axis[i] == 3: ax1.plot(mat_SCU[:,mat_dict[j]],mat_z,'--',c=cbm,label=("- " + str(mat_labl[j])+" MPa (DIANA)"))
                    if plot_axis[i] == 4 :
                        if j == tstep_list[-1]:
                            ax1.plot(mat_asei[:,mat_dict[j]],mat_z,'--',c=cbm,label=("Buijze et al. (2019)"))
                        else:
                            ax1.plot(mat_asei[:,mat_dict[j]],mat_z,'--',c=cbm)
                    # if i == 3: ax1.plot(mat_trac_normal[:,mat_dict[j]],mat_z,'-',label=("Step0" + str(j)))
                else:
                    pcheck, = ax1.plot(xp,yp,'--',label=("step " + str(j)))

            ax1.set_xlabel(xlabel)
            if i ==0: 
                ax1.set_ylabel(ylabel)
            else:
                ax1.tick_params(
                    axis='y',          # changes apply to the x-axis
                labelleft=False) # labels along the bottom edge are off
            if hline0:  ax1.axhline(y=hline0, color='r', linestyle='--')
            if hline1:  ax1.axhline(y=hline1, color='g', linestyle='--')
            if hline2:  ax1.axhline(y=hline2, color='b', linestyle='--')
            if zlim0:
                ax1.set_ylim(zlim0,zlim1)
            if xlim1:
                ax1.set_xlim(xlim0,xlim1)
            if plot_axis[i] == 4: ax1.legend(loc='lower left',prop={'size': 14})
            # titleax = { 0 : 'a',
            #              1 : 'b',
            #              2 : 'c',
            #              3 : 'd',}
            # ax1.set_title(titleax[i], y=0.98,x=0.08, pad=-14)
            # ax1.axhspan(-2.8, -3.0, facecolor='gray', alpha=0.2)
            x_lim_ax = ax1.get_xlim()
            x_lim_min = x_lim_ax[0]
            x_lim_max = x_lim_ax[1]
            x_lim_span = np.abs(x_lim_max - x_lim_min)/2
            polygon1 = Polygon([(x_lim_min,-2.85), (x_lim_min+x_lim_span,-2.85), (x_lim_min+x_lim_span*0.6,-3.05),(x_lim_min,-3.05)])
            patch = ax1.add_patch(polygon1)
            patch.set_color('gray')
            patch.set_alpha(0.2)
            polygon1 = Polygon([(x_lim_max,-2.8), (x_lim_max-x_lim_span*0.6,-2.8), (x_lim_max-x_lim_span,-3.0),(x_lim_max,-3.0)])
            patch = ax1.add_patch(polygon1)
            patch.set_color('gray')
            patch.set_alpha(0.2)
            ax1.axhline(y=-2.8, color='gray', linestyle='-',alpha=0.6)
            ax1.axhline(y=-2.85, color='gray', linestyle='-',alpha=0.6)
            ax1.axhline(y=-3.0, color='gray', linestyle='-',alpha=0.6)
            ax1.axhline(y=-3.05, color='gray', linestyle='-',alpha=0.6)
            plt.grid()
        # fig.align_labels() 
        
        plt.show()


    def plot_static_group_func(self,tstep_list,axis,delta=False,
                         mask=0,zlim0=False,zlim1=False,
                         xlim0=False,xlim1=False,punique=False,hline0=False,hline1=False,hline2=False,):
        x = self.fcoord[:,0][:]*1e0
        y = self.fcoord[:,1][:]*1e0
        for i in (tstep_list):
            if axis < 3:
                z1 = (self.dat_trac_sta[:,axis,i-1])/1e6
                if delta:
                    z1 = z1 - (self.dat_trac_sta[:,axis,1])/1e6
                    print (abs(z1).max())

            elif axis == 3:
                dc = 0.005
                mu_node = 0.6*np.ones(len(self.dat_slip_sta[:,1,i-1]))
                slip_node = np.abs(self.dat_slip_sta[:,0,i-1])
                # print (mu_node.shape)
                # print (self.dat_trac_sta[:,0,i-1].shape)
                for j in range(len(mu_node)):
                    if slip_node[j]>dc:
                        mu_node[j] = 0.45
                    else:
                        mu_node[j] = 0.6 - 0.15*(slip_node[j]/dc)

                z1 = (self.dat_trac_sta[:,0,i-1]/self.dat_trac_sta[:,1,i-1])/mu_node
                # print (slip_node.max(),slip_node.min())
                # print (mu_node.max(),mu_node.min())
                z1 = np.abs(z1)
                if mask:
                    mask_ = z1>=mask
                    z1[~mask_] = 0
            # aseismic 
            elif axis == 4:
                z1 = np.abs(self.dat_slip_sta[:,0,i-1]*100)

            elif axis == 5:
                if i > 5:
                    z1 = np.abs((self.dat_trac_sta[:,0,i-1]-self.dat_trac_sta[:,0,1])/ (self.dat_trac_sta[:,1,i-1]-self.dat_trac_sta[:,1,1]))
                    if mask:
                        mask_ = z1>=mask
                        z1[~mask_] = 0
                else: 
                    z1 = np.abs((self.dat_trac_sta[:,0,i-1]))
                    z1[:] = 0
                z1 = np.abs(z1)

            elif axis == 6:
                z1 = (self.dat_slip_sta[:,0,i-1])

            # if punique:
            #     print (np.unique(z1))       
            # ax1.plot(z1[self.fsort],y[self.fsort],'--',label=("Step" + str(i)))

        # ax1.legend(loc='lower left')

        ylabel = 'Depth (km)'
        # if hline0:  ax1.axhline(y=hline0, color='r', linestyle='--')
        # if hline1:  ax1.axhline(y=hline1, color='g', linestyle='--')
        # if hline2:  ax1.axhline(y=hline2, color='b', linestyle='--')
        # if zlim0:
        #     ax1.set_ylim(zlim0,zlim1)
        # if xlim1:
        #     ax1.set_xlim(xlim0,xlim1)
        if axis == 0:
            xlabel = 'Shear stress (MPa)'
        elif axis == 1:
            xlabel = 'Effective normal stress (MPa)'
        elif axis == 2:
            xlabel = 'Pore pressure (MPa)'
        elif axis == 3:
            xlabel = 'SCU'
        elif axis == 4:
            xlabel = 'Aseismic slip (mm)'
        # fig.tight_layout()
        # plt.grid()
        # plt.show()
        return z1[self.fsort],y[self.fsort],xlabel,ylabel,


    def plot_dyn_group(self,step,tstep_list,benchmark=False,delta=False,
                         mask=0,zlim0=False,zlim1=False,
                         xlim0=False,xlim1=False,punique=False,hline0=False,hline1=False,hline2=False,):

        x = self.fcoord[:,0][:]*1e0
        y = self.fcoord[:,1][:]*1e0

        # 50m offset dyn label
        # mat_dict = { 4 : 0,
        #              11 : 2,}
        # mat_labl = { 4 : 0,
        #              11 : 7,}
        plot_labl = { 4 : 0,
                     11 : 7,}

        fig = plt.figure(figsize=(10,8),tight_layout=True)
        gs = gridspec.GridSpec(1, 2)
        if benchmark: 
            # mat = scipy.io.loadmat('./benchmark/compaction_fault_0m_noslip.mat')
            # mat_slip = scipy.io.loadmat('./benchmark/basecase_50m/dva_fault.mat')
            # mat_stress = scipy.io.loadmat('./benchmark/basecase_50m/stress_fault.mat')
            mat_slip = scipy.io.loadmat('./benchmark/basecase_0m/dva_fault.mat')
            mat_stress = scipy.io.loadmat('./benchmark/basecase_0m/stress_fault.mat')
            mat_z_slip = mat_slip['out'][0][0][10][:,:][:,0]/1e3
            mat_z_stress = mat_stress['out'][0][0][13][:,:][:,0]/1e3
            mat_trac_normal = mat_stress['out'][0][0][6][:,:]/1e6
            mat_trac_shear = -1*mat_stress['out'][0][0][5][:,:]/1e6
            mat_slip_shear = np.abs(mat_slip['out'][0][0][4][:,:])
            # print (np.abs(mat_trac_shear).max())
            # print (np.abs(mat_slip_shear).max())
            # print (np.abs(mat_trac_shear))
            # print (np.abs(mat_slip_shear).shape)
            # print (np.abs(mat_slip_shear))

        plot_axis = [0,4]
        kk = 0
        for i in range(2):
            ax1 = fig.add_subplot(gs[0, i])
            for j in (tstep_list):
                xp,yp,xlabel,ylabel = self.plot_dyn_group_func(step=step,tstep_list=[j],axis=plot_axis[i]) # axis = slip and shear trac        

                if benchmark:
                    if plot_axis[i] == 4:
                        if (kk==0 or kk==len(tstep_list)-1):
                            pcheck, = ax1.plot(xp,yp,'--',label=("- " + str([j])+"step"))
                        kk += 1
                    else:
                        pcheck, = ax1.plot(xp,yp,'--',label=("- " + str([j])+"step"))
                    cbm = pcheck.get_color()
                    if plot_axis[i] == 0: ax1.plot(mat_slip_shear[:,j*5], mat_z_slip,'-',c=cbm,label=("- " + str([j])+"step (DIANA)"))
                    if plot_axis[i] == 4:
                        if kk==1: 
                            # print (mat_trac_shear[:,0].shape)
                            ax1.plot(mat_trac_shear[:,0],mat_z_stress,'-',c=cbm,label=("- " + str([j])+"step (DIANA)"))
                        if kk==len(tstep_list): 
                            # print (mat_trac_shear[:,-1].shape)
                            ax1.plot(mat_trac_shear[:,-1],mat_z_stress,'-',c=cbm,label=("- " + str([j])+"step (DIANA)"))
                    # if i == 3: ax1.plot(mat_trac_normal[:,mat_dict[j]],mat_z,'-',label=("Step0" + str(j)))

                else:
                    pcheck, = ax1.plot(xp,yp,'--',label=("step " + str(j)))

            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(ylabel)
            if hline0:  ax1.axhline(y=hline0, color='r', linestyle='--')
            if hline1:  ax1.axhline(y=hline1, color='g', linestyle='--')
            if hline2:  ax1.axhline(y=hline2, color='b', linestyle='--')
            if zlim0:
                ax1.set_ylim(zlim0,zlim1)
            if xlim1 and i ==0:
                ax1.set_xlim(xlim0,xlim1)
            if plot_axis[i] == 0: ax1.legend(loc='lower left',prop={'size': 10},ncol=2)
            if plot_axis[i] == 4: ax1.legend(loc='lower left',prop={'size': 10},ncol=2)
            titleax = { 0 : 'a',
                         1 : 'b',
                         2 : 'c',
                         3 : 'd',}
            ax1.set_title(titleax[i], y=0.98,x=0.08, pad=-14)
            ax1.axhspan(-2.8, -3.0, facecolor='gray', alpha=0.2)
            plt.grid()
        # fig.align_labels() 
        
        plt.show()

    def plot_dyn_group_func(self,step,tstep_list,axis,velocity=False,delta=False,
                         mask=0,zlim0=False,zlim1=False,
                         xlim0=False,xlim1=False,hline0=False,hline1=False,hline2=False,):
        # for i in (np.arange(0,t1-t0,plot_step)):    
        #     f2_ax1.plot(obs_data1[:,i][fsort],x[:-1][fsort],label=str(t_step[i])+'ms')

        x = self.fcoord[:,0][:]*1e0
        y = self.fcoord[:,1][:]*1e0

        for i in tstep_list: 
            slip_shear = (self.dat_slip_sort['step '+ str(step)][:,0,i])
            slip_normal = (self.dat_slip_sort['step '+ str(step)][:,1,i])
            trac_shear = (self.dat_trac_sort['step '+ str(step)][:,0,i])
            trac_normal = (self.dat_trac_sort['step '+ str(step)][:,1,i])
            trac_f2 = (self.dat_trac_sort['step '+ str(step)][:,2,i]) # friction
            trac_f3 = (self.dat_trac_sort['step '+ str(step)][:,3,i]) # counter normal force
            if velocity:
                slip_shear = np.sum(slip_shear*dt_dyn)
            if axis == 0:
                z1 = slip_shear
            elif axis == 1:
                z1 = slip_normal    
            elif axis == 2:
                z1 = trac_shear/1e6
            elif axis == 3:
                z1 = trac_normal/1e6
            elif axis == 4:
                z1 = trac_f2/1e6
            elif axis == 5:
                z1 = trac_f3/1e6
            elif axis == 6:
                z1 = trac_shear/trac_f2

        if axis == 0:
            xlabel='Shear slip (m)'
        elif axis == 1:
            xlabel='Normal dispalcement'
        elif axis == 2:
            xlabel='Shear stress'
        elif axis == 3:
            xlabel='Effective normal stress'
        elif axis == 4:
            xlabel='Shear stress (MPa)'
        elif axis == 5:
            xlabel='normal lamda'
        elif axis == 6:
            xlabel='SCU'
        ylabel = 'Depth (m)'
        return z1[self.fsort],y[self.fsort],xlabel,ylabel,




    def plot_static_list2D(self,tstep_list,axis,delta=False,
                         mask=0,zlim0=False,zlim1=False,
                         xlim0=False,xlim1=False,punique=False,hline0=False,hline1=False,hline2=False,):
        x = self.fcoord[:,0][:]*1e0
        y = self.fcoord[:,1][:]*1e0
        fig = plt.figure(figsize=(5,6))
        ax1 = fig.add_subplot(1, 1, 1)
        for i in (tstep_list):
            if axis < 3:
                z1 = (self.dat_trac_sta[:,axis,i-1])/1e6
                if delta:
                    z1 = z1 - (self.dat_trac_sta[:,axis,1])/1e6
                    print (abs(z1).max())

            elif axis == 3:
                z1 = (self.dat_trac_sta[:,0,i-1]/self.dat_trac_sta[:,1,i-1])
                z1 = np.abs(z1)
                if mask:
                    mask_ = z1>=mask
                    z1[~mask_] = 0
                    
            elif axis == 4:
                if i > 5:
                    z1 = np.abs((self.dat_trac_sta[:,0,i-1]-self.dat_trac_sta[:,0,1])/ (self.dat_trac_sta[:,1,i-1]-self.dat_trac_sta[:,1,1]))
                    if mask:
                        mask_ = z1>=mask
                        z1[~mask_] = 0
                else: 
                    z1 = np.abs((self.dat_trac_sta[:,0,i-1]))
                    z1[:] = 0
                z1 = np.abs(z1)

            elif axis == 5:
                z1 = (self.dat_slip_sta[:,0,i-1])

                    
#             elif axis == 5:
#                 z1 = (self.dat_trac_sta[:,2,i-1]-self.dat_trac_sta[:,3,i-1])/1e6
#                 if delta:
#                     z1 = z1 - (self.dat_trac_sta[:,2,2]+self.dat_trac_sta[:,3,2])/1e6
            if punique:
                print (np.unique(z1))       
            # ax1.scatter(z1,y,label=("Step" + str(i)))
            ax1.plot(z1[self.fsort],y[self.fsort],'--',label=("Step" + str(i)))

        ax1.legend(loc='lower left')
        ax1.set_ylabel('Z (km)')
        if hline0:  ax1.axhline(y=hline0, color='r', linestyle='--')
        if hline1:  ax1.axhline(y=hline1, color='g', linestyle='--')
        if hline2:  ax1.axhline(y=hline2, color='b', linestyle='--')
        if zlim0:
            ax1.set_ylim(zlim0,zlim1)
        if xlim1:
            ax1.set_xlim(xlim0,xlim1)
        if axis == 0:
            ax1.set_xlabel('Shear stress (MPa)')
        elif axis == 1:
            ax1.set_xlabel('Effective normal stress (MPa)')
        elif axis == 2:
            ax1.set_xlabel('Pressure (MPa)')
        elif axis == 3:
            ax1.set_xlabel(r'$ \tau / \sigma_{n} $')
        elif axis == 4:
            ax1.set_xlabel(r'$ Stress path $')
        fig.tight_layout()
        plt.grid()
        
        plt.show()

    def plot_dyn_list2D(self,step,tstep_list,axis,velocity=False,delta=False,
                         mask=0,zlim0=False,zlim1=False,
                         xlim0=False,xlim1=False,hline0=False,hline1=False,hline2=False,):
        # for i in (np.arange(0,t1-t0,plot_step)):    
        #     f2_ax1.plot(obs_data1[:,i][fsort],x[:-1][fsort],label=str(t_step[i])+'ms')

        x = self.fcoord[:,0][:]*1e0
        y = self.fcoord[:,1][:]*1e0
        fig = plt.figure(figsize=(5,6))
        ax1 = fig.add_subplot(1, 1, 1)
        for i in tstep_list: 
            slip_shear = (self.dat_slip_sort['step '+ str(step)][:,0,i])
            slip_normal = (self.dat_slip_sort['step '+ str(step)][:,1,i])
            trac_shear = (self.dat_trac_sort['step '+ str(step)][:,0,i])
            trac_normal = (self.dat_trac_sort['step '+ str(step)][:,1,i])
            trac_f2 = (self.dat_trac_sort['step '+ str(step)][:,2,i]) # friction
            trac_f3 = (self.dat_trac_sort['step '+ str(step)][:,3,i]) # counter normal force
            if velocity:
                slip_shear = np.sum(slip_shear*dt_dyn)
            if axis == 0:
                z1 = slip_shear
            elif axis == 1:
                z1 = slip_normal    
            elif axis == 2:
                z1 = trac_shear/1e6
            elif axis == 3:
                z1 = trac_normal/1e6
            elif axis == 4:
                z1 = trac_f2/1e6
            elif axis == 5:
                z1 = trac_f3/1e6
            elif axis == 6:
                z1 = trac_shear/trac_f2

            ax1.plot(z1[self.fsort],y[self.fsort],'--',label=("step" + str(i))) # label=str(t_step[i])+'ms')
        print ()

        ax1.legend(loc='lower left')
        ax1.set_ylabel('Z (km)', fontsize = 20)
        if hline0:  ax1.axhline(y=hline0, color='r', linestyle='--')
        if hline1:  ax1.axhline(y=hline1, color='g', linestyle='--')
        if hline2:  ax1.axhline(y=hline2, color='b', linestyle='--')
        
        if zlim1:
            ax1.set_ylim(zlim0,zlim1)
        if xlim1:
            ax1.set_xlim(xlim0,xlim1)
        if axis == 0:
            ax1.set_xlabel(r'$shear\ slip\ (m)$', fontsize = 20)
        elif axis == 1:
            ax1.set_xlabel(r'$normal\ disp\ (m)$', fontsize = 20)
        elif axis == 2:
            ax1.set_xlabel(r'$shear\ trac\ (m)$', fontsize = 20)
        elif axis == 3:
            ax1.set_xlabel(r'$normal\ trac\ (m)$', fontsize = 20)
        elif axis == 4:
            ax1.set_xlabel(r'$friction\ (m)$', fontsize = 20)
        elif axis == 5:
            ax1.set_xlabel(r'$normal(counter)\  (m)$', fontsize = 20)
        elif axis == 6:
            ax1.set_xlabel(r'$SCU$', fontsize = 20)
        fig.tight_layout()
        plt.grid()
        
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
