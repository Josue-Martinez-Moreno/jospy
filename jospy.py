# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import cmocean
from scipy.interpolate import griddata
from scipy.interpolate import interp2d
import time
import datetime
from matplotlib import ticker

def find(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def sind(x):
    return np.sin(np.radians(x))

def cosd(x):
    return np.cos(np.radians(x))

def tand(x):
    return np.tan(np.radians(x))

def pcolor_cute(X, Y, mask, data, coastline, n=1,cbpos=[], Xb=[], Yb=[], plotname='',namefile='cuteplot',outputfile="./", cmap='', scale='', vmin='', vmax='', mask_value=0,formatfile='png',savefile=True,nbins=''):
    if len(np.shape(X))== 1 and len(np.shape(Y)) == 1:
        X,Y=np.meshgrid(X,Y)
    elif len(np.shape(X))== 2 and len(np.shape(Y)) == 2:
	print "Good Grid"
    else:
  	print "The grids must be a matrix of dimensions (N,M)"

    if cmap == 'thermal':
        cmap = cmocean.cm.thermal
    elif cmap == 'haline':
        cmap = cmocean.cm.haline
    elif cmap == 'solar':
        cmap = cmocean.cm.solar
    elif cmap == 'ice':
        cmap = cmocean.cm.ice
    elif cmap == 'gray':
        cmap = cmocean.cm.gray
    elif cmap == 'oxy':
        cmap = cmocean.cm.oxy
    elif cmap == 'deep':
        cmap = cmocean.cm.deep_r
    elif cmap == 'dense':
        cmap = cmocean.cm.dense
    elif cmap == 'algae':
        cmap = cmocean.cm.algae
    elif cmap == 'matter':
        cmap = cmocean.cm.matter
    elif cmap == 'turbid':
        cmap = cmocean.cm.turbid
    elif cmap == 'speed':
        cmap = cmocean.cm.speed
    elif cmap == 'amp':
        cmap = cmocean.cm.amp
    elif cmap == 'tempo':
        cmap = cmocean.cm.tempo
    elif cmap == 'phase':
        cmap = cmocean.cm.phase
    elif cmap == 'balance':
        cmap = cmocean.cm.balance
    else:
        cmap = cmocean.cm.speed
    data_masked=np.ma.masked_where(mask==mask_value, data)

    cmap.set_bad(color ='silver', alpha = 1.)

    if n==1 and Xb==[] and Yb==[]:
        if cbpos==[]:
            cbpos=[0.15, 0.85, 0.35, 0.03]
        fig,im,cax,ax=single_plot(X,Y,data_masked,coastline,scale,vmin,vmax,cmap,cbpos)
    else:
        if cbpos==[]:
            cbpos=[0.34, 0.95, 0.35, 0.03]
        fig,im,cax,ax=multiple_plot(X,Y,data_masked,coastline,n,Xb,Yb,scale,vmin,vmax,cmap,cbpos)

    cbar=fig.colorbar(im, cax=cax, orientation='horizontal', extend='both')
    cbar.set_label(plotname.decode('utf-8'))
    if nbins!='':
        tick_locator = ticker.MaxNLocator(nbins=nbins) 
        cbar.locator = tick_locator
        cbar.update_ticks()

    ax.set_aspect('equal')
    plt.rcParams.update({'font.size': 22})
    if savefile==True:
        plt.savefig(outputfile+namefile+"."+formatfile,format=formatfile,dpi=400)
        plt.close()
    return ax
    

def single_plot(X,Y,data_masked,coastline,scale,vmin,vmax,cmap,cbpos):
    fig, ax = plt.subplots(figsize=(20,12.5))
    cax = fig.add_axes(cbpos)
    if scale == '' and vmin == '' and vmax == '':
        vmin=data_masked.min()
        vmax=data_masked.max()
        im=ax.pcolormesh(X,Y,data_masked,cmap=cmap,vmin=vmin,vmax=vmax)
    elif scale == 'SymLog' and vmin == '' and vmax == '':
        vmin=data_masked.min()
        vmax=data_masked.max()
        im=ax.pcolormesh(X,Y,data_masked,cmap=cmap,norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=vmin, vmax=vmax))
    elif scale == '' and vmin != '' and vmax != '':
        im=ax.pcolormesh(X,Y,data_masked,cmap=cmap,vmin=vmin, vmax=vmax)
    elif scale == 'SymLog' and vmin != '' and vmax != '':
        im=ax.pcolormesh(X,Y,data_masked,cmap=cmap,norm=colors.SymLogNorm(linthresh=0.01, linscale=0.01,vmin=vmin, vmax=vmax))
    elif scale == 'Log' and vmin != '' and vmax != '':
        im=ax.pcolormesh(X,Y,abs(data_masked),cmap=cmap,norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    ax.locator_params(axis='x',nbins=10)
    #xloc,lab=ax.set_xticks()
    fig.canvas.draw()

    min_x=X[1,1]
    min_y=Y[1,1]
    
    if min_x < 0:
        xlabel = [item.get_text()for item in ax.get_xticklabels()]
        xlabel=[(lab)+unicode(' ºW','utf-8') for lab in xlabel]
        #xlabel=[unicode(str(item)+' ºW',"utf-8") for item in xloc]
    elif min_x > 0:
        xlabel = [item.get_text() for item in ax.get_xticklabels()]
        xlabel=[str(lab)+unicode(' ºE','utf-8') for lab in xlabel]
        #xlabel=[unicode(str(item)+' ºE',"utf-8") for item in xloc]
    #yloc,lab=ax.set_yticks()
    if min_y < 0:
        ylabel = [item.get_text() for item in ax.get_yticklabels()]
        ylabel=[str(lab)+unicode(' ºS','utf-8') for lab in ylabel]
        #ylabel=[unicode(str(item)+' ºS',"utf-8") for item in yloc]
    elif min_y > 0:
        ylabel = [item.get_text() for item in ax.get_yticklabels()]
        ylabel=[str(lab)+unicode(' ºN','utf-8') for lab in ylabel]
        #ylabel=[unicode(str(item)+' ºN',"utf-8") for item in yloc]

    ax.set_xticklabels(xlabel)
    ax.set_yticklabels(ylabel)
    ax.grid()
#    ax.colorbar(extend='both', orientation='horizontal')
    ax.plot(coastline[:,0],coastline[:,1],'.k',markersize=1.)
    ax.axis([X.min(), X.max(), Y.min(), Y.max()]);
    return fig,im,cax,ax


def multiple_plot(X,Y,data_masked,coastline,n,Xb,Yb,scale,vmin,vmax,cmap,cbpos):
    fig, ax = plt.subplots(n/2,n/2,figsize=(20,12.5))
    cax = fig.add_axes(cbpos)
        
    for ii in range(0,n):
	if ii % 2 == 0:
             j=0
	else:
             j=1
	i=ii/2

	X_i=Xb[2*ii]
        X_f=Xb[2*ii+1]
	Y_i=Yb[2*ii]
	Y_f=Yb[2*ii+1]

        if scale == '' and vmin == '' and vmax == '':
            vmin=data_masked.min()
            vmax=data_masked.max()
            im=ax[i,j].pcolormesh(X[Y_i:Y_f,X_i:X_f],Y[Y_i:Y_f,X_i:X_f],data_masked[Y_i:Y_f,X_i:X_f],cmap=cmap,vmin=vmin,vmax=vmax)
        elif scale == 'SymLog' and vmin == '' and vmax == '':
            vmin=data_masked.min()
            vmax=data_masked.max()
            im=ax[i,j].pcolormesh(X[Y_i:Y_f,X_i:X_f],Y[Y_i:Y_f,X_i:X_f],data_masked[Y_i:Y_f,X_i:X_f],cmap=cmap,norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=vmin, vmax=vmax))
        elif scale == '' and vmin != '' and vmax != '':
            im=ax[i,j].pcolormesh(X[Y_i:Y_f,X_i:X_f],Y[Y_i:Y_f,X_i:X_f],data_masked[Y_i:Y_f,X_i:X_f],cmap=cmap,vmin=vmin, vmax=vmax)
        elif scale == 'SymLog' and vmin != '' and vmax != '':
            im=ax[i,j].pcolormesh(X[Y_i:Y_f,X_i:X_f],Y[Y_i:Y_f,X_i:X_f],data_masked[Y_i:Y_f,X_i:X_f],cmap=cmap,norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=vmin, vmax=vmax))
	
	
        ax[i,j].locator_params(axis='x',nbins=10)
        #xloc,lab=ax.set_xticks()
        fig.canvas.draw()
            
        min_x=X[1,1]
        min_y=Y[1,1]
        
        if min_x < 0:
            xlabel = [item.get_text()for item in ax[i,j].get_xticklabels()]
            xlabel=[(lab)+unicode(' ºW','utf-8') for lab in xlabel]
            #xlabel=[unicode(str(item)+' ºW',"utf-8") for item in xloc]
        elif min_x > 0:
            xlabel = [item.get_text() for item in ax[i,j].get_xticklabels()]
            xlabel=[str(lab)+unicode(' ºE','utf-8') for lab in xlabel]
            #xlabel=[unicode(str(item)+' ºE',"utf-8") for item in xloc]
        #yloc,lab=ax.set_yticks()
        if min_y < 0:
            ylabel = [item.get_text() for item in ax[i,j].get_yticklabels()]
            ylabel=[str(lab)+unicode(' ºS','utf-8') for lab in ylabel]
            #ylabel=[unicode(str(item)+' ºS',"utf-8") for item in yloc]
        elif min_y > 0:
            ylabel = [item.get_text() for item in ax[i,j].get_yticklabels()]
            ylabel=[str(lab)+unicode(' ºN','utf-8') for lab in ylabel]
            #ylabel=[unicode(str(item)+' ºN',"utf-8") for item in yloc]
    
        ax[i,j].set_xticklabels(xlabel)
        ax[i,j].set_yticklabels(ylabel)
        ax[i,j].grid()
        ax[i,j].plot(coastline[:,0],coastline[:,1],'.k',markersize=1.)
        ax[i,j].axis([X[Y_i:Y_f,X_i:X_f].min(), X[Y_i:Y_f,X_i:X_f].max(), Y[Y_i:Y_f,X_i:X_f].min(), Y[Y_i:Y_f,X_i:X_f].max()]);
    return fig,im,cax,ax   

def get_mask(data):
    mask=np.zeros(np.shape(data))
    
    for ii in range(0,data.shape[1]):
        for jj in range(0,data.shape[0]):
            if data[jj,ii]>0:
                mask[jj,ii]=1
            else:
                mask[jj,ii]=0
    return mask

def cumsumEW(zerodist,distarray,length=0):
    """
    INPUT:
    zerodist: Start point the zero to go east or west.
    distarray: array of distances.
    length: length of original array
    OUTPUT:
    cumsumE,cumsumW: Acumulated sum to the east or west of zerodist
    """
    if length==0:
        length=len(distarray)
    sum_dist=0
    cumsumW=[]
    for ii in range(1,zerodist+1):
        sum_dist=sum_dist+distarray[zerodist-ii]
        cumsumW.append(-sum_dist)
    sum_dist=0
    cumsumE=[]
    for ii in range(zerodist,length-1):
        sum_dist=sum_dist+distarray[ii]
        cumsumE.append(sum_dist)
    return cumsumW,cumsumE


def delta_latlon(delta_x,delta_y,lat):
    """ 
    INPUT:
    delta_x: Delta in meters longitude
    delta_y: Delta in meters latitude
    lat: Point latitude
    OUTPUT:
    delta_lon: Delta in longitude
    delta_lat: Delta in latitude
    """
    R=6378137
    delta_lon=(delta_x/(R*np.cos(np.pi*lat/180)))*180/np.pi
    delta_lat=(delta_y/R)*180/np.pi
    return delta_lon,delta_lat

def delta_dist(mydist,zerodist,distarray,alphadist,length=0):
    """
    INPUT:
    mydist: The distance where I want to obtain the perpendicular vector.
    zerodist: Start point the zero to go east or west.
    distarray: array of distances.
    alphadist: angle between distances.
    length: length of original array
    ***Use dist of seawater Library*****
    OUTPUT:
    nearest_point: The nearest point to the chosen distance
    deltadist_x: Delta between my point and the nearest in x 
    deltadist_y: Delta between my point and the nearest in y
    """
    if length==0:
        length=len(distarray)
    if mydist < 0:
        for ii in range(0,zerodist-1):
            if distarray[ii] <= mydist:
                nearest_point=ii

        deltadist=distarray[nearest_point]-mydist
        deltadist_x=cosd(alphadist[nearest_point-1])*deltadist
        deltadist_y=sind(alphadist[nearest_point-1])*deltadist
    elif mydist== 0:
        nearest_point=zerodist
        deltadist_x=0
        deltadist_y=0
    
    elif mydist> 0:
        for ii in range(zerodist,length-1):
            if distarray[ii] <= mydist:
                nearest_point=ii

        deltadist=distarray[nearest_point]-mydist
        deltadist_x=cosd(alphadist[nearest_point-1])*deltadist
        deltadist_y=sind(alphadist[nearest_point-1])*deltadist
    return nearest_point,deltadist_x,deltadist_y



def perp2coast(X1,X2,Y1,Y2,X0=0,Y0=0,hip=10000,deltahip=1000,units='m'):
    """
    INPUT:
    X1: Initial point in X for compute slope
    X2: Final point in X for compute slope
    Y1: Initial point in Y for compute slope
    Y2: Final point in Y for compute slope
    X0: Point to colocate the slope in X
    Y0: Point to colocate the slope in Y

    OUTPUT:
    
    """

    slope=0
    slope=((Y2-Y1)/(X2-X1))
    angle=np.arctan(slope)*180.0/np.pi
    
    if X0==0 and Y0==0:
        X0=X1
        Y0=Y1
    
    if slope==0:
        perp_slope=inf
    else:
        perp_slope=abs(-1/slope)
    
    perp_angle=np.arctan(perp_slope)*180.0/np.pi
    
    if angle> 0 and slope!=0:
        x_perpslope=-hip*cosd(perp_angle)
        y_perpslope=hip*sind(perp_angle)
    else:
        x_perpslope=hip*cosd(perp_angle)
        y_perpslope=hip*sind(perp_angle)

    x_normalslope=1*cosd(angle)
    y_normalslope=1*sind(angle)
    
    vec_perp_x=[]
    vec_perp_y=[]

    dist_perp=0
    
    while dist_perp <= hip:
        X_slope_vect=dist_perp*cosd(perp_angle)
        Y_slope_vect=dist_perp*sind(perp_angle)
        
        if units=='latlon':
            X_slope_vect,Y_slope_vect=delta_latlon(X_slope_vect,Y_slope_vect,Y0)

        if angle> 0 and slope!=0 :
            vec_perp_x.append(X0-X_slope_vect)
        else:
            vec_perp_x.append(X0+X_slope_vect)
        vec_perp_y.append(Y0+Y_slope_vect)
        dist_perp=dist_perp+deltahip
    return vec_perp_x,vec_perp_y,x_perpslope,y_perpslope,x_normalslope,y_normalslope

def transperpcoast(pos,t,lon,lat,var,parm,U,V,Z,levels,days,zerodist,distarray,alphadist,coastline,hip=10000,deltahip=1000,lon1=[],lat1=[],length=0,cplot=False):
    """
    
    """
    global trans_h_u
    global trans_h_u_n_p
    if t==0 and days>1:
        trans_h_u=np.zeros([len(pos),days])
        trans_h_u_n_p=np.zeros([len(pos),days])
    elif days == 1:
        trans_h_u=np.zeros([len(pos)])
        trans_h_u_n_p=np.zeros([len(pos)])
    
    vector_coord_x=np.zeros([len(pos),hip/deltahip+1])
    vector_coord_y=np.zeros([len(pos),hip/deltahip+1])
    x_perpslope=np.zeros([len(pos)])
    y_perpslope=np.zeros([len(pos)])
    x_normalslope=np.zeros([len(pos)])
    y_normalslope=np.zeros([len(pos)])
    new_coord_X=np.zeros([len(pos)])
    new_coord_Y=np.zeros([len(pos)])

    size=var.shape

    for ll in range(0,len(pos)): 

        nearest_point,deltadist_x,deltadist_y = delta_dist(pos[ll],zerodist,distarray,alphadist,length=0)

        delta_lon,delta_lat=delta_latlon(deltadist_x,deltadist_y,coastline[nearest_point,1])
      
        new_coord_X[ll]=coastline[nearest_point,0]+delta_lon;
        new_coord_Y[ll]=coastline[nearest_point,1]+delta_lat;

        vector_coord_x[ll,:],vector_coord_y[ll,:],x_perpslope[ll],y_perpslope[ll],x_normalslope[ll],y_normalslope[ll]=perp2coast(coastline[nearest_point,0],coastline[nearest_point-1,0],coastline[nearest_point,1],coastline[nearest_point-1,1],new_coord_X[ll],new_coord_Y[ll],hip,deltahip,units='latlon')

        vector_x, vector_y = np.meshgrid(vector_coord_x[ll,:],vector_coord_y[ll,:])
        
        Z2interp=interp2d(lon, lat, Z[:,:], kind='linear')
        Z_interp = Z2interp(vector_coord_x[ll,:],vector_coord_y[ll,:])
        
        if lon1 != [] and lat1 != [] :
            for zz in range(0,size[0]):
                U_interp=interp2d(lon, lat, U[zz,:,:], kind='linear')
                V_interp=interp2d(lon, lat, V[zz,:,:], kind='linear')
                U[zz,:,:] = U_interp(lon1,lat1)
                V[zz,:,:] = V_interp(lon1,lat1)
        else:
            lon1=lon
            lat1=lat

        data_coatz_interp=np.zeros([size[0],len(vector_coord_x[ll,:]),len(vector_coord_y[ll,:])])
        vel_u_coatz_interp=np.zeros([size[0],len(vector_coord_x[ll,:]),len(vector_coord_y[ll,:])])
        vel_v_coatz_interp=np.zeros([size[0],len(vector_coord_x[ll,:]),len(vector_coord_y[ll,:])])
        
        for zz in range(0,size[0]):
            data_interp=interp2d(lon, lat, var[zz,:,:], kind='linear')
            u_interp=interp2d(lon1, lat1, U[zz,:,:], kind='linear')
            v_interp=interp2d(lon1, lat1, V[zz,:,:], kind='linear')
            data_coatz_interp[zz,:,:]=data_interp(vector_coord_x[ll,:],vector_coord_y[ll,:])
            vel_u_coatz_interp[zz,:,:]=u_interp(vector_coord_x[ll,:],vector_coord_y[ll,:])
            vel_v_coatz_interp[zz,:,:]=v_interp(vector_coord_x[ll,:],vector_coord_y[ll,:])

        max_depth=Z_interp.max()
        for pp in range(0,len(vector_coord_y)):
            for zz in range (0,size[0]):
                if data_coatz_interp[zz,pp,pp]>=parm and Z_interp[pp,pp]<=max_depth:
                    trans_h=((vel_u_coatz_interp[zz,pp,pp]*x_normalslope[ll])+(y_normalslope[ll]*vel_v_coatz_interp[zz,pp,pp]))*(deltahip)*levels[zz]*data_coatz_interp[zz,pp,pp]
                    trans_h_n_p=((vel_u_coatz_interp[zz,pp,pp]*x_normalslope[ll])+(y_normalslope[ll]*vel_v_coatz_interp[zz,pp,pp]))*(deltahip)*levels[zz]
                    trans_h_u[ll,t]=(trans_h_u[ll,t]+trans_h)
                    trans_h_u_n_p[ll,t]=(trans_h_u_n_p[ll,t]+trans_h_n_p) 
        if cplot==True and t==0:
            plt.plot(coastline[:,0],coastline[:,1],'-r')
            plt.plot(vector_coord_x[ll,:],vector_coord_y[ll,:])
            plt.plot(coastline[nearest_point,0],coastline[nearest_point,1],'om')
            plt.plot(coastline[nearest_point-1,0],coastline[nearest_point-1,1],'og')
            plt.gca().set_aspect('equal')
    if cplot==True and t==0:
        plt.show()
    return vector_coord_x,vector_coord_y,x_perpslope,y_perpslope,trans_h_u,trans_h_u_n_p,new_coord_X,new_coord_Y,Z_interp,data_coatz_interp,vel_u_coatz_interp,vel_u_coatz_interp

def multicore():
    pool = multiprocessing.Pool( 8 )
    tic = time.time()
    print 'tic '+str(tic)
    if __name__ == '__main__':
        default=multiprocessing.cpu_count()
     
        # Start my pool
        pool = multiprocessing.Pool(default)
        outputDir='./'
        plotNum = 4
                             
        lon1=[]
        lat1=[]
        # Run tasks
        results = pool.apply_async( transperpcoast, (position_in_km,tt,XG_coatz,YG_coatz,data_coatz,concentration,vel_u_coatz,vel_v_coatz,Depth_coatz,DRC,num_days,river_position_good_line_coast,position_distance_good_array[:,1],position_distance_good_array[:,2],coast_line_coatz,hip,deltahip,lon1,lat1,coast_good_size[0]))
        # Process results
        (vector_coord_x,vector_coord_y,vector_x,vector_y,trans_h_u,trans_h_u_n_p) = results.get()
    
        pool.close()
        pool.join()
     
    toc = time.time()
    print 'toc '+str(toc)
    print '1: '+str(toc - tic)
 
    #lon1=[]
    #lat1=[]
    #vector_coord_x,vector_coord_y,vector_x,vector_y,trans_h_u,trans_h_u_n_p=pool.apply_async( transperpcoast, (position_in_km,tt,XG_coatz,YG_coatz,data_coatz,concentration,vel_u_coatz,vel_v_coatz,Depth_coatz,DRC,num_days,river_position_good_line_coast,position_distance_good_array[:,1],position_distance_good_array[:,2],coast_line_coatz,hip,deltahip,lon1,lat1,coast_good_size[0]))
    tic = time.time()

























