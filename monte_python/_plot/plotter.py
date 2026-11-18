import matplotlib.pyplot as plt 
import matplotlib
import numpy as np 
from scipy.stats import multivariate_normal  # noqa: F401
from .._synthetic import create_fake_storms  # noqa: F401  (re-export)
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.colors import ListedColormap
cmap = ListedColormap(["white", "red", "blue", "green", "purple", 'gray'])


### NWS Reflectivity Colors (courtesy MetPy library):
c5 =  (0.0,                 0.9254901960784314, 0.9254901960784314)
c10 = (0.00392156862745098, 0.6274509803921569, 0.9647058823529412)
c15 = (0.0,                 0.0,                0.9647058823529412)
c20 = (0.0,                 1.0,                0.0)
c25 = (0.0,                 0.7843137254901961, 0.0)
c30 = (0.0,                 0.5647058823529412, 0.0)
c35 = (1.0,                 1.0,                0.0)
c40 = (0.9058823529411765,  0.7529411764705882, 0.0)
c45 = (1.0,                 0.5647058823529412, 0.0)
c50 = (1.0,                 0.0,                0.0)
c55 = (0.8392156862745098,  0.0,                0.0)
c60 = (0.7529411764705882,  0.0,                0.0)
c65 = (1.0,                 0.0,                1.0)
c70 = (0.6,                 0.3333333333333333, 0.788235294117647)
c75 = (0.0,                 0.0,                0.0) 

nws_dz_cmap = matplotlib.colors.ListedColormap([c20, c25, c30, c35, c40, c45, 
                 c50, c55, c60, c65, c70])
dz_levels_nws = np.arange(20.0,80.,5.)

def convert_modes(modes):
    new_modes = np.zeros(modes.shape)
    nan_inds = modes<0 
    new_modes[modes>=0] = modes[modes>=0]
    new_modes[modes>=0]+= 1
    new_modes[nan_inds] = 0
    return new_modes



# Create fake storm data
def plot_fake_storms(x,y,data, ax=None, colorbar=True, alpha=1.0):
    """ Plot fake storms """
    if ax is None:
        f, ax = plt.subplots(figsize=(5, 4), dpi=150, facecolor='w', edgecolor='k')  
        
    c = ax.pcolormesh(x, y, data, cmap=nws_dz_cmap, vmin=20, vmax=75, alpha=alpha)
    ax.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    if colorbar:
        plt.colorbar(c, label='Fake Reflectivity', ax=ax)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(alpha=0.5, ls='dashed')
    
    return ax 


def label_centroid(x, y, ax, object_props, area_thresh=0, label_modes=False):
    """Place object label on object's centroid"""
    #if storm_modes is not None:
    #    print('Remember that the storm mode label is based on the minimal integer label in a reigon!')
    
    for region in object_props:
        x_cent,y_cent = region.centroid
        x_cent=int(x_cent)
        y_cent=int(y_cent)
        xx, yy = x[x_cent,y_cent], y[x_cent,y_cent]
        
        if label_modes:
            fontsize=4
            #coords = region.coords
            #ind = int(np.min(storm_modes[coords[:,0], coords[:,1]]))
            txt = region.storm_mode
        else:
            fontsize = 6.5 if region.label >= 10 else 8
            if region.area >= area_thresh:
                txt = region.label
            else:
                txt = '*'; fontsize=4
        ax.text(xx,yy,
                    txt,
                    fontsize=fontsize,
                    ha='center',
                    va='center',
                    color = 'k'
                    )    

def plot_storm_labels(x, y, labels, label_props, ax=None, alpha=1.0, area_thresh=0):
    """ Plot Storm Labels """
    if ax is None:
        f, ax = plt.subplots(figsize=(5, 4), dpi=150, facecolor='w', edgecolor='k')
    
    labels = np.ma.masked_where(labels==0, labels)
    c = ax.pcolormesh(x, y, labels, cmap='tab20', vmin=1, vmax=np.max(labels), alpha=alpha)
    ax.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(alpha=0.5, ls='dashed')
    
    label_centroid(x, y, ax, label_props, area_thresh=area_thresh) 
    
    return ax    

def plot_storm_modes(x, y, modes, label_props=None, ax=None, alpha=1.0):
    """ Plot Storm Labels """
    if ax is None:
        f, ax = plt.subplots(figsize=(5, 4), dpi=150, facecolor='w', edgecolor='k')
    
    modes = np.ma.masked_where(modes==0, modes)
    norm = colors.BoundaryNorm(0.5+np.arange(7 + 1), plt.cm.Set1.N)
    #ax.contourf(x,y,rot_vals, cmap='jet', levels=np.arange(25, 175, 12.5))
    c = ax.pcolormesh(x, y, modes, cmap='Set1', alpha=alpha, norm=norm)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    cbar = plt.colorbar(c, ticks=range(8), label='Storm Mode', cax=cax)
    cbar.ax.set_yticklabels(['ORDINARY', 'SUPERCELL', 'QLCS', 
                                 'SUP_CLUST', 'QLCS_ORD', 'QLCS_MESO', 'OTHER' ], fontsize=5)
    
    ax.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(alpha=0.5, ls='dashed')
    
    if label_props is not None:
        label_centroid(x, y, ax, label_props, label_modes=True) 
    
    return ax    


def get_centroid_coords(label_props):
    '''Creates a dictionary with the object labels as keys 
    and the tuple of centroid coordinates as items'''
    centroids = {}
    for region in label_props:
        centroids[region.label] = region.centroid
    return centroids


        
def matching_path(label_cent_0, label_cent_1, matched_0, matched_1):
    '''Associate matched objects and create their matching path'''
    x_set = []
    y_set = []
    for label_0, label_1 in zip(matched_0, matched_1):
        cent_0 = label_cent_0[label_0]
        cent_1 = label_cent_1[label_1]
        
        x1 = (cent_0[1], cent_1[1])
        y1 = (cent_0[0], cent_1[0])
        x_set.append(x1)
        y_set.append(y1)
    return x_set, y_set
    

def plot_displacement(dists, ax=None):
    if ax is None:
        f, ax = plt.subplots(figsize=(5, 4), dpi=150, facecolor='w', edgecolor='k')

    dy = [d[0] for d in dists]
    dx = [d[1] for d in dists]

    ax.scatter(dx, dy) 
    ax.set_xlabel('X-Displacement')
    ax.set_ylabel('Y-Displacement')

    return ax 




    
    
    


    
