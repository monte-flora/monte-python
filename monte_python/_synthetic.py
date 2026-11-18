"""Synthetic storms for tests and tutorials (no matplotlib)."""
import numpy as np
from scipy.stats import multivariate_normal


def create_fake_storms(centers, add_small_area=False, nx=150):
    """ Create artifical storms """
    rng = range(nx)
    x,y = np.meshgrid(rng, rng)
    coords = np.dstack((x, y))
    
    storm_types = [[[50, 30], [30, 30]], 
                   [[40, 20], [20, 20]],
                   [[60, 20], [20, 50]],
                  ]
    
    random_state = np.random.RandomState(123)
    inds = random_state.choice(len(storm_types), size=len(centers), p=[0.4, 0.3, 0.3])
    
    data = 0 
    for ind, (i, j) in zip(inds, centers):
        data += (1e4 * multivariate_normal(mean=[i, j], cov=np.array(storm_types[ind])).pdf(coords))

    if add_small_area:
        small_area = (1e4 * multivariate_normal(mean=[70, 70], cov=np.array( [[50, 30], [30, 50]])).pdf(coords))
        small_area[small_area<30] = 0 
        data += small_area
        
        i,j = centers[0]
        small_area = (1e4 * multivariate_normal(mean=[i+13,j+13], cov=np.array([[50, 30], [30, 50]])).pdf(coords))-5.
        small_area[small_area<30] = 0
        data += small_area
    
    data = np.ma.masked_where(data<10., data)    
        
    return data, x, y
