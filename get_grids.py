import os
import xarray
import xarray as xr
from glob import glob
fn = sorted(glob('Crossover/*'))

def get_grid(fn):
    dd=xr.open_dataset(fn)
    tt=dd['Theta'].shape
    size=os.path.getsize(fn)/1024/1024
    lat=dd['YC'][:,0].mean()
    lon=dd['XC'][0,:].mean()
    print('%20s , %5.1f, %5.1f, %4i, %4i, %4i, %4i, %4i, %8i'%(fn.split('/')[1],lat,lon,nn,tt[0],tt[1],tt[2],tt[3],size))
    return tt

def plotit(fn):
    dd=xr.open_dataset(fn)



for ff in fn[:1]:
    fns=glob(ff+'/*.nc')
    nn=len(fns)
    #get_grid(fns[0])
    plotit(fns[0])

   

