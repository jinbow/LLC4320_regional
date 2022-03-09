import numpy as np
import xarray as xr
from glob import glob
import pandas as pd
import json,os
import netCDF4 as nc4


netcdf_fill_value = nc4.default_fillvals['f4']
pth='/home6/dmenemen/llc_4320/regions/'
pth='/home6/dmenemen/llc_4320/regions/Crossover/'


def parse(fn):
    nn=fn.split('_')[-1].split('.')
    nx,ny,nz=int(nn[-3]),int(nn[-2]),int(nn[-1])
    nn=fn.split('_')[-2].split('.')
    i0,j0,k0=int(nn[-3]),int(nn[-2]),int(nn[-1])

    tt=fn.split('/')[-1][:10]
    return tt,nx,ny,nz,i0,j0,k0

def read_grid(ph,nx,ny,nz):
    """read coordinate metadata for the model grid"""

    rc=np.fromfile('/home6/dmenemen/llc_4320/grid/RC.data','>f4')
    rf=np.fromfile('/home6/dmenemen/llc_4320/grid/RF.data','>f4')

    a={}
    f=open('ECCOv4r4_geometry_metadata_for_native_datasets.json')
    ff=json.load(f)

    for fff in ff:
   
        cc=fff.copy()
        del cc['name']
        #del cc['grid_dimension']
        
        #print(cc)
        a[fff['name']]=cc
        
    fns=glob(ph+'/grid/*x*')
    dd={}
    for fn in fns:
        var=fn.split('/')[-1].split('_')[0] 
        #print(var)
        if 'hFac' in var:
            dd[var]=np.fromfile(fn,'>f4').reshape(nz,ny,nx)
        else:
            dd[var]=np.fromfile(fn,'>f4').reshape(ny,nx)
    
    #print(dd.keys())
    i=i_g=np.arange(nx,dtype='int16')
    j=j_g=np.arange(ny,dtype='int16')
    k=k_u=k_l=np.arange(nz)
    k_p1=np.arange(nz+1)

    dout={}
    for na in ['i','i_g']:
        dout[na]=xr.DataArray(i,dims=(na),coords={na:i},
                            attrs=a[na])
    for na in ['j','j_g']:
        dout[na]=xr.DataArray(j,dims=(na),coords={na:j},
                            attrs=a[na])
    for na in ['k','k_u','k_l']:
        dout[na]=xr.DataArray(k,dims=(na),coords={na:k},
                            attrs=a[na])
    dout['k_p1']=xr.DataArray(k_p1,dims=('k_p1'),coords={'k_p1':k_p1},
                            attrs=a[na])                         

    for na in ['XC','YC','DXV','DYU','Depth','AngleSN','AngleCS']:
        try: 
            aa=a[na]
            try:
                del aa['GCMD_keywords']
            except:
                pass
            dout[na]=xr.DataArray(dd[na],dims=('j','i'),coords={'j':dout['j'],'i':dout['i']},attrs=a[na])
        except:
            pass

        
    for na in ['DXC','DYG']:
        dout[na]=xr.DataArray(dd[na],dims=('j','i_g'),coords={'j':dout['j'],'i_g':dout['i_g']},attrs=a[na])

    for na in ['DYC','DXG']:
        dout[na]=xr.DataArray(dd[na],dims=('j_g','i'),coords={'j_g':dout['j_g'],'i':dout['i']},attrs=a[na])

    for na in ['XG','YG','RAZ']:
        dout[na]=xr.DataArray(dd[na],dims=('j_g','i_g'),coords={'j_g':dout['j_g'],'i_g':dout['i_g']},attrs=a[na])

    if na=='hFacC':
        dout[na]=xr.DataArray(dd[na],dims=('k','j','i'),coords={'k':dout['k'],'j':dout['j'],'i':dout['i']},attrs=a[na])
        
    if na=='hFacW':
        dout[na]=xr.DataArray(dd[na],dims=('k','j','i_g'),coords={'k':dout['k'],'j':dout['j'],'i_g':dout['i_g']},attrs=a[na])

    if na=='hFacS':
        dout[na]=xr.DataArray(dd[na],dims=('k','j_g','i'),coords={'k':dout['k'],'j_g':dout['j_g'],'i':dout['i']},attrs=a[na])


    c=dd['XC']
    dc=(c[:,1:]-c[:,0:-1])/2.0
    dc=np.c_[dc[:,0:1],dc]
    cbnd=np.zeros((ny,nx,2))
    cbnd[:,:,0]=c-dc
    cbnd[:,:,1]=c+dc
    dout['XC_bnds']=xr.DataArray(cbnd,dims=('j','i','nb'),coords={'j':dout['j'],'i':dout['i'],'nb':[0,1]},attrs=a['XC_bnds'])

    c=dd['YC']
    dc=(c[1:,:]-c[0:-1,:])/2.0
    dc=np.r_[dc[0:1,:],dc]
    cbnd=np.zeros((ny,nx,2))
    cbnd[:,:,0]=c-dc
    cbnd[:,:,1]=c+dc
    dout['YC_bnds']=xr.DataArray(cbnd,dims=('j','i','nb'),coords={'j':dout['j'],'i':dout['i'],'nb':[0,1]},attrs=a['YC_bnds'])


    dout['Z']=xr.DataArray(rc[:nz],dims=('k'),coords={'k':dout['k']},attrs=a['Z'])
    dout['Zp1']=xr.DataArray(rf[:nz+1],dims=('k_p1'),coords={'k_p1':dout['k_p1']},attrs=a['Zp1'])
    dout['Zu']=xr.DataArray(rf[1:nz+1],dims=('k_u'),coords={'k_u':dout['k_u']},attrs=a['Zu'])
    dout['Zl']=xr.DataArray(rf[:nz],dims=('k_l'),coords={'k_l':dout['k_l']},attrs=a['Zl'])
    dout['Z_bnds']=xr.DataArray(np.c_[rf[:nz].ravel(),rf[1:nz+1].ravel()], dims=('k','nb'),coords={'k':dout['k'],'nb':[0,1]},attrs=a['Z_bnds']) 

    dout=xr.Dataset(dout)
    G=dout
    dv_encoding={}
    if True:
        for coord in dout.coords:
            dv_encoding[coord]={'_FillValue':None, 'dtype':'float32'}
            if (G[coord].values.dtype == np.int32) or \
                   (G[coord].values.dtype == np.int64) :
                dv_encoding[coord]['dtype'] ='int32'

                if coord == 'time' or coord == 'time_bnds':
                    dv_encoding[coord]['dtype'] ='int32'
                    if 'units' in G[coord].attrs:
                        # apply units as encoding for time
                        dv_encoding[coord]['units'] = G[coord].attrs['units']
                        # delete from the attributes list
                        #del G[coord].attrs['units']

                elif coord == 'time_step':
                    dv_encoding[coord]['dtype'] ='int32'

#    dout.to_netcdf('ECCO_pre-SWOT_test_grid.nc',encoding=dv_encoding) 

    import maxnmin
    for nam in dout.keys():
        if nam in maxnmin.maxmin.keys():
            dout[nam].attrs['valid_min']=float(maxnmin.maxmin[nam][0])
            dout[nam].attrs['valid_max']=float(maxnmin.maxmin[nam][1])
    
    return dout


def meta_var(fn):
    import maxnmin

    a={}
    f=open(fn) 
    ff=json.load(f)

    for fff in ff:
        nam=fff['name']
        del fff['name']
        try:
            del fff['grid_dimension']
        except:
            pass

        try:
            del fff['GCMD_keywords']
        except:
            pass
        if nam in maxnmin.maxmin.keys():
            fff['valid_min']=float(maxnmin.maxmin[nam][0])
            fff['valid_max']=float(maxnmin.maxmin[nam][1])
        a[nam]=fff
    del f,ff,fff
    
    return a
    
def meta_global(fn):
    a={}
    f=open(fn) 
    ff=json.load(f)
    for fff in ff:
        nam=fff['name']
        del fff['name']
        try:
            del fff['GCMD_keywords']
        except:
            pass
        a[nam]=fff['value']
    del f,ff,fff
    
    return a


def get_region(region_name,comm):
    import pandas as pd
    import sys

    size=comm.Get_size()
    rank=comm.Get_rank()

    if rank==0:
        os.makedirs('netcdf/%s'%region_name,exist_ok=True)

    a=meta_var('ECCOv4r4_variable_metadata.json')
    #print(a.keys())

    days=pd.date_range('2011-09-13 00:00:00','2012-11-14',freq='1D')

    #the following is only for calswot2 region, for which earlier days are not available. 
    #days=pd.date_range('2011-11-13 00:00:00','2012-11-14',freq='1D')

    fn=glob(pth+'%s/U/*U*'%region_name)[0]

    tt,nx,ny,nz,i0,j0,k0=parse(fn)
    grid=read_grid(pth+'/'+region_name,nx,ny,nz)
    

    app3=fn.split('/')[-1][12:]
    app2=fn.split('/')[-1][12:-2]+'1'

    vars1=['Theta','Salt']
    vars2=['U','V','W']
    vars3=['Eta','KPPhbl','PhiBot','oceFWflx','oceQnet','oceQsw','oceSflux','oceTAUX','oceTAUY']

    nn=429//size

    ii0=rank*nn
    if rank==size-1:
        ii1=429
    else:
        ii1=rank*nn+nn

    #for testing and manual indexing
    ii0,ii1=rank,rank+1

    for day in days[ii0:ii1]:
        dout=read_grid(pth+'/'+region_name,nx,ny,nz)
        #print(day)
        #time_axis=pd.date_range(day,freq='1h',periods=24)
        
        time_axis=np.arange(24)+(day-np.datetime64('2011-01-01 00:00:00') )/np.timedelta64(1,'h')

        ############ 2D fields
        for varn in vars3:
            dd=np.zeros((24,ny,nx))
            for hour in range(24):
                #print(varn,time_axis[hour])
                index=(day-np.datetime64('2011-09-13 00:00:00'))/np.timedelta64(1,'s')/3600*144+10368 + hour*144
                if 'TAUY' in varn:
                    try:
                        fn=pth+'%s/%s/%010i_%s%s'%(region_name,varn,index,varn,'_%i.%i.%i_%i.%i.%i'%(i0,j0-1,k0,nx,ny,1) )
                        dd[hour,...]=np.fromfile(fn,'>f4').reshape(ny,nx)
                    except:
                        fn=pth+'%s/%s/%010i_%s%s'%(region_name,varn,index,varn,'_%i.%i.%i_%i.%i.%i'%(i0,j0,k0,nx,ny,1) )
                        dd[hour,...]=np.fromfile(fn,'>f4').reshape(ny,nx)
                else:
                    fn=pth+'%s/%s/%010i_%s%s'%(region_name,varn,index,varn,'_%i.%i.%i_%i.%i.%i'%(i0,j0,k0,nx,ny,1) )
                    dd[hour,...]=np.fromfile(fn,'>f4').reshape(ny,nx)


            dd=np.where(dd==0,netcdf_fill_value,dd)

            if 'TAUX' in varn:
                dout[varn]=xr.DataArray(dd,dims=('time','j','i_g'),coords={'time':time_axis,'j':grid['j'],'i_g':grid['i_g']},attrs=a[varn])
            elif 'TAUY' in varn:
                dout[varn]=xr.DataArray(dd,dims=('time','j_g','i'),coords={'time':time_axis,'j':grid['j_g'],'i':grid['i']},attrs=a[varn])
            else:
                dout[varn]=xr.DataArray(dd,dims=('time','j','i'),coords={'time':time_axis,'j':grid['j'],'i':grid['i']},attrs=a[varn])


       
############################################################
######## T/S 3D ######################################
        for varn in vars1:
            dd=np.zeros((24,nz,ny,nx))
            for hour in range(24):
                #print(varn,time_axis[hour])
                index=(day-np.datetime64('2011-09-13 00:00:00'))/np.timedelta64(1,'s')/3600*144+10368 + hour*144
        
                fn=pth+'%s/%s/%010i_%s%s'%(region_name,varn,index,varn,'_%i.%i.%i_%i.%i.%i'%(i0,j0,k0,nx,ny,nz) )

                dd[hour,...]=np.fromfile(fn,'>f4').reshape(nz,ny,nx)
            dd=np.where(dd==0,netcdf_fill_value,dd)
            dout[varn]=xr.DataArray(dd,dims=('time','k','j','i'),coords={'time':time_axis,'k':grid['k'],'j':grid['j'],'i':grid['i']},attrs=a[varn])

        #for nn in ['XC','YC','Z','Z_bnds','XC_bnds','YC_bnds']:
        #    dout[nn]=grid[nn]

        #dout=xr.Dataset(dout)
 
        #global_att=meta_global('ECCOv4r4_global_metadata_for_TS.json')
        #dout.attrs=global_att

        #dv_encoding = {}
       # for dv in dout.data_vars:
       #     dv_encoding[dv] =  {'zlib':True, \
       #                             'complevel':5,\
       #                             'shuffle':True,\
       #                             '_FillValue':netcdf_fill_value}
       # G=dout
       # for coord in dout.coords:
       #     print(coord)
       #     dv_encoding[coord]={'_FillValue':None, 'dtype':'float32'}
       #     if (G[coord].values.dtype == np.int32) or \
       #            (G[coord].values.dtype == np.int64) :
       #         dv_encoding[coord]['dtype'] ='int32'
#
#                if coord == 'time' or coord == 'time_bnds':
#                    dv_encoding[coord]['dtype'] ='int32'
#                    if 'units' in G[coord].attrs:
#                        # apply units as encoding for time
#                        dv_encoding[coord]['units'] = G[coord].attrs['units']
#                        # delete from the attributes list
#                        #del G[coord].attrs['units']

#                elif coord == 'time_step':
#                    dv_encoding[coord]['dtype'] ='int32'

        #dout.to_netcdf('ECCO_pre-SWOT_test_3D.nc',encoding=dv_encoding)

       
 ############################################################
######## U/V/W 3D ######################################
        for varn in vars2:
        
            dd=np.zeros((24,nz,ny,nx))
            for hour in range(24):
                #print(varn,time_axis[hour])
                index=(day-np.datetime64('2011-09-13 00:00:00'))/np.timedelta64(1,'s')/3600*144+10368 + hour*144
        
                if varn=='V':
                    try:
                        fn=pth+'%s/%s/%010i_%s%s'%(region_name,varn,index,varn,'_%i.%i.%i_%i.%i.%i'%(i0,j0-1,k0,nx,ny,nz) )
                        dd[hour,...]=np.fromfile(fn,'>f4').reshape(nz,ny,nx)
                    except:
                        fn=pth+'%s/%s/%010i_%s%s'%(region_name,varn,index,varn,'_%i.%i.%i_%i.%i.%i'%(i0,j0,k0,nx,ny,nz) )
                        dd[hour,...]=np.fromfile(fn,'>f4').reshape(nz,ny,nx)

                else:
                    fn=pth+'%s/%s/%010i_%s%s'%(region_name,varn,index,varn,'_%i.%i.%i_%i.%i.%i'%(i0,j0,k0,nx,ny,nz) )

                    dd[hour,...]=np.fromfile(fn,'>f4').reshape(nz,ny,nx)
            dd=np.where(dd==0,netcdf_fill_value,dd)

            if 'U'==varn:
                dout[varn]=xr.DataArray(dd.astype('f4'),dims=('time','k','j','i_g'),coords={'time':time_axis,'k':grid['k'],'j':grid['j'],'i_g':grid['i_g']},attrs=a[varn])
            if 'V'==varn:
                dout[varn]=xr.DataArray(dd,dims=('time','k','j_g','i'),coords={'time':time_axis,'k':grid['k'],'j_g':grid['j_g'],'i':grid['i']},attrs=a[varn])
            if 'W'==varn:
                dout[varn]=xr.DataArray(dd,dims=('time','k_l','j','i'),coords={'time':time_axis,'k_l':grid['k_l'],'j':grid['j'],'i':grid['i']},attrs=a[varn])


        #for nn in ['XC','XG','YC','YG','Z','Z_bnds','XC_bnds','YC_bnds']:
        #    dout[nn]=grid[nn]

        dout=xr.Dataset(dout)
 
        global_att=meta_global('ECCOv4r4_global_metadata_for_TS.json')
        global_att['geospatial_lat_max']=grid['YC'].max().data
        global_att['geospatial_lat_min']=grid['YC'].min().data
        global_att['geospatial_lon_max']=grid['XC'].max().data
        global_att['geospatial_lon_min']=grid['XC'].min().data
        global_att['geospatial_lon_resolution']='variable'
        global_att['geospatial_lat_resolution']='variable'
 
        global_att['time_coverage_end']=str(day+np.timedelta64(23,'h'))
        global_att['time_coverage_start']=str(day)
        dout.attrs=global_att


        dv_encoding = {}
        for dv in dout.data_vars:
            dv_encoding[dv] =  {'zlib':True, \
                                    'complevel':5,\
                                    'shuffle':True,\
                                    '_FillValue':netcdf_fill_value}
        G=dout
        for coord in dout.coords:
            #print(coord)
            dv_encoding[coord]={'_FillValue':None, 'dtype':'float32'}
            if (G[coord].values.dtype == np.int32) or \
                   (G[coord].values.dtype == np.int64) :
                dv_encoding[coord]['dtype'] ='int32'

                if coord == 'time' or coord == 'time_bnds':
                    dv_encoding[coord]['dtype'] ='int32'
                    if 'units' in G[coord].attrs:
                        # apply units as encoding for time
                        dv_encoding[coord]['units'] = G[coord].attrs['units']
                        # delete from the attributes list
                        #del G[coord].attrs['units']

                elif coord == 'time_step':
                    dv_encoding[coord]['dtype'] ='int32'

        for varn in dout.data_vars:
            if 'bounds' in dout[varn].attrs:
                if dout[varn].attrs['bounds']==" ":
                    dout[varn].attrs['bounds']=""

        for varn in dout.coords:
            if 'bounds' in dout.coords[varn].attrs:
                if dout.coords[varn].attrs['bounds']==" ":
                    dout.coords[varn].attrs['bounds']=""

        dout['time'].attrs['Longname']='center time of snapshots'
        dout['time'].attrs['axis']='T'
        dout['time'].attrs['_FillValue']=9.96920996838687e+36
        dout['time'].attrs['coverage_content_type']='coordinate'
        dout['time'].attrs['standard_name']='time'
        dout['time'].attrs['units']='hours since 2011-01-01 00:00:00'
      
       
        dout.attrs['platform']="MITgcm"
        dout.attrs['title']+=' '+names[region_name]
        short_name="MITgcm_LLC4320_Pre-SWOT_JPL_L4_%s_v1.0"%region_name 
        dout.attrs['metadata_link']='http://podaac.jpl.nasa.gov/ws/metadata/dataset/?format=iso&shortName='+short_name

        dout.attrs['id']=short_name

        for key in dout.keys():
            try:
                del dout[key].attrs['grid_dimension']
            except:
                pass


        dout['nb'].attrs={'long_name':'grid index for coordinate bounds',
                          'valid_min':np.int32(0),
                          'valid_max':np.int32(1),
                          'coverage_content_type':"coordinate"}
        #dout['nb'].valid_min.dtype='int32'
        #dout['nb'].valid_max.dtype='int32'
        

        aa=day.strftime('%Y%m%d')
        dout.to_netcdf('netcdf/%s/LLC4320_pre-SWOT_%s_%s.nc'%(region_name,region_name,aa),encoding=dv_encoding)
        if rank==0:
            print('netcdf/%s/LLC4320_pre-SWOT_%s_%s.nc'%(region_name,region_name,aa))
        del dout

if __name__=='__main__':
    from mpi4py import MPI
    import sys
    comm=MPI.COMM_WORLD

    region_names=['ACC_SMST','ROAM_MIZ','RockallTrough','WesternMed']
    region_names=['GotlandBasin','Boknis','NewCaledonia','NWAustralia','CalSWOT2','SOFS','Yongala','WestAtlantic','ACC_SMST']
    #region_names=['GotlandBasin','Boknis','NewCaledonia','NWAustralia','SOFS','Yongala','WestAtlantic','ACC_SMST']
    names={'ACC_SMST':"Southern Ocean",
           'BassStrait':"Bass Strait",
           'CapeBasin':"Cape Basin",
           'LabradorSea':"Labrador Sea",
           'MarmaraSea':"Marmara Sea",
           'NWPacific':"Northwest Pacific",
           'NewCaledonia':"New Caledonia", 
           'ROAM_MIZ':"Northeast Weddell Sea",
           'RockallTrough':"Rockall Trough",
           'CalSWOT2':"California Current System",
           'SOFS':"Southern Ocean Flux Station",
           'Yongala':"Yongala National Reference Mooring",
           'WestAtlantic':"Gulf Stream",
           'GotlandBasin':"CONWEST_DYCO site 1",
           'Boknis':"CONWEST_DYCO site 2",
           'NWAustralia':"Northwest Australian Shelf",
           'WesternMed':"MEDITERRANEAN SEA"}
    i=int(sys.argv[1])
    for region_name in region_names[i:i+1]:
        get_region(region_name,comm)
        comm.Barrier()
