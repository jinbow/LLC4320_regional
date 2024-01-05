import xarray as xr

dd=xr.open_dataset('netcdf/ACC_SMST/LLC4320_pre-SWOT_ACC_SMST_20110920.nc')

for varn in dd.keys():
    d0,d1=dd[varn].min().data,dd[varn].max().data
    print("\'%s\':[%f,%f],"%(varn,d0,d1) )


for varn in dd.coords.keys():
    d0,d1=dd.coords[varn].min().data,dd.coords[varn].max().data
    print("\'%s\':[%f,%f],"%(varn,d0,d1) )

print(dd['time'])
