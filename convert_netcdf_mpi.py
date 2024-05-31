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
    """
    Parse the filename to extract metadata.
    
    This function parses the given filename to extract the time, grid dimensions, and initial indices.
    The filename is expected to follow a specific format where these pieces of information are embedded.

    Parameters:
    fn (str): Filename to be parsed.

    Returns:
    tuple: Parsed metadata including:
        - tt (str): Extracted time string from the filename.
        - nx (int): Number of grid points in the x-direction.
        - ny (int): Number of grid points in the y-direction.
        - nz (int): Number of grid points in the z-direction.
        - i0 (int): Initial index in the x-direction.
        - j0 (int): Initial index in the y-direction.
        - k0 (int): Initial index in the z-direction.
    """
    nn = fn.split('_')[-1].split('.')
    nx, ny, nz = int(nn[-3]), int(nn[-2]), int(nn[-1])
    nn = fn.split('_')[-2].split('.')
    i0, j0, k0 = int(nn[-3]), int(nn[-2]), int(nn[-1])

    tt = fn.split('/')[-1][:10]
    return tt, nx, ny, nz, i0, j0, k0


def read_grid(ph, nx, ny, nz):
    """
    Read coordinate metadata for the model grid.

    This function reads various grid data files, constructs the grid coordinates and metadata,
    and returns an xarray.Dataset containing this information.

    Parameters:
    ph (str): Path to the grid data.
    nx (int): Number of grid points in the x-direction.
    ny (int): Number of grid points in the y-direction.
    nz (int): Number of grid points in the z-direction.

    Returns:
    xarray.Dataset: Dataset containing grid metadata and coordinates.
    """
    rc = np.fromfile('/home6/dmenemen/llc_4320/grid/RC.data', '>f4')
    rf = np.fromfile('/home6/dmenemen/llc_4320/grid/RF.data', '>f4')

    a = {}
    with open('ECCOv4r4_geometry_metadata_for_native_datasets.json') as f:
        ff = json.load(f)

    for fff in ff:
        cc = fff.copy()
        del cc['name']
        a[fff['name']] = cc

    fns = glob(ph + '/grid/*x*')
    dd = {}
    for fn in fns:
        var = fn.split('/')[-1].split('_')[0]
        if 'hFac' in var:
            dd[var] = np.fromfile(fn, '>f4').reshape(nz, ny, nx)
        else:
            dd[var] = np.fromfile(fn, '>f4').reshape(ny, nx)

    i = i_g = np.arange(nx, dtype='int16')
    j = j_g = np.arange(ny, dtype='int16')
    k = k_u = k_l = np.arange(nz)
    k_p1 = np.arange(nz + 1)

    dout = {}
    for na in ['i', 'i_g']:
        dout[na] = xr.DataArray(i, dims=(na), coords={na: i}, attrs=a[na])
    for na in ['j', 'j_g']:
        dout[na] = xr.DataArray(j, dims=(na), coords={na: j}, attrs=a[na])
    for na in ['k', 'k_u', 'k_l']:
        dout[na] = xr.DataArray(k, dims=(na), coords={na: k}, attrs=a[na])
    dout['k_p1'] = xr.DataArray(k_p1, dims=('k_p1'), coords={'k_p1': k_p1}, attrs=a[na])

    for na in ['XC', 'YC', 'DXV', 'DYU', 'Depth', 'AngleSN', 'AngleCS']:
        try:
            aa = a[na]
            aa.pop('GCMD_keywords', None)
            dout[na] = xr.DataArray(dd[na], dims=('j', 'i'), coords={'j': dout['j'], 'i': dout['i']}, attrs=a[na])
        except KeyError:
            pass

    for na in ['DXC', 'DYG']:
        dout[na] = xr.DataArray(dd[na], dims=('j', 'i_g'), coords={'j': dout['j'], 'i_g': dout['i_g']}, attrs=a[na])

    for na in ['DYC', 'DXG']:
        dout[na] = xr.DataArray(dd[na], dims=('j_g', 'i'), coords={'j_g': dout['j_g'], 'i': dout['i']}, attrs=a[na])

    for na in ['XG', 'YG', 'RAZ']:
        dout[na] = xr.DataArray(dd[na], dims=('j_g', 'i_g'), coords={'j_g': dout['j_g'], 'i_g': dout['i_g']}, attrs=a[na])

    if 'hFacC' in dd:
        dout['hFacC'] = xr.DataArray(dd['hFacC'], dims=('k', 'j', 'i'), coords={'k': dout['k'], 'j': dout['j'], 'i': dout['i']}, attrs=a['hFacC'])

    if 'hFacW' in dd:
        dout['hFacW'] = xr.DataArray(dd['hFacW'], dims=('k', 'j', 'i_g'), coords={'k': dout['k'], 'j': dout['j'], 'i_g': dout['i_g']}, attrs=a['hFacW'])

    if 'hFacS' in dd:
        dout['hFacS'] = xr.DataArray(dd['hFacS'], dims=('k', 'j_g', 'i'), coords={'k': dout['k'], 'j_g': dout['j_g'], 'i': dout['i']}, attrs=a['hFacS'])

    c = dd['XC']
    dc = (c[:, 1:] - c[:, :-1]) / 2.0
    dc = np.c_[dc[:, 0:1], dc]
    cbnd = np.zeros((ny, nx, 2))
    cbnd[:, :, 0] = c - dc
    cbnd[:, :, 1] = c + dc
    dout['XC_bnds'] = xr.DataArray(cbnd, dims=('j', 'i', 'nb'), coords={'j': dout['j'], 'i': dout['i'], 'nb': [0, 1]}, attrs=a['XC_bnds'])

    c = dd['YC']
    dc = (c[1:, :] - c[:-1, :]) / 2.0
    dc = np.r_[dc[0:1, :], dc]
    cbnd = np.zeros((ny, nx, 2))
    cbnd[:, :, 0] = c - dc
    cbnd[:, :, 1] = c + dc
    dout['YC_bnds'] = xr.DataArray(cbnd, dims=('j', 'i', 'nb'), coords={'j': dout['j'], 'i': dout['i'], 'nb': [0, 1]}, attrs=a['YC_bnds'])

    dout['Z'] = xr.DataArray(rc[:nz], dims=('k'), coords={'k': dout['k']}, attrs=a['Z'])
    dout['Zp1'] = xr.DataArray(rf[:nz + 1], dims=('k_p1'), coords={'k_p1': dout['k_p1']}, attrs=a['Zp1'])
    dout['Zu'] = xr.DataArray(rf[1:nz + 1], dims=('k_u'), coords={'k_u': dout['k_u']}, attrs=a['Zu'])
    dout['Zl'] = xr.DataArray(rf[:nz], dims=('k_l'), coords={'k_l': dout['k_l']}, attrs=a['Zl'])
    dout['Z_bnds'] = xr.DataArray(np.c_[rf[:nz].ravel(), rf[1:nz + 1].ravel()], dims=('k', 'nb'), coords={'k': dout['k'], 'nb': [0, 1]}, attrs=a['Z_bnds'])

    dout = xr.Dataset(dout)
    G = dout
    dv_encoding = {}
    if True:
        for coord in dout.coords:
            dv_encoding[coord] = {'_FillValue': None, 'dtype': 'float32'}
            if (G[coord].values.dtype == np.int32) or (G[coord].values.dtype == np.int64):
                dv_encoding[coord]['dtype'] = 'int32'

                if coord == 'time' or coord == 'time_bnds':
                    dv_encoding[coord]['dtype'] = 'int32'
                    if 'units' in G[coord].attrs:
                        dv_encoding[coord]['units'] = G[coord].attrs['units']

                elif coord == 'time_step':
                    dv_encoding[coord]['dtype'] = 'int32'

    import maxnmin
    for nam in dout.keys():
        if nam in maxnmin.maxmin.keys():
            dout[nam].attrs['valid_min'] = float(maxnmin.maxmin[nam][0])
            dout[nam].attrs['valid_max'] = float(maxnmin.maxmin[nam][1])

    return dout


def meta_var(fn):
    """
    Load variable metadata from a JSON file and enhance it with additional information.
    
    This function reads a JSON file containing metadata for various variables, removes
    unnecessary fields, and adds valid minimum and maximum values if available.

    Parameters:
    fn (str): Path to the JSON file containing variable metadata.

    Returns:
    dict: Dictionary containing enhanced variable metadata, with variable names as keys
          and their metadata as values.
    """
    import maxnmin

    a = {}
    with open(fn) as f:
        ff = json.load(f)

    for fff in ff:
        nam = fff['name']
        del fff['name']
        
        # Remove 'grid_dimension' if it exists
        fff.pop('grid_dimension', None)
        
        # Remove 'GCMD_keywords' if it exists
        fff.pop('GCMD_keywords', None)

        # Add valid_min and valid_max if they exist in maxnmin
        if nam in maxnmin.maxmin.keys():
            fff['valid_min'] = float(maxnmin.maxmin[nam][0])
            fff['valid_max'] = float(maxnmin.maxmin[nam][1])
        
        a[nam] = fff
    
    return a
def meta_global(fn):
    """
    Load global metadata from a JSON file and extract relevant values.
    
    This function reads a JSON file containing global metadata, removes
    unnecessary fields, and extracts the values associated with each metadata
    entry.

    Parameters:
    fn (str): Path to the JSON file containing global metadata.

    Returns:
    dict: Dictionary containing global metadata values, with metadata names as keys
          and their corresponding values as dictionary values.
    """
    a = {}
    with open(fn) as f:
        ff = json.load(f)
    
    for fff in ff:
        nam = fff['name']
        del fff['name']
        
        # Remove 'GCMD_keywords' if it exists
        fff.pop('GCMD_keywords', None)
        
        a[nam] = fff['value']
    
    return a
def get_region(region_name, comm, output_dir='netcdf/'):
    """
    Process and generate NetCDF files for a specified region using MPI for parallel computation.
    
    This function divides the processing task among available MPI ranks, reads grid and variable
    metadata, processes daily data files, and generates NetCDF files for the specified region.

    Parameters:
    region_name (str): Name of the region to process.
    comm (MPI.Comm): MPI communicator for parallel processing.

    Returns:
    None
    """
    import pandas as pd
    import sys

    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        os.makedirs(f'{output_dir}/{region_name}', exist_ok=True)

    a = meta_var('ECCOv4r4_variable_metadata.json')

    days = pd.date_range('2011-09-13 00:00:00', '2012-11-14', freq='1D')

    fn = glob(pth + f'{region_name}/U/*U*')[0] 
    
    tt, nx, ny, nz, i0, j0, k0 = parse(fn)
    grid = read_grid(pth + '/' + region_name, nx, ny, nz)

    app3 = fn.split('/')[-1][12:]
    app2 = fn.split('/')[-1][12:-2] + '1'

    vars1 = ['Theta', 'Salt']
    vars2 = ['U', 'V', 'W']
    vars3 = ['Eta', 'KPPhbl', 'PhiBot', 'oceFWflx', 'oceQnet', 'oceQsw', 'oceSflux', 'oceTAUX', 'oceTAUY']

    nn = 429 // size

    ii0 = rank * nn
    ii1 = 429 if rank == size - 1 else rank * nn + nn

    # For testing and manual indexing
    ii0, ii1 = rank + 215, rank + 1 + 215

    for day in days[ii0:ii1]:
        dout = read_grid(pth + '/' + region_name, nx, ny, nz)

        time_axis = np.arange(24) + (day - np.datetime64('2011-01-01 00:00:00')) / np.timedelta64(1, 'h')

        # Process 2D fields
        for varn in vars3:
            dd = np.zeros((24, ny, nx))
            for hour in range(24):
                index = (day - np.datetime64('2011-09-13 00:00:00')) / np.timedelta64(1, 's') / 3600 * 144 + 10368 + hour * 144
                try:
                    if 'TAUY' in varn:
                        fn = pth + '%s/%s/%010i_%s%s' % (region_name, varn, index, varn, '_%i.%i.%i_%i.%i.%i' % (i0, j0 - 1, k0, nx, ny, 1))
                    else:
                        fn = pth + '%s/%s/%010i_%s%s' % (region_name, varn, index, varn, '_%i.%i.%i_%i.%i.%i' % (i0, j0, k0, nx, ny, 1))
                    dd[hour, ...] = np.fromfile(fn, '>f4').reshape(ny, nx)
                except:
                    fn = pth + '%s/%s/%010i_%s%s' % (region_name, varn, index, varn, '_%i.%i.%i_%i.%i.%i' % (i0, j0, k0, nx, ny, 1))
                    dd[hour, ...] = np.fromfile(fn, '>f4').reshape(ny, nx)

            dd = np.where(dd == 0, netcdf_fill_value, dd)

            if 'TAUX' in varn:
                dout[varn] = xr.DataArray(dd, dims=('time', 'j', 'i_g'), coords={'time': time_axis, 'j': grid['j'], 'i_g': grid['i_g']}, attrs=a[varn])
            elif 'TAUY' in varn:
                dout[varn] = xr.DataArray(dd, dims=('time', 'j_g', 'i'), coords={'time': time_axis, 'j': grid['j_g'], 'i': grid['i']}, attrs=a[varn])
            else:
                dout[varn] = xr.DataArray(dd, dims=('time', 'j', 'i'), coords={'time': time_axis, 'j': grid['j'], 'i': grid['i']}, attrs=a[varn])

        # Process T/S 3D fields
        for varn in vars1:
            dd = np.zeros((24, nz, ny, nx))
            for hour in range(24):
                index = (day - np.datetime64('2011-09-13 00:00:00')) / np.timedelta64(1, 's') / 3600 * 144 + 10368 + hour * 144
                fn = pth + '%s/%s/%010i_%s%s' % (region_name, varn, index, varn, '_%i.%i.%i_%i.%i.%i' % (i0, j0, k0, nx, ny, nz))
                dd[hour, ...] = np.fromfile(fn, '>f4').reshape(nz, ny, nx)

            dd = np.where(dd == 0, netcdf_fill_value, dd)
            dout[varn] = xr.DataArray(dd, dims=('time', 'k', 'j', 'i'), coords={'time': time_axis, 'k': grid['k'], 'j': grid['j'], 'i': grid['i']}, attrs=a[varn])

        # Process U/V/W 3D fields
        for varn in vars2:
            dd = np.zeros((24, nz, ny, nx))
            for hour in range(24):
                index = (day - np.datetime64('2011-09-13 00:00:00')) / np.timedelta64(1, 's') / 3600 * 144 + 10368 + hour * 144
                try:
                    if varn == 'V':
                        fn = pth + '%s/%s/%010i_%s%s' % (region_name, varn, index, varn, '_%i.%i.%i_%i.%i.%i' % (i0, j0 - 1, k0, nx, ny, nz))
                    else:
                        fn = pth + '%s/%s/%010i_%s%s' % (region_name, varn, index, varn, '_%i.%i.%i_%i.%i.%i' % (i0, j0, k0, nx, ny, nz))
                    dd[hour, ...] = np.fromfile(fn, '>f4').reshape(nz, ny, nx)
                except:
                    fn = pth + '%s/%s/%010i_%s%s' % (region_name, varn, index, varn, '_%i.%i.%i_%i.%i.%i' % (i0, j0, k0, nx, ny, nz))
                    dd[hour, ...] = np.fromfile(fn, '>f4').reshape(nz, ny, nx)

            dd = np.where(dd == 0, netcdf_fill_value, dd)

            if varn == 'U':
                dout[varn] = xr.DataArray(dd.astype('f4'), dims=('time', 'k', 'j', 'i_g'), coords={'time': time_axis, 'k': grid['k'], 'j': grid['j'], 'i_g': grid['i_g']}, attrs=a[varn])
            elif varn == 'V':
                dout[varn] = xr.DataArray(dd, dims=('time', 'k', 'j_g', 'i'), coords={'time': time_axis, 'k': grid['k'], 'j_g': grid['j_g'], 'i': grid['i']}, attrs=a[varn])
            elif varn == 'W':
                dout[varn] = xr.DataArray(dd, dims=('time', 'k_l', 'j', 'i'), coords={'time': time_axis, 'k_l': grid['k_l'], 'j': grid['j'], 'i': grid['i']}, attrs=a[varn])

        dout = xr.Dataset(dout)

        global_att = meta_global('ECCOv4r4_global_metadata_for_TS.json')
        global_att['geospatial_lat_max'] = grid['YC'].max().data
        global_att['geospatial_lat_min'] = grid['YC'].min().data
        global_att['geospatial_lon_max'] = grid['XC'].max().data
        global_att['geospatial_lon_min'] = grid['XC'].min().data
        global_att['geospatial_lon_resolution'] = 'variable'
        global_att['geospatial_lat_resolution'] = 'variable'

        global_att['time_coverage_end'] = str(day + np.timedelta64(23, 'h'))
        global_att['time_coverage_start'] = str(day)
        dout.attrs = global_att

        dv_encoding = {}
        for dv in dout.data_vars:
            dv_encoding[dv] = {'zlib': True, 'complevel': 5, 'shuffle': True, '_FillValue': netcdf_fill_value}
        G = dout
        for coord in dout.coords:
            dv_encoding[coord] = {'_FillValue': None, 'dtype': 'float32'}
            if (G[coord].values.dtype == np.int32) or (G[coord].values.dtype == np.int64):
                dv_encoding[coord]['dtype'] = 'int32'

                if coord == 'time' or coord == 'time_bnds':
                    dv_encoding[coord]['dtype'] = 'int32'
                    if 'units' in G[coord].attrs:
                        dv_encoding[coord]['units'] = G[coord].attrs['units']

                elif coord == 'time_step':
                    dv_encoding[coord]['dtype'] = 'int32'

        for varn in dout.data_vars:
            if 'bounds' in dout[varn].attrs:
                if dout[varn].attrs['bounds'] == " ":
                    dout[varn].attrs['bounds'] = ""

        for varn in dout.coords:
            if 'bounds' in dout.coords[varn].attrs:
                if dout.coords[varn].attrs['bounds'] == " ":
                    dout.coords[varn].attrs['bounds'] = ""

        dout['time'].attrs['Longname'] = 'center time of snapshots'
        dout['time'].attrs['axis'] = 'T'
        dout['time'].attrs['_FillValue'] = 9.96920996838687e+36
        dout['time'].attrs['coverage_content_type'] = 'coordinate'
        dout['time'].attrs['standard_name'] = 'time'
        dout['time'].attrs['units'] = 'hours since 2011-01-01 00:00:00'

        dout.attrs['platform'] = "MITgcm"
        dout.attrs['title'] += ' ' + names[region_name]
        short_name = "MITgcm_LLC4320_Pre-SWOT_JPL_L4_%s_v1.0" % region_name
        dout.attrs['metadata_link'] = 'http://podaac.jpl.nasa.gov/ws/metadata/dataset/?format=iso&shortName=' + short_name
        dout.attrs['id'] = short_name

        for key in dout.keys():
            try:
                del dout[key].attrs['grid_dimension']
            except:
                pass

        dout['nb'].attrs = {'long_name': 'grid index for coordinate bounds', 'valid_min': np.int32(0), 'valid_max': np.int32(1), 'coverage_content_type': "coordinate"}

        aa = day.strftime('%Y%m%d')
        dout.to_netcdf('netcdf/%s/LLC4320_pre-SWOT_%s_%s.nc' % (region_name, region_name, aa), encoding=dv_encoding)
        if rank == 0:
            print('netcdf/%s/LLC4320_pre-SWOT_%s_%s.nc' % (region_name, region_name, aa))
        del dout

if __name__ == '__main__':
    """
    Main function to initiate the processing of specified regions using MPI for parallel computation.

    This function sets up the MPI communicator, defines the region names and their metadata, and 
    calls the `get_region` function for the specified region based on the MPI rank.
    """
    from mpi4py import MPI
    import sys

    comm = MPI.COMM_WORLD

    region_names = ['GotlandBasin', 'Boknis', 'NewCaledonia', 'NWAustralia', 'CalSWOT2', 'SOFS', 'Yongala', 'WestAtlantic', 'ACC_SMST']

    names = {
        'ACC_SMST': "Southern Ocean",
        'BassStrait': "Bass Strait",
        'CapeBasin': "Cape Basin",
        'LabradorSea': "Labrador Sea",
        'MarmaraSea': "Marmara Sea",
        'NWPacific': "Northwest Pacific",
        'NewCaledonia': "New Caledonia",
        'ROAM_MIZ': "Northeast Weddell Sea",
        'RockallTrough': "Rockall Trough",
        'CalSWOT2': "California Current System",
        'SOFS': "Southern Ocean Flux Station",
        'Yongala': "Yongala National Reference Mooring",
        'WestAtlantic': "Gulf Stream",
        'GotlandBasin': "CONWEST_DYCO site 1",
        'Boknis': "CONWEST_DYCO site 2",
        'NWAustralia': "Northwest Australian Shelf",
        'WesternMed': "MEDITERRANEAN SEA"
    }


    for region_name in region_names:
        get_region(region_name, comm)
        comm.Barrier()
