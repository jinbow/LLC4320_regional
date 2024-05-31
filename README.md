
# LLC4320_regional

A tool to convert the regional LLC4320 data to NetCDF files. Without modification, it only works within the AMES supercomputer.

## Prerequisites

1. Install Miniconda in the home directory.
2. Include necessary modules (this may be different after software updates):
    ```bash
    source miniconda3/bin/activate
    module load pkgsrc/2021Q2 mpi-hpe/mpt
    ```
3. Run from an interactive cluster:
    1. Start an interactive cluster:
        ```bash
        qsub -I -q devel -lselect=$1:ncpus=$2:model=$3,walltime=2:00:00
        ```

## Installation

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/jinbow/LLC4320_regional.git
    ```

2. Install the required Python libraries:
    ```bash
    pip install mpi4py numpy xarray pandas netCDF4
    ```

## Usage

### Running the Script

To run the script, use the following command from the cluster:
```bash
mpiexec -np <number_of_processes> python convert_netcdf_mpi.py
```
Replace `<number_of_processes>` with the number of parallel processes you will use. You may need to change the settings in the main routine of the program.

### Example
```bash
mpiexec -np 4 python convert_netcdf_mpi.py 0
```
This command will run the script with 4 MPI processes to process the first region in the `region_names` list.

### submit large jobs

Use pbs.sh to submit large jobs that make more than 2 hours.

### Output

The output files are saved in the folder: `/nobackup/jwang23/llc4320_stripe/regional.subsets.adac.netcdf`. Change this to your nobackup folder. 

## Region Names and Metadata

The script processes data for the following regions:

- GotlandBasin
- Boknis
- NewCaledonia
- NWAustralia
- CalSWOT2
- SOFS
- Yongala
- WestAtlantic
- ACC_SMST

The metadata for these regions is defined in the `names` dictionary within the script.

The output has been published on podaac: `https://podaac.jpl.nasa.gov/cloud-datasets?search=Pre-SWOT%20Level-4%20Hourly%20MITgcm%20LLC4320`

## Function Descriptions

### `get_region(region_name, comm)`

Processes and generates NetCDF files for a specified region using MPI for parallel computation.

**Parameters:**

- `region_name` (str): Name of the region to process.
- `comm` (MPI.Comm): MPI communicator for parallel processing.

### `meta_var(fn)`

Loads variable metadata from a JSON file and enhances it with additional information.

**Parameters:**

- `fn` (str): Path to the JSON file containing variable metadata.

**Returns:**

- dict: Dictionary containing enhanced variable metadata, with variable names as keys and their metadata as values.

### `meta_global(fn)`

Loads global metadata from a JSON file and extracts relevant values.

**Parameters:**

- `fn` (str): Path to the JSON file containing global metadata.

**Returns:**

- dict: Dictionary containing global metadata values, with metadata names as keys and their corresponding values as dictionary values.

### `read_grid(ph, nx, ny, nz)`

Reads coordinate metadata for the model grid.

**Parameters:**

- `ph` (str): Path to the grid data.
- `nx` (int): Number of grid points in the x-direction.
- `ny` (int): Number of grid points in the y-direction.
- `nz` (int): Number of grid points in the z-direction.

**Returns:**

- `xarray.Dataset`: Dataset containing grid metadata and coordinates.

### `parse(fn)`

Parses the filename to extract metadata.

**Parameters:**

- `fn` (str): Filename to be parsed.

**Returns:**

- tuple: Parsed metadata including:
    - `tt` (str): Extracted time string from the filename.
    - `nx` (int): Number of grid points in the x-direction.
    - `ny` (int): Number of grid points in the y-direction.
    - `nz` (int): Number of grid points in the z-direction.
    - `i0` (int): Initial index in the x-direction.
    - `j0` (int): Initial index in the y-direction.
    - `k0` (int): Initial index in the z-direction.

## Contributing

Feel free to open issues or submit pull requests if you have any suggestions or improvements.

## Acknowledgment

The JSON files for the MITgcm output are from Ian Fenty.

## License

This project is licensed under the MIT License.
