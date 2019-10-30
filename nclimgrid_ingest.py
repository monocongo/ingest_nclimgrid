import argparse
from datetime import date, datetime
from glob import glob
import logging
import multiprocessing
import os
import re
import subprocess
import sys

import netCDF4
import numpy as np
import pandas as pd

# set up a basic, global logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
def find_files_within_range(directory,
                            initial_year_month,
                            final_year_month):
    """
    This function finds all the nClimGrid ASCII files in a specifed directory
    which have file names corresponding to the specified date range. The nClimGrid
    ASCII files are expected to have the form YYYYMM.pnt, for example 201009.pnt
    for September of 2010.

    :param directory: directory under which the nClimGrid raw ASCII files should
        be found
    :param initial_year_month: string containing six digits representing a year
        and month, ex. 197501 for January of 1975
    :param final_year_month: string containing six digits representing a year
        and month, ex. 197501 for January of 1975
    :return: list of files within the specified date range specified by the
        initial and final (year/month) dates, sorted in chronologically
        ascending order
    :rtype: list of strings
    """
    matching_files = []
    valid_range = range(initial_year_month, final_year_month + 1)
    regex = re.compile('.*(\d{6})\.pnt')

    # get a list of all *.pnt files in the specified directory
    ascii_files = glob('/'.join([directory, '*.pnt']))

    # add all files to our list of matching files if it matches
    # the pattern and the date part of the filename is within range
    for ascii_file in ascii_files:
        if regex.match(ascii_file) and (int(ascii_file[-10:-4]) in valid_range):
            matching_files.append(ascii_file)

    # return the files sorted in ascending order
    return sorted(matching_files)


# ------------------------------------------------------------------------------
# TODO extract this out into a netCDF utilities module, as it's useful (and duplicated) in other codes
def compute_days(initial_year,
                 initial_month,
                 total_months,
                 start_year=1800):
    """
    Computes the "number of days" equivalent of the regular, incremental monthly time steps from an initial year/month.

    :param initial_year: the initial year from which the increments should start
    :param initial_month: the initial month from which the increments should start
    :param total_months: the total number of monthly increments (time steps measured in days) to be computed
    :param start_year: the start year from which the monthly increments (time steps measured in days) to be computed
    :return: an array of time step increments, measured in days since midnight of January 1st of the start_year
    :rtype: ndarray of ints
    """

    # compute an offset from which the day values should begin (assuming that we're using "days since <start_date>" as time units)
    start_date = datetime(start_year, 1, 1)

    # initialize the list of day values we'll build
    days = np.empty(total_months, dtype=int)

    # loop over all time steps (months)
    for i in range(total_months):
        years = int((i + initial_month - 1) / 12)  # the number of years since the initial year
        months = int((i + initial_month - 1) % 12)  # the number of months since January

        # cook up a date for the current time step (month)
        current_date = datetime(initial_year + years, 1 + months, 1)

        # leverage the difference between dates operation available with datetime objects
        days[i] = (current_date - start_date).days

    return days


# ------------------------------------------------------------------------------
def get_coordinate_values(ascii_file):
    """
    This function takes a nCLimGrid ASCII file for a single month and extracts
    a list of lat and lon coordinate values for the regular grid contained therein.

    :param ascii_file:
    :return: lats and lons respectively
    :rtype: two 1-D numpy arrays of floats
    """

    # create a dataframe from the file
    data_frame = \
        pd.read_csv(
            ascii_file,
            delim_whitespace=True,
            names=['lat', 'lon', 'value'],
        )

    # successive lats and lons are separated by 1/24th of a degree (regular grid)
    increment = 1 / 24.

    # determine the minimum lat and lon values
    min_lat = min(data_frame.lat)
    min_lon = min(data_frame.lon)

    # create lat and lon index columns corresponding to the dataframe's lat and
    # lon values the index starts at the minimum, i.e. lat_index[0] == min_lat
    data_frame['lat_index'] = (round((data_frame.lat - min_lat) / increment)).astype(int)
    data_frame['lon_index'] = (round((data_frame.lon - min_lon) / increment)).astype(int)

    # the lat|lon indices start at zero so the number
    # of lats|lons is the length of the index plus one
    lats_count = max(data_frame.lat_index) + 1
    lons_count = max(data_frame.lon_index) + 1

    # since we know the starting lat|lon and the increment between then we can
    # create a full list of lat and lon values based on the number of lats|lons
    lat_values = (np.arange(lats_count) * increment) + min_lat
    lon_values = (np.arange(lons_count) * increment) + min_lon

    return lat_values, lon_values


# ------------------------------------------------------------------------------
def get_variable_attributes(variable_name):
    """
    This function builds a dictionary of variable attributes based on the
    variable name. Four variable names are supported: 'prcp', 'tavg', 'tmin',
    and 'tmax'.

    :param variable_name:
    :return: attributes relevant to the specified variable name
    :rtype: dictionary with string keys corresponding to attribute names
        specified by the NCEI NetCDF template for gridded datasets
        (see https://www.nodc.noaa.gov/data/formats/netcdf/v2.0/grid.cdl)
    """

    # initialize the attributes dictionary with values
    # applicable to all supported variable names
    attributes = {
        'coordinates': 'time lat lon',
        'references': 'GHCN-Monthly Version 3 (Vose et al. 2011), NCEI/NOAA, https://www.ncdc.noaa.gov/ghcnm/v3.php',
    }

    # flesh out additional attributes, based on the variable type
    if variable_name == 'prcp':
        attributes['long_name'] = 'Precipitation, monthly total'
        attributes['standard_name'] = 'precipitation_amount'
        attributes['units'] = 'millimeter'
        attributes['valid_min'] = np.float32(0.0)
        attributes['valid_max'] = np.float32(2000.0)
    else:
        attributes['standard_name'] = 'air_temperature'
        attributes['units'] = 'degree_Celsius'
        attributes['valid_min'] = np.float32(-100.0)
        attributes['valid_max'] = np.float32(100.0)
        if variable_name == 'tavg':
            attributes['long_name'] = 'Temperature, monthly average of daily averages'
        elif variable_name == 'tmax':
            attributes['long_name'] = 'Temperature, monthly average of daily maximums'
        elif variable_name == 'tmin':
            attributes['long_name'] = 'Temperature, monthly average of daily minimums'
        else:
            raise ValueError('The variable_name argument \"{}\" is unsupported.'.format(variable_name))

    return attributes


# ------------------------------------------------------------------------------
def build_netcdf(
        ascii_files,
        netcdf_file,
        var_name: str,
):
    """
    This function builds a NetCDF file for a nClimGrid dataset defined by a set
    of ASCII files. The list of ASCII files is assumed to be sorted in ascending
    order with the first file representing the first time step.

    :param ascii_files: a list nClimGrid ASCII files for a single variable dataset
    :param netcdf_file: the NetCDF file (full path) to create and build out from
        the data contained in the ASCII files
    :param var_name: name of the variable, supported variables are 'prcp',
        'tavg', 'tmin' and 'tmax'
    :rtype: None
    """

    # get the start/end months/years from the initial/final file names in the list,
    # which is assumed to be sorted in ascending order, and file names are assumed
    # to be in the format <YYYYMM>.pnt, eg. "201004.pnt" for April 2010
    initial_year = int(ascii_files[0][-10:-6])
    initial_month = int(ascii_files[0][-6:-4])
    final_year = int(ascii_files[-1][-10:-6])
    final_month = int(ascii_files[-1][-6:-4])

    # use NaN as the fill value for missing data
    output_fill_value = np.float32(np.NaN)

    # determine the lat and lon coordinate values by extracting these
    # from the initial ASCII file in our list (assumes that each ASCII
    # file contains the same lat/lon coordinates)
    lat_values, lon_values = get_coordinate_values(ascii_files[0])
    min_lat = np.float32(min(lat_values))
    max_lat = np.float32(max(lat_values))
    min_lon = np.float32(min(lon_values))
    max_lon = np.float32(max(lon_values))
    lat_units = 'degrees_north'
    lon_units = 'degrees_east'
    total_lats = lat_values.shape[0]
    total_lons = lon_values.shape[0]

    # build the NetCDF
    with netCDF4.Dataset(netcdf_file, 'w') as dataset:
        # create dimensions for a time series, 2-D dataset
        dataset.createDimension('time', None)
        dataset.createDimension('lat', total_lats)
        dataset.createDimension('lon', total_lons)

        # set global group attributes
        dataset.date_created = str(datetime.now())
        dataset.date_modified = str(datetime.now())
        dataset.Conventions = 'CF-1.6, ACDD-1.3'
        dataset.ncei_template_version = 'NCEI_NetCDF_Grid_Template_v2.0'
        dataset.title = 'nClimGrid'
        dataset.naming_authority = 'gov.noaa.ncei'
        dataset.standard_name_vocabulary = 'Standard Name Table v35'
        dataset.institution = 'National Centers for Environmental Information (NCEI), NOAA, Department of Commerce'
        dataset.geospatial_lat_min = min_lat
        dataset.geospatial_lat_max = max_lat
        dataset.geospatial_lon_min = min_lon
        dataset.geospatial_lon_max = max_lon
        dataset.geospatial_lat_units = lat_units
        dataset.geospatial_lon_units = lon_units

        # create a time coordinate variable with one
        # increment per month of the period of record
        start_year = 1800
        total_timesteps = len(ascii_files)
        chunk_sizes = [total_timesteps]
        time_variable = dataset.createVariable('time', 'i4', ('time',), chunksizes=chunk_sizes)
        time_variable[:] = compute_days(initial_year, initial_month, total_timesteps, start_year)
        time_variable.long_name = 'Time, in monthly increments'
        time_variable.standard_name = 'time'
        time_variable.calendar = 'gregorian'
        time_variable.units = 'days since ' + str(start_year) + '-01-01 00:00:00'
        time_variable.axis = 'T'

        # create the lat coordinate variable
        lat_variable = dataset.createVariable('lat', 'f4', ('lat',))
        lat_variable.standard_name = 'latitude'
        lat_variable.long_name = 'Latitude'
        lat_variable.units = lat_units
        lat_variable.axis = 'Y'
        lat_variable.valid_min = min_lat  # - 0.0001
        lat_variable.valid_max = max_lat  # + 0.0001
        lat_variable.units = lat_units
        lat_variable[:] = lat_values

        # create the lon coordinate variable
        lon_variable = dataset.createVariable('lon', 'f4', ('lon',))
        lon_variable.standard_name = 'longitude'
        lon_variable.long_name = 'Longitude'
        lon_variable.units = lon_units
        lon_variable.axis = 'X'
        lon_variable.valid_min = min_lon
        lon_variable.valid_max = max_lon
        lon_variable.units = lon_units
        lon_variable[:] = lon_values

        # create the data variable
        variable = dataset.createVariable(var_name,
                                          'f4',
                                          ('time', 'lat', 'lon'),
                                          fill_value=output_fill_value,
                                          zlib=True,
                                          least_significant_digit=3)

        # set the variable's attributes
        variable.setncatts(get_variable_attributes(var_name))

        # array to contain variable data values
        variable_data = \
            np.full(
                (time_variable.shape[0], total_lats, total_lons),
                output_fill_value,
                dtype=np.float32,
            )

        # loop over the ASCII files in order to build the variable's data array
        for time_index, ascii_file in enumerate(ascii_files):
            # create a dataframe from the file
            data_frame = \
                pd.read_csv(
                    ascii_file,
                    delim_whitespace=True,
                    names=['lat', 'lon', 'value'],
                )

            # successive lats and lons are separated by 1/24th of a degree (regular grid)
            increment = 1 / 24.

            # create lat and lon index columns corresponding to
            # the dataframe's lat and lon values, with the index
            # starting at the minimum, i.e. lat_index[0] == min_lat
            data_frame['lat_index'] = (round((data_frame.lat - min_lat) / increment)).astype(int)
            data_frame['lon_index'] = (round((data_frame.lon - min_lon) / increment)).astype(int)

            # fill the data array with data values, using the lat|lon indices
            variable_data[time_index, data_frame['lat_index'], data_frame['lon_index']] = data_frame['value']

        # assign the data array to the data variable
        variable[:] = variable_data

    logger.info(f'NetCDF file created successfully for variable \"{var_name}\": {netcdf_file}')


# ------------------------------------------------------------------------------
def join_base_with_incremental(base_file,
                               incremental_file,
                               joined_file):
    """
    This function concatenates base and incremental NetCDF files into an
    (assumed to be) contiguous period joined file.

    The original base and incremental files that are joined will be left in place
    (any clean up of these files is assumed to be handled outside of this function).

    :param base_file:
    :param incremental_file:
    :param joined_file:
    """

    logger.info(
        f'Joining the base file {base_file} with the incremental '
        f'file {incremental_file} to create the joined/contiguous '
        f'result file {joined_file}',
    )

    # set up the various parts of the NCO command we'll use,
    # for this, based on whether a Linux or Windows environment
    # TODO replace with pynco equivalent(s)
    ncrcat_command = 'ncrcat'
    if (sys.platform == 'linux') or (sys.platform == 'linux2'):
        nco_home = '/home/james.adams/anaconda3/bin'
    else:  # assume Windows (testing and development)
        nco_home = 'C:/nco'
        windows_suffix = '.exe --no_tmp_fl'
        ncrcat_command += windows_suffix

    # get the proper executable path for the NCO command that'll be used to perform the concatenation operation
    normalized_executable_path = os.path.normpath(nco_home)
    ncrcat = os.path.join(os.sep, normalized_executable_path, ncrcat_command)

    # build and run the command used to concatenate the two files into a single NetCDF
    concatenate_command = ncrcat + ' -O -h -D 0 ' + base_file + ' ' + incremental_file + ' ' + joined_file
    logger.info(f'NCO concatenation command:   {concatenate_command}')
    subprocess.call(concatenate_command, shell=True)


# ------------------------------------------------------------------------------
def ingest_nclimgrid_dataset(parameters):
    """
    This function creates a NetCDF for the full period of record of an nClimGrid
    dataset.

    :param parameters: dictionary containing all required parameters, used
        instead of individual parameters since this function will be called
        from a process pool mapping which requires a single function argument
    """

    try:
        # determine the input directory containing the ASCII files for the variable based on the variable's name
        if parameters['variable_name'] == 'prcp':
            input_directory = '/gcad/nclimdiv/us/por/prcp/pnt'
        elif parameters['variable_name'] == 'tavg':
            input_directory = '/gcad/nclimdiv/us/por/tave/pnt'
        elif parameters['variable_name'] == 'tmax':
            input_directory = '/gcad/nclimdiv/us/por/tmax/pnt'
        elif parameters['variable_name'] == 'tmin':
            input_directory = '/gcad/nclimdiv/us/por/tmin/pnt'
        else:
            raise ValueError('The variable_name argument \"{}\" is unsupported.'.format(parameters['variable_name']))

        logger.info(
            f'Ingesting ASCII files for variable \"{parameters["variable_name"]}\"'
            f' from directory {input_directory}',
        )

        # find the files matching to the base and incremental
        base_ascii_files = \
            find_files_within_range(
                input_directory,
                int(parameters['base_start']),
                int(parameters['base_end']),
            )
        incremental_ascii_files = \
            find_files_within_range(
                input_directory,
                int(parameters['incremental_start']),
                int(parameters['incremental_end']),
            )

        # create the file names for the various output NetCDFs we'll create
        # TODO also get the base filename (eg. currently 'nclimgrid')
        #  as a parameter from command line argument
        base_netcdf_file = parameters['storage_dir'] + '/nclimgrid_' + \
                           parameters['base_start'] + '_' + \
                           parameters['base_end'] + '_' + \
                           parameters['variable_name'] + '.nc'
        incremental_netcdf_file = parameters['storage_dir'] + '/nclimgrid_' + \
                                  parameters['incremental_start'] + '_' + \
                                  parameters['incremental_end'] + '_' + \
                                  parameters['variable_name'] + '.nc'
        destination_netcdf_file = parameters['dest_dir'] + '/nclimgrid_' + \
                                  parameters['variable_name'] + '.nc'

        # make sure the base file is present, if not then we'll recreate it as we would during January's ingest
        if (parameters['current_month'] == 1) or not os.path.isfile(base_netcdf_file):
            logger.info(
                f'Building the {parameters["variable_name"]} NetCDF for base '
                f'period [{parameters["base_start"]} through {parameters["base_end"]}]',
            )

            # create the base period NetCDF
            build_netcdf(
                base_ascii_files,
                base_netcdf_file,
                parameters['variable_name'],
            )

            logger.info(
                f'Completed building the {parameters["variable_name"]} '
                f'NetCDF for base period, result file:  {base_netcdf_file}',
            )

        logger.info(
            f'Building the {parameters["variable_name"]} NetCDF for the '
            f'incremental period [{parameters["incremental_start"]} '
            f'through {parameters["incremental_end"]}]',
        )

        # create the incremental period NetCDF
        build_netcdf(incremental_ascii_files, incremental_netcdf_file, parameters['variable_name'])

        logger.info('Completed building the {0} NetCDF for incremental period, result file:  {1}'.format(
            parameters['variable_name'],
            incremental_netcdf_file))

        # append the incremental file to the base file resulting in the final "destination" file
        join_base_with_incremental(base_netcdf_file, incremental_netcdf_file, destination_netcdf_file)

        logger.info(
            'Completed ingest for variable \"{0}\":  result NetCDF file: {1}'.format(parameters['variable_name'],
                                                                                     destination_netcdf_file))

    except:
        # catch all exceptions, log rudimentary error information
        logger.error('Failed to complete', exc_info=True)
        raise


# ------------------------------------------------------------------------------
if __name__ == '__main__':

    try:
        # parse the command line arguments
        argument_parser = argparse.ArgumentParser()
        argument_parser.add_argument("-d",
                                     "--dest_dir",
                                     required=True,
                                     help="directory where the final, full time series NetCDF should be written")
        argument_parser.add_argument("-s",
                                     "--storage_dir",
                                     required=True,
                                     help="directory where the base and (optionally) incremental NetCDF files should be stored")
        args = argument_parser.parse_args()

        # get the current year and month, used to determine the start/end points for the base and incremental periods
        current_year = date.today().year
        current_month = date.today().month

        """
        If we're running this code during January then we'll ingest the "base" period of record, i.e. from January 1895 
        through December of the year which is two years before the current year, as well as the "incremental" period
        which includes all months after the base period to the previous month. For this initial monthly run of the year
        the base period files will be cached in the specified storage directory for use with subsequent monthly runs,
        eliminating the need to reingest that portion of the dataset, since the base period data is guaranteed to be
        updated only once a year, at the start of each calendar year.
        """

        # the base period is the start of the dataset up to two years previous
        # (for example during 2014 the base period goes through December 2012)
        base_start = '189501'  # the period of record for GHCN-M, from which nClimGrid was derived, begins January 1895
        base_end = str(current_year - 2) + '12'

        #         # DEBUG/TESTING ONLY -- REMOVE before operational deployment of this code
        #         current_month = 1
        #         base_start = '201001'

        # the starting year/month of the incremental period is always January of the previous year
        incremental_start = str(current_year - 1) + '01'

        # set the incremental period's end point
        if current_month == 1:

            # the incremental end year/month is the December of the previous year
            incremental_end = str(current_year - 1) + '12'

        else:

            # set the incremental end year/month to the previous month of the current year
            incremental_end = str(current_year) + str(current_month - 1).zfill(2)

        # create an iterable containing dictionaries of parameters, with one dictionary of parameters per variable,
        # since there will be a separate ingest process per variable, with each process having its own set of parameters
        variables = ['prcp', 'tavg', 'tmin', 'tmax']
        params_list = []
        for variable_name in variables:
            params = {'current_month': current_month,
                      'variable_name': variable_name,
                      'storage_dir': args.storage_dir,
                      'dest_dir': args.dest_dir,
                      'base_start': base_start,
                      'base_end': base_end,
                      'incremental_start': incremental_start,
                      'incremental_end': incremental_end}
            params_list.append(params)

        # create a process pool, mapping the ingest process to the iterable of parameter lists
        pool = multiprocessing.Pool(min(len(variables), multiprocessing.cpu_count()))
        result = pool.map_async(ingest_nclimgrid_dataset, params_list)

        # get the result exception, if any
        pool.close()
        pool.join()

        # set the permissions (recursively) on the destination directory
        logger.info(
            'Changing permissions to 775 on all files under the destination directory: {}'.format(args.dest_dir))
        for root, dirs, files in os.walk(args.dest_dir):
            #             # join the directory name with the root directory path, change the permissions
            #             for dir in dirs:
            #                 os.chmod(os.path.join(root, dir), 0o775)
            # join the file name with the root directory path, change the permissions
            for file in files:
                os.chmod(os.path.join(root, file), 0o775)

    except:
        # catch all exceptions, log rudimentary error information
        logger.error('Failed to complete', exc_info=True)
        raise
