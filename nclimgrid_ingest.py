import argparse
from datetime import datetime
import ftplib
import logging
import multiprocessing
import os
import shutil
import tarfile
import tempfile
from typing import Dict, List

import numpy as np
import pandas as pd
import xarray as xr

# ------------------------------------------------------------------------------
# FTP locations at NCEI
_NCLIMGRID_FTP_URL = "ftp.ncdc.noaa.gov"
_NCLIMGRID_FTP_DIR = "pub/data/climgrid"

# ------------------------------------------------------------------------------
# set up a basic, global logger
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d  %H:%M:%S")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
def find_files_within_range(
        year_month_initial: str,
        year_month_final: str,
) -> List[str]:
    """

    :param year_month_initial:
    :param year_month_final:
    :return:
    """

    # get the list of files under ftp://ftp.ncdc.noaa.gov/pub/data/climgrid/
    # ftp = ftplib.FTP(_NCLIMGRID_FTP_URL)
    with ftplib.FTP(host=_NCLIMGRID_FTP_URL) as ftp:
        ftp.login("anonymous", "ftplib-example")
        ftp.cwd(_NCLIMGRID_FTP_DIR)
        data = []
        ftp.dir(data.append)
    # ftp.quit()

    # get all the file names that fall within the data range
    file_names_normal = []
    file_names_preliminary = []
    for line in data:
        parts = line.split()
        file_name = parts[-1]
        if file_name.startswith("nClimGrid_v1.0_monthly") \
                and file_name.endswith(".tar.gz"):
            year_month = int(file_name[-23:-17])
            if (year_month >= int(year_month_initial)) \
                    and (year_month <= int(year_month_final)):
                file_names_normal.append(file_name)
        elif file_name.startswith("nClimGrid_v1.0-preliminary") \
                and file_name.endswith(".tar.gz"):
            year_month = int(file_name[-23:-17])
            if (year_month >= int(year_month_initial)) \
                    and (year_month <= int(year_month_final)):
                file_names_preliminary.append(file_name)

    # return the list sorted in ascending order
    return sorted(file_names_normal) + sorted(file_names_preliminary)


# ------------------------------------------------------------------------------
# TODO extract this out into a netCDF utilities module, as it's
#  useful (and duplicated) in other codes
def compute_days(initial_year,
                 initial_month,
                 total_months,
                 start_year=1800):
    """
    Computes the "number of days" equivalent of the regular, incremental monthly
    time steps from an initial year/month.

    :param initial_year: the initial year from which the increments should start
    :param initial_month: the initial month from which the increments should start
    :param total_months: the total number of monthly increments (time steps
        measured in days) to be computed
    :param start_year: the start year from which the monthly increments
        (time steps measured in days) to be computed
    :return: an array of time step increments, measured in days since midnight
        of January 1st of the start_year
    :rtype: ndarray of ints
    """

    # compute an offset from which the day values should begin
    # (assuming that we're using "days since <start_date>" as time units)
    start_date = datetime(start_year, 1, 1)

    # initialize the list of day values we'll build
    days = np.empty(total_months, dtype=int)

    # loop over all time steps (months)
    for i in range(total_months):

        # the number of years since the initial year
        years = int((i + initial_month - 1) / 12)
        # the number of months since January
        months = int((i + initial_month - 1) % 12)

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
            names=["lat", "lon", "value"],
        )

    # successive lats and lons are separated by 1/24th of a degree (regular grid)
    increment = 1 / 24.

    # determine the minimum lat and lon values
    min_lat = min(data_frame.lat)
    min_lon = min(data_frame.lon)

    # create lat and lon index columns corresponding to the dataframe's lat and
    # lon values the index starts at the minimum, i.e. lat_index[0] == min_lat
    data_frame["lat_index"] = (round((data_frame.lat - min_lat) / increment)).astype(int)
    data_frame["lon_index"] = (round((data_frame.lon - min_lon) / increment)).astype(int)

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
def get_variable_attributes(var_name):
    """
    This function builds a dictionary of variable attributes based on the
    variable name. Four variable names are supported: 'prcp', 'tave', 'tmin',
    and 'tmax'.

    :param var_name:
    :return: attributes relevant to the specified variable name
    :rtype: dictionary with string keys corresponding to attribute names
        specified by the NCEI NetCDF template for gridded datasets
        (see https://www.nodc.noaa.gov/data/formats/netcdf/v2.0/grid.cdl)
    """

    # initialize the attributes dictionary with values
    # applicable to all supported variable names
    attributes = {
        "coordinates": "time lat lon",
        "references": "GHCN-Monthly Version 3 (Vose et al. 2011), NCEI/NOAA, https://www.ncdc.noaa.gov/ghcnm/v3.php",
    }

    # flesh out additional attributes, based on the variable type
    if var_name == "prcp":
        attributes["long_name"] = "Precipitation, monthly total"
        attributes["standard_name"] = "precipitation_amount"
        attributes["units"] = "millimeter"
        attributes["valid_min"] = np.float32(0.0)
        attributes["valid_max"] = np.float32(2000.0)
    else:
        attributes["standard_name"] = "air_temperature"
        attributes["units"] = "degree_Celsius"
        attributes["valid_min"] = np.float32(-100.0)
        attributes["valid_max"] = np.float32(100.0)
        if var_name == "tave":
            attributes["long_name"] = "Temperature, monthly average of daily averages"
        elif var_name == "tmax":
            attributes["long_name"] = "Temperature, monthly average of daily maximums"
        elif var_name == "tmin":
            attributes["long_name"] = "Temperature, monthly average of daily minimums"
        else:
            raise ValueError(f"The variable_name argument \"{var_name}\" is unsupported.")

    return attributes


# ------------------------------------------------------------------------------
def download_var_ascii(
        dest_dir: str,
        source_file_name: str,
        var_name,
) -> str:

    with tempfile.TemporaryDirectory() as download_dir:

        # download the GZIP file from NCEI
        ftp = ftplib.FTP(_NCLIMGRID_FTP_URL)
        ftp.login("anonymous", "ftplib-example")
        ftp.cwd(_NCLIMGRID_FTP_DIR)
        destination_path = os.path.join(download_dir, source_file_name)
        ftp.retrbinary("RETR " + source_file_name, open(destination_path, "wb").write)
        ftp.quit()

        # unzip the GZIP file and get the variable's point (ASCII) file
        ascii_file_path = None
        with tarfile.open(destination_path, "r:gz") as tar_file:
            tar_file.extractall(path=download_dir)
            for point_file in tar_file.getmembers():

                if point_file.name.endswith(f"{var_name}.conus.pnt"):
                    source_file_path = os.path.join(download_dir, point_file.name)
                    ascii_file_path = os.path.join(dest_dir, point_file.name)
                    shutil.move(source_file_path, ascii_file_path)
                    break

            if ascii_file_path is None:
                logger.warning(
                    f"Unable to find point file for {var_name} "
                    f"within {destination_path}",
                )

    return ascii_file_path


# ------------------------------------------------------------------------------
def download_template_ascii(
        ascii_dir: str,
        source_file_name: str,
) -> str:

    # get the precipitation point (ASCII) file from the GZIP source file
    return download_var_ascii(ascii_dir, source_file_name, "prcp")


# ------------------------------------------------------------------------------
def initialize_dataset(
        template_file_name: str,
        var_name: str,
        year_start: int,
        month_start: int,
        year_end: int,
        month_end: int,
) -> xr.Dataset:
    """

    :param template_file_name:
    :param var_name:
    :param year_start:
    :param month_start:
    :param year_end:
    :param month_end:
    :return:
    """

    # use a temporary directory for downloading the ASCII file to use as a template
    with tempfile.TemporaryDirectory() as work_dir:

        # determine the lat and lon coordinate values by extracting these
        # from the initial ASCII file in our list (assumes that each ASCII
        # file contains the same lat/lon coordinates)
        template_ascii_file_path = download_template_ascii(work_dir, template_file_name)
        lat_values, lon_values = get_coordinate_values(template_ascii_file_path)

    min_lat = np.float32(min(lat_values))
    max_lat = np.float32(max(lat_values))
    min_lon = np.float32(min(lon_values))
    max_lon = np.float32(max(lon_values))
    lat_units = "degrees_north"
    lon_units = "degrees_east"
    total_lats = lat_values.shape[0]
    total_lons = lon_values.shape[0]

    # set global group attributes
    global_attributes = {
        "date_created": str(datetime.now()),
        "date_modified": str(datetime.now()),
        "Conventions": "CF-1.6, ACDD-1.3",
        "ncei_template_version": "NCEI_NetCDF_Grid_Template_v2.0",
        "title": "nClimGrid (monthly)",
        "naming_authority": "gov.noaa.ncei",
        "standard_name_vocabulary": "Standard Name Table v35",
        "institution": "National Centers for Environmental Information (NCEI), NOAA, Department of Commerce",
        "geospatial_lat_min": min_lat,
        "geospatial_lat_max": max_lat,
        "geospatial_lon_min": min_lon,
        "geospatial_lon_max": max_lon,
        "geospatial_lat_units": lat_units,
        "geospatial_lon_units": lon_units,
    }

    # create a time coordinate variable with one
    # increment per month of the period of record
    time_units_start_year = 1800
    total_months = ((year_end - year_start) * 12) + month_end - month_start + 1
    time_values = compute_days(year_start, month_start, total_months, time_units_start_year)
    time_attributes = {
        "long_name": "Time, in monthly increments",
        "standard_name": "time",
        "calendar": "gregorian",
        "units": f"days since {time_units_start_year}-01-01 00:00:00",
        "axis": "T",
    }
    time_variable = xr.Variable(dims="time", data=time_values, attrs=time_attributes)

    # create the lat coordinate variable
    lat_attributes = {
        "standard_name": "latitude",
        "long_name": "Latitude",
        "units": lat_units,
        "axis": "Y",
        "valid_min": min_lat,
        "valid_max": max_lat,
    }
    lat_variable = xr.Variable(dims="lat", data=lat_values, attrs=lat_attributes)

    # create the lon coordinate variable
    lon_attributes = {
        "standard_name": "longitude",
        "long_name": "Longitude",
        "units": lon_units,
        "axis": "X",
        "valid_min": min_lon,
        "valid_max": max_lon,
    }
    lon_variable = xr.Variable(dims="lon", data=lon_values, attrs=lon_attributes)

    # create the data variable's array
    variable_data = \
        np.full(
            (time_variable.shape[0], total_lats, total_lons),
            fill_value=np.float32(np.NaN),
            dtype=np.float32,
        )
    coordinates = {
        "time": time_variable,
        "lat": lat_variable,
        "lon": lon_variable,
    }
    data_array = \
        xr.DataArray(
            data=variable_data,
            coords=coordinates,
            dims=["time", "lat", "lon"],
            name=var_name,
            attrs=get_variable_attributes(var_name),
        )

    # package it all as an xarray.Dataset
    dataset = \
        xr.Dataset(
            data_vars={var_name: data_array},
            coords=coordinates,
            attrs=global_attributes,
        )

    return dataset


# ------------------------------------------------------------------------------
def download_data(
        file_name: str,
        var_name: str,
        min_lat: float,
        min_lon: float,
        total_lats: int,
        total_lons: int,
) -> np.ndarray:
    """

    :param file_name:
    :param var_name:
    :param min_lat:
    :param min_lon:
    :param total_lats:
    :param total_lons:
    :return:
    """

    # allocate the numpy array we'll fill and return
    grid_shape = (total_lats, total_lons)
    variable_data = np.full(grid_shape, np.NaN, dtype=np.float32)

    try:
        with tempfile.TemporaryDirectory() as work_directory:

            # get the ASCII file we need for the nClimGrid variable
            ascii_file = download_var_ascii(work_directory, file_name, var_name)

            # if we didn't find data for the date range then return an empty array
            if ascii_file is None:
                logger.warning(
                    f"Unable to find {var_name} data in source file {file_name}",
                )
                return variable_data

            # create a Pandas DataFrame from the file
            df = pd.read_csv(ascii_file, delim_whitespace=True, names=["lat", "lon", "value"])

        # successive lats and lons are separated by 1/24th of a degree (regular grid)
        increment = 1 / 24.

        # create lat and lon index columns corresponding to
        # the dataframe's lat and lon values, with the index
        # starting at the minimum, i.e. lat_index[0] == min_lat
        df["lat_index"] = (round((df.lat - min_lat) / increment)).astype(int)
        df["lon_index"] = (round((df.lon - min_lon) / increment)).astype(int)

        # fill the data array with data values, using the
        # lat/lon indices to easily account for missing points
        variable_data[df["lat_index"], df["lon_index"]] = df["value"]

        return variable_data

    except Exception as ex:

        logger.error("Unable to download data", ex)


# ------------------------------------------------------------------------------
def ingest_nclimgrid(
        arguments: Dict,
) -> str:
    """
    Ingests one of the four nClimGrid variables into a NetCDF for a specified
    date range.

    :param arguments: dictionary of argument values
         dest_dir: directory where NetCDF file should be written
         var_name: name of variable, "prcp", "tmin", "tmax", or "tave"
         date_start: starting year and month of date range (inclusive), with format "YYYYMM"
         date_end: ending year and month of date range (inclusive), with format "YYYYMM"
    :return:
    """

    logger.info(
        f"Ingesting nClimGrid data for variable '{arguments['var_name']}' "
        f"and date range {arguments['date_start']} - {arguments['date_end']}",
    )

    # find the FTP files within our date range
    try:
        ftp_files_within_range = \
            find_files_within_range(
                arguments["date_start"],
                arguments["date_end"],
            )
    except Exception as ex:
        logger.error(f"Failed to get files for variable '{arguments['var_name']}'", ex)
        raise ex

    # initialize the xarray.DataSet
    dataset = \
        initialize_dataset(
            ftp_files_within_range[0],
            arguments["var_name"],
            int(arguments["date_start"][:4]),
            int(arguments["date_start"][4:]),
            int(arguments["date_end"][:4]),
            int(arguments["date_end"][4:]),
        )

    # logger.info(f"Dataset for variable '{arguments['var_name']}': {dataset}")

    # minimum coordinate values (used for later function calls)
    min_lat = min(dataset["lat"].data)
    min_lon = min(dataset["lon"].data)

    # for each month download the data from FTP and convert to a numpy array,
    # adding it into the xarray dataset at the appropriate index
    for time_step, ftp_file_name in enumerate(ftp_files_within_range):

        # get the values for the month
        monthly_values = \
            download_data(
                ftp_file_name,
                arguments["var_name"],
                min_lat,
                min_lon,
                len(dataset["lat"]),
                len(dataset["lon"]),
            )

        # add the monthly values into the data array for the variable
        dataset[arguments["var_name"]][time_step] = monthly_values

    # write the xarray DataSet as NetCDF file into the destination directory
    file_name = \
        f"nclimgrid_v1.0_{arguments['var_name']}_"\
        f"{arguments['date_start']}_{arguments['date_end']}.nc"
    var_file_path = os.path.join(arguments["dest_dir"], file_name)
    logger.info(f"Writing nClimGrid NetCDF {var_file_path}")
    dataset.to_netcdf(var_file_path)

    return var_file_path


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # parse the command line arguments
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--dest_dir",
        required=True,
        help="directory where the final, full time series NetCDF files should be written",
    )
    argument_parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="year/month date at which the dataset should start (inclusive), in 'YYYYMM' format",
    )
    argument_parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="year/month date at which the dataset should end (inclusive), in 'YYYYMM' format",
    )
    args = vars(argument_parser.parse_args())

    # create an iterable containing dictionaries of parameters, with one
    # dictionary of parameters per variable, since there will be a separate
    # ingest process per variable, with each process having its own set
    # of parameters
    variables = ["prcp", "tave", "tmin", "tmax"]
    params_list = []
    for variable_name in variables:
        params = {
            "var_name": variable_name,
            "dest_dir": args["dest_dir"],
            "date_start": args["start"],
            "date_end": args["end"],
        }
        ingest_nclimgrid(params)
    #     params_list.append(params)
    #
    # # create a process pool, mapping the ingest
    # # process to the iterable of parameter lists
    # workers_count = 1  # min(len(variables), multiprocessing.cpu_count())
    # logger.info(f"Using a process pool with {workers_count} workers")
    # pool = multiprocessing.Pool()
    # result = pool.map_async(ingest_nclimgrid, params_list)
    #
    # # get the result exception, if any
    # pool.close()
    # pool.join()

    exit(0)
