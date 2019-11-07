# ingest_nclimgrid
Ingest code to convert NOAA's 5km gridded GHCN-D temperature and precipitation dataset 
([nClimGrid](https://data.noaa.gov/dataset/dataset/gridded-5km-ghcn-daily-temperature-and-precipitation-dataset-version-1)) 
from ASCII to [NetCDF](https://www.unidata.ucar.edu/software/netcdf/).

ASCII versions of the monthly averaged nClimGrid datasets in GIS "point file" format 
are available for download from [NCEI/NOAA](ftp://ftp.ncdc.noaa.gov/pub/data/climgrid), 
and are used as the input sources for this conversion process.

The results of this ingest process should be available from NOAA [here](https://www.ncei.noaa.gov/thredds/catalog/data-in-development/nclimgrid/catalog.html) 
by the end of the first week of the month. However if you're impatient or if NOAA 
delays posting the NetCDF versions of the data (it's often out-of-date by several 
months) then use this code to cook up the datasets yourself.

#### Example usage:
```bash
$ python3 nclimgrid_ingest.py --dest_dir /data/nclimgrid --start 189501 --end 201909
```
