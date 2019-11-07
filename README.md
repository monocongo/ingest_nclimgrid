# ingest_nclimgrid
Ingest code to convert the gridded GHCN-D temperature and precipitation dataset 
([nClimGrid](https://data.noaa.gov/dataset/dataset/gridded-5km-ghcn-daily-temperature-and-precipitation-dataset-version-1)) 
from ASCII datsets into NetCDF.

ASCII versions of the monthly averaged nClimGrid datasets in GIS "point file" format a
re available from NOAA [here](ftp://ftp.ncdc.noaa.gov/pub/data/climgrid), and are 
used as the input sources for this conversion process.

#### Example usage:
```bash
$ python3 nclimgrid_ingest.py --dest_dir /data/nclimgrid --start 189501 --end 201909
```
