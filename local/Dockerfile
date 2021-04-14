# BASE IMAGE
FROM ubuntu:latest


# WORKING DIRECTORY
ENV DEBIAN_FRONTEND=noninteractive
ENV APP_HOME /app
WORKDIR ${APP_HOME}


# PASTE ALL FILES IN THE WORKING DIRECTORY
COPY . ./


# INSTALL PYTHON AND PIP
RUN apt-get update && apt-get install -y python3 python3-pip
RUN python3 -m pip install pip --upgrade


# INSTALL SOFTWARE REQUIRED BY DATACUBE
RUN apt-get install -y build-essential python3-dev python3-pip python3-venv libyaml-dev libpq-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libproj-dev proj-bin libgdal-dev
RUN apt-get install -y libgeos libgeos-dev libgeos++-dev libudunits2-dev libnetcdf-dev libhdf4-alt-dev libhdf5-serial-dev gfortran
RUN apt-get install -y postgresql-doc libhdf5-doc netcdf-doc libgdal-doc 
RUN apt-get install -y hdf5-tools netcdf-bin gdal-bin pgadmin3


# CREATE VIRTUAL ENVIRONMENT
RUN python3 -m venv /app/opt/venv


# ADD DATACUBE
RUN /app/opt/venv/bin/pip install -U pip setuptools
RUN /app/opt/venv/bin/pip install -U wheel 'setuptools_scm[toml]' cython
RUN /app/opt/venv/bin/pip install -U 'pyproj==2.*' 'datacube[all]' --no-binary=rasterio,pyproj,shapely,fiona,psycopg2,netCDF4,h5py


# GDAL
RUN /app/opt/venv/bin/pip install GDAL==$(gdal-config --version)


# INSTALL REST OF LIBRARIES
RUN /app/opt/venv/bin/pip install -r ./requirements.txt
#RUN /opt/venv/bin/python3 -m pip install -r ./requirements.txt

# CALL NOTEBOOKS
CMD ["/app/scripts/entrypoint.sh"]
