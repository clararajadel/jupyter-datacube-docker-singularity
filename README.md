# Jupyter-Datacube-Docker-Singularity
- Run **Jupyter Notebooks** in A4Floods VM (**virtual machine**) without super-user root. 
- Install **datacube** to import sentinel images. 
- Use **Docker** to automate the set up between different universities = users. (Having your jupyter server run as a container is a must as it allows one to seamlessly move their lab, as it were, from one cloud to another). 
- Use **Singularity** to work in the ACube4Floods VM **without super-user root**.

# Workflow
1. Request access to Acube server (ssh key)
2. Locally: write Dockerfile, build Docker image and push to docker server
3. Acube server: build and run singularity image

# Generate key to connect with acube server
- Keys are saved in .ssh folder: one is the public (end with .pub). More info to customize your ssh key is found in: http://man.openbsd.org/cgi-bin/man.cgi/OpenBSD-current/man1/ssh-keygen.1?query=ssh-keygen&sec=1
- Create key:

```
(base) clara@LAPTOP-RKJGL9HN:~$ cd .ssh
(base) clara@LAPTOP-RKJGL9HN:~/.ssh$ ssh-keygen -t rsa
```
if necessary make .ssh directory:
```
(base) clara@LAPTOP-RKJGL9HN:~$ mkdir .ssh
```
- Get access to server (Emma)

# Local set up
## Intro (theory)
Docker image should be build locally because there is not super user root in the server to run Docker. To write a Dockerfile and build the docker image some previous steps should be done:

**1. Create workspace:** folder to save all files together + virtual environment. Docker will read files inside this workspace.

**2. Jupyter notebook:** 
    
   - install jupyter notebook
   - make directory called "nbs" to save all notebooks (this step is important: docker will only read notebooks inside this folder)
   - change jupyter configuration to:
        - launch notebooks inside "nbs" directory
        - avoid authetication issues (adding your own password)
        - launch it in local browser from acube server

**3. Docker**

   - write Dockerfile: all instructions writen to: 
        - launch jupyter notebook 
        - read files inside "nbs" 
        - use enviroment with required libraries (included datacube)
   - Build, run, tag and push docker image in docker server
  
  ## 1. Create workspace
  - **Clone this repository** (if cloned you can avoid following steps):
  ```
  $ git clone https://github.com/clararajadel/jupyter-datacube-docker-singularity.git
  ```
- **Create local workspace**. This folder will contain all files required to build the docker image. Later it will be copied (or cloned) in the VM to have access also to this files from the VM.
```
$ mkdir local
```
- **Create and activate virtual environment** named "jupy-docker" using conda (all path is written because the environment is saved out from conda /envs):
```
$ conda create -p /home/clara/projects/jupyter-datacube-docker-singularity/jupy-docker
$ source activate /home/clara/projects/jupyter-datacube-docker-singularity/jupy-docker
```
## 2. Jupyter notebook
- **Install jupyter and add ipykernel**. Ipykernel is a  Jupyter kernel to work with Python code in Jupyter notebooks.
```
$ conda install jupyter
$ python -m ipykernel install --user --name jupy-docker --display-name "Python (jupy-docker)"
```
if necessary install ipykernel:
```
$ conda install ipykernel
```

- **Make directory "nbs"** inside "local" directory (this step is important: docker will only read notebooks inside this folder)
 ```
 $ cd local
 $ mkdir nbs
 ```
 - **Create password for jupyter**. The terminal will ask you for a password and then will print a string. Copy this string to paste it later in the jupyter configuration.
 ```
$ ipython -c "from notebook.auth import passwd; passwd()"
 ```
- **Customize jupyter configuration** creating a jupyter.py file inside a "local/conf" folder. (Note: default jupyter config can be accessed typing in the terminal: jupyter notebook --generate-config). We create a new config file to not change defaults in your local jupyter configuration.
```
$ mkdir conf
$ nano jupyter.py
```
In jupyter.py:
```
import os
c = get_config()
# Kernel config
c.IPKernelApp.pylab = 'inline'  # if you want plotting support always in your notebook
# Notebook config
c.NotebookApp.notebook_dir = 'nbs'
c.NotebookApp.allow_origin = '*' # put your public IP Address here or * to allow all
c.NotebookApp.ip = '*'
c.NotebookApp.allow_remote_access = True
c.NotebookApp.open_browser = False
c.NotebookApp.password = u'argon2:$argon2id$v=19$m=10240,t=10,p=8$qJRSNqPjEqzc/O97Wzb/Rg$bP+S2ixO8Zh3N/h4HRobxg'
c.NotebookApp.port = int(os.environ.get("PORT", 5200))
c.NotebookApp.allow_root = True
c.NotebookApp.allow_password_change = True
c.ConfigurableHTTPProxy.command = ['configurable-http-proxy', '--redirect-port', '80']
```
 ---
 **Notes about jupyter.py config:**
 - nbs: root directory
 - allow_origin: to all (to launch it from the VM)
 - port: 5200 (use one port that is not used by other users in the VM)
 - allow_root: because the project is shared maybe I should change to False
 ---

- **Access jupyter notebook**:
```
$ jupyter notebook --config=./conf/jupyter
```

## 3. Docker
First we will write a Dockerfile. This Dockerfile will:
1. Install Ubuntu
2. Install python and pip
3. Install all required packages. For that create a file "requirements.txt" with all package names
4. Read an entrypoint.sh file with the order to launch jupyter notebook according to configuration in jupyter.py

From Dockerfile a docker image will be build, tagged and run.

- **Install Docker in your local machine** (if necessary). Docker is installed out from this github repository so files are not visible for you. *I'm using Ubuntu WSL2 in Windows 10 so I followed this documentation https://docs.docker.com/engine/install/ubuntu/. Note that Docker Engine does not run on WSL, you have to have Docker For Windows installed on your host machine and you need to tell the Docker client where the Docker host is if you run Ubuntu in Windows 10: https://medium.com/@sebagomez/installing-the-docker-client-on-ubuntus-windows-subsystem-for-linux-612b392a44c4). For installing Docker Desktop in Windows: https://hub.docker.com/editions/community/docker-ce-desktop-windows/.*
- **Create requirements.txt** in /local with a list of required libraries:
    ```
    (base) clara@LAPTOP-RKJGL9HN:~/projects/jupyter-datacube-docker-singularity/local$ nano requirements.txt
    ```
    - Write inside requirements.txt:
    ```
    jupyter
    numpy
    matplotlib
    random2
    scikit-learn
    ```
- **Create entrypoint.sh** in /local/scripts (create scripts):
    ```
    $ mkdir scripts
    $ cd scripts
    $ nano entrypoint.sh
    ```
    - In entrypoint.sh add the order to open Jupyter Notebook following jupyter.py config. /app path is used because the environment in the Dockerfile will be named /app and singularity needs the entire path to build the image.
    ```
    /usr/local/bin/jupyter-notebook --config=/app/conf/jupyter.py
    ```
    - The format of entrypoint.sh should be changed (íf not maybe you get error while building the image). In the terminal type:
    ```
    chmod +x entrypoint.sh
    ```
- **Create a file named Dockerfile** in /local:
    ```
    (base) clara@LAPTOP-RKJGL9HN:~/projects/jupyter-datacube-docker-singularity/local$ nano Dockerfile
    ```
    - Inside Dockerfile:
    ```
    # BASE IMAGE: UBUNTU
    FROM ubuntu:latest

    # WORKSPACE
    ENV DEBIAN_FRONTEND=noninteractive
    ENV APP_HOME /app
    WORKDIR ${APP_HOME}

    COPY . ./

    # INSTALL PYTHON AND PIP
    RUN apt-get update && apt-get install -y python3 python3-pip
    RUN python3 -m pip install pip --upgrade

    # INSTALL SOFTWARE REQUIRED BY DATACUBE
    RUN apt-get install -y build-essential python3-dev python3-pip python3-venv libyaml-dev libpq-dev
    RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libproj-dev proj-bin libgdal-dev
    RUN apt-get install -y libgeos-dev libgeos++-dev libudunits2-dev libnetcdf-dev libhdf4-alt-dev libhdf5-serial-dev gfortran
    RUN apt-get install -y postgresql-doc libhdf5-doc netcdf-doc libgdal-doc
    RUN apt-get install -y hdf5-tools netcdf-bin gdal-bin pgadmin3

    # ADD DATACUBE
    RUN python3 -m pip install -U pip setuptools
    RUN python3 -m pip install -U wheel 'setuptools_scm[toml]' cython
    RUN python3 -m pip install -U 'pyproj==2.*' 'datacube[all]' --no-binary=rasterio,pyproj,shapely,fiona,psycopg2,netCDF4,h5py

    # GDAL
    RUN python3 -m pip install GDAL==$(gdal-config --version)

    # INSTALL REST OF LIBRARIES
    RUN python3 -m pip install -r ./requirements.txt

    # LAUNCH NOTEBOOKS
    CMD ["/app/scripts/entrypoint.sh"]
    ```
    ---
    **Notes about Dockerfile**

    - The base image is Ubuntu to work with Linux (but python base image I think also includes linux software).
    - The new working environment is /app. That is why the entrypoint.sh order includes /app in the config. (ENV: declare the name of environmental variables and WORKDIR: specifies the work environment)
    - Python and pip are installed
    - I need an environment, if not libraries such as scikit-learn won't work. The issue is because singularity doesn't work as docker regarding isolating the system. So, when using singularity, it is needed to install package or copy your data/files into a location where your user account has the permission. The normal place is "/opt", where all users have the rw access:  /opt is for "the installation of add-on application software packages".  /usr/local is "for use by the system administrator when installing software locally".
    - Then are followed the instructions to install datacube by the documentation: https://datacube-core.readthedocs.io/en/latest/ops/ubuntu.html#python-venv-installation. Moreover the documentation recommends making the installation in a virtual environment (odc), in this case we use: /opt. 
    - After that gdal is installed. Datacube doesn't need GDAL but we use it in our project.
    - Pip installs the requirements.txt libraries.
    - The final command is for reading the entrypoint.sh. You have to write all the filepath if you want to run the code (with Singularity is not possible writing: ./scripts/entrypoint.sh).
    - All commands that include <DEBIAN_FRONTEND=noninteractive> are created to avoid interaction while building the image.
    ---
- **Build the image**: In the terminal, inside the folder where is the Dockerfile type:
```
docker build -t eodc-jupyter -f Dockerfile .
```
- **Create the resgistry image** (this step is in the A4Cube ppt of Dockers). If you avoid this step it won't work.
```
docker run -d -p 5000:5000 --restart=always --name registry registry:2
```
- **Tag your image** with the version (creating "other image": same ID, different name). This is the image to push in the docker server. The port of the computer from where is launched is specified.
```
docker tag eodc-jupyter localhost:5000/eodc-jupyter:1.0
```
- Finally, **push the docker image** to the server and it will become available in other machine through docker server.
```
docker push localhost:5000/eodc-jupyter:1.0
```

# Run singularity in server
- **Acces to A4Floods VM**. You can do local and remote port forwarding in one command --> write in the terminal:
```
ssh -L 5200:localhost:5200 -R 5201:localhost:5000 boku@acube4floods.eodchosting.eu
```
-  **Clone this repository**: jupyter-datacube-docker-singularity repository.
 ```
  $ git clone https://github.com/clararajadel/jupyter-datacube-docker-singularity.git
  ```
- **Allow singularity work in server** running the following command. (I did a similar step for running docker in Ubuntu for Windows, see in install Docker section)
```
export SINGULARITY_NOHTTPS=1
```
- **Build and run singularity image**. 
```
singularity build datacube.simg docker://localhost:5201/eodc-jupyter:1.0
singularity exec -B /eodc:/eodc datacube.simg  /app/scripts/entrypoint.sh
```
    - B /eodc:/eodc : the /eodc storage is not available inside the singularity container. Therefore, You need to bind the /eodc to the singularity container with -B option.

-  **Access your jupyter notebooks** at your browser at url: localhost:5200

# Save new notebooks

