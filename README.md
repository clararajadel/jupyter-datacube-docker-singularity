# Jupyter-Datacube-Docker-Singularity
## Objective
- Run **Jupyter Notebooks** in A4Floods VM (**virtual machine**) without super-user root. 
- Install **datacube** to import sentinel images. 
- Use **Docker** to automate the set up between different universities = users. (Having your jupyter server run as a container is a must as it allows one to seamlessly move their lab, as it were, from one cloud to another). 
- Use **Singularity** to work in the ACube4Floods VM **without super-user root**.

## Workflow
1. Request access to Acube server (ssh key)
2. Locally: write Dockerfile, build Docker image and push to docker server
3. Acube server: build and run singularity image

## Table of Contents  
* [Generate key to connect with acube server](#Generate-key-to-connect-with-acube-server)
* [Local set up](#Local-set-up)
   * [Create workspace](#Create-workspace)
   * [Jupyter notebook](#Jupyter-notebook)
   * [Docker](#Docker)
      * [Dockerfile](#Dockerfile)
      * [Docker Image](#Docker-Image) 
* [Run singularity in server](#Run-singularity-in-server)
* [Working with singularity after setup](#Working-with-singularity-after-setup)
* [Errors](#Errors)

# Generate key to connect with acube server
- Keys are saved in .ssh folder: one is the public (end with .pub). More info to customize your ssh key is found in: http://man.openbsd.org/cgi-bin/man.cgi/OpenBSD-current/man1/ssh-keygen.1?query=ssh-keygen&sec=1
- Create key:

```
$ cd .ssh
$ ssh-keygen -t rsa
```
if necessary make .ssh directory:
```
$ mkdir .ssh
```
- Get access to server (Emma)
- Access server: bind A4F VM 5200 to your local 5200
```
$ ssh -L 5200:localhost:5200 boku@acube4floods.eodchosting.eu
```

# Local set up

Docker image should be build locally because there is not super user root in the server to run Docker. To write a Dockerfile and build the docker image some previous steps should be done:

**1. Create workspace:** folder to save all files together + virtual environment. Docker will read files inside this workspace.

**2. Jupyter notebook:** 
    
   - install jupyter notebook
   - change jupyter configuration for:
        - launching notebooks contained inside "nbs" directory
        - avoiding authentication issues (adding your own password)
        - launching notebooks in local browser from acube server

**3. Docker**

   - write Dockerfile: file with instructions written for: 
        - creating an enviroment with all required libraries (included datacube)
        - launching jupyter notebooks inside "nbs" 
   - Build, run, tag and push docker image in docker server
  
  ## Create workspace
  - **Clone this repository** (if cloned you can avoid some of following steps):
  ```
  $ git clone https://github.com/clararajadel/jupyter-datacube-docker-singularity.git
  ```
- **Create local workspace**. This folder will contain all files required to build the docker image. Later it will be cloned (using git) in the VM to have access to these files from the VM.
```
$ mkdir local
```
- **Create and activate virtual environment** named "jupy-docker" using conda. The environment is created to use jupyter locally (without datacube) and verify that the new configuration works properly. This step is not mandatory.
```
$ conda create -p /home/clara/projects/jupyter-datacube-docker-singularity/jupy-docker
$ source activate /home/clara/projects/jupyter-datacube-docker-singularity/jupy-docker
```
(all path is written because the environment is saved out from default path: /miniconda3/envs)
## Jupyter notebook
- **Activate conda environment** (if it is not activated):
```
$ source activate /home/clara/projects/jupyter-datacube-docker-singularity/jupy-docker
```
- **Install jupyter and add ipykernel**. Ipykernel is a  Jupyter kernel to work with Python code in Jupyter notebooks.
```
$ conda install jupyter
$ python -m ipykernel install --user --name jupy-docker --display-name "Python (jupy-docker)"
```
before: if necessary install ipykernel:
```
$ conda install ipykernel
```

- **Make directory "nbs" and add notebooks** outside "local" directory. Write a small code in a .ipynb file and save it in "nbs" folder. This directory will contain your notebooks and will be cloned in the VM. Adding at least one notebook is necessary because if not "nbs" folder won't be cloned. Later from the VM, using jupyter notebook with singularity image, you will be able to modify notebooks and save new changes. It is outside from "local" directory because there we save all files related with docker image. To modify notebooks using singularity image they have to be outside from docker image and, as well, singularity image. If they are inside, you can not modify them once images are built. Jupyter will open the directory that you configure in its jupyter configuration, meanwhile its path contains folder "nbs" you can work on it.
 ```
 $ cd jupyter-datacube-docker-singularity/
 $ mkdir nbs
 ```
 - **Create password for jupyter**. The terminal will ask you for a password and then will print a string. Copy this string to paste it later in the jupyter configuration.
 ```
$ ipython -c "from notebook.auth import passwd; passwd()"
 ```
- **Customize jupyter configuration** creating [`jupyter.py`](https://github.com/clararajadel/jupyter-datacube-docker-singularity/blob/main/local/conf/jupyter.py) file inside "local/conf" folder. (Note: default jupyter config can be accessed typing in the terminal: jupyter notebook --generate-config). We create a new config file to not change defaults in your local jupyter configuration. All info about jupyter configuration can be found here https://jupyter-notebook.readthedocs.io/en/stable/config.html.
```
$ cd local
$ mkdir conf
$ cd conf
$ nano jupyter.py
```

 ---
 **Notes about jupyter.py config:**
 - c.NotebookApp.notebook_dir: if you want to open jupyter locally write your local path (e.g. /home/clara). But when you are going to create Dockerfile, change this path to your server path (e.g. /home/boku)
 - allow_origin: to all (to launch it from the VM)
 - port: 5200 (use one port that is not used by other users in the VM)
 - allow_root: because the project is shared maybe I should change to False
 ---

- **Access jupyter notebook and save a .ipynb**: If you want to check jupyter configuration is working (path, password, ect) make sure you did step: Install jupyter and ipykernel, and your path in "c.NotebookApp.notebook_dir" in [`jupyter.py`](https://github.com/clararajadel/jupyter-datacube-docker-singularity/blob/main/local/conf/jupyter.py) is to your local machine, not the server. Then type:
```
$ jupyter notebook --config=./conf/jupyter
```

## Docker
- **Install Docker in your local machine** (if necessary). Docker is installed out from this github repository so files are not visible for you. 

- **Create a [Dockerfile](#Dockerfile)**. This Dockerfile will contain instructions to:
1. Install Ubuntu
2. Install python and pip
3. Install all required packages --> create a file "requirements.txt" with all package names
4. Read an entrypoint.sh file to launch jupyter notebook according to configuration in jupyter.py --> create file "entrypoint.sh"

- **Create a [Docker Image](#Docker-Image)**. From Dockerfile a docker image will be build, tagged and run.

---

**Note: install Docker for Ubuntu in Windows 10:** 

I followed this documentation https://docs.docker.com/engine/install/ubuntu/. It didn't work: note that Docker Engine does not run on WSL, you have to have Docker For Windows installed on your host machine and you need to tell the Docker client where the Docker host is if you run Ubuntu in Windows 10: https://medium.com/@sebagomez/installing-the-docker-client-on-ubuntus-windows-subsystem-for-linux-612b392a44c4). For installing Docker Desktop in Windows: https://hub.docker.com/editions/community/docker-ce-desktop-windows/. I had to use WSL2.

---

### Dockerfile

According to previous [Docker](#Docker) explanation, Dockerfile requires:
  a) "requirements.txt" file for installing libraries. It allows to simplify libraries installation in one line in Dockerfile.
  b) "entrypoint.sh" that contains the order of opening jupyter notebook (with specific configuration) and will be read by Dockerfile.

- **Create [`requirements.txt`](https://github.com/clararajadel/jupyter-datacube-docker-singularity/blob/main/local/requirements.txt)** in /local with a list of libraries:
    ```
    $ nano requirements.txt
    ```

- **Create [`entrypoint.sh`](https://github.com/clararajadel/jupyter-datacube-docker-singularity/blob/main/local/scripts/entrypoint.sh)** in /local/scripts (create scripts). This script will contain the order to install ipykernel and open Jupyter Notebook following jupyter.py config. It refers to /app/opt path because the environment in the Dockerfile will be named /app/opt and singularity needs an entire path to build the image.
    ```
    $ mkdir scripts
    $ cd scripts
    $ nano entrypoint.sh
    ```
    - The format of entrypoint.sh should be changed to allow access (íf not maybe you get error while building the image). In the terminal type:
    ```
    chmod +x entrypoint.sh
    ```
- **Create a file named Dockerfile** inside /local folder:
    ```
    $ nano Dockerfile
    ```
    - Dockerfile text: [`Dockerfile`](https://github.com/clararajadel/jupyter-datacube-docker-singularity/blob/main/local/Dockerfile)
    
    ---
    **Notes about Dockerfile**

    - The base image is Ubuntu to work with Linux (but python base image I think also includes linux software).
    - The new working environment is /app. That is why the entrypoint.sh order includes /app/opt in the config. (ENV: declare the name of environmental variables and WORKDIR: specifies the work environment)
    - Python and pip are installed
    - I need an environment, if not libraries such as scikit-learn won't work. The issue is because singularity doesn't work as docker regarding isolating the system. So, when using singularity, it is needed to install package or copy your data/files into a location where your user account has the permission. The normal place is "/opt", where all users have the rw access:  /opt is for "the installation of add-on application software packages".  /usr/local is "for use by the system administrator when installing software locally".
    - Then are followed instructions to install datacube by the documentation: https://datacube-core.readthedocs.io/en/latest/ops/ubuntu.html#python-venv-installation. Moreover the documentation recommends making the installation in a virtual environment (odc), in this case we use: /opt. 
    - After gdal is installed. Datacube doesn't need GDAL but we use it in our project.
    - Pip installs requirements.txt libraries.
    - Final command reads entrypoint.sh. You have to write all the filepath if you want to run the code (with Singularity is not possible writing: ./scripts/entrypoint.sh).
    - All commands that include <DEBIAN_FRONTEND=noninteractive> are created to avoid interaction while building the image.
    ---
### Docker Image
- **Build image**: In the terminal, inside the folder where is Dockerfile type:
```
$ docker build -t eodc-jupyter -f Dockerfile .
```
- **Create the resgistry image** (this step is in the A4Cube ppt of Dockers). If you avoid this step it won't work.
```
$ docker run -d -p 5000:5000 --restart=always --name registry registry:2
```
- **Tag your image** with version (creating "other image": same ID, different name). This is the image to push in docker server. The port of launching computer is specified.
```
$ docker tag eodc-jupyter localhost:5000/eodc-jupyter:1.0
```
- Finally, **push docker image** to the server and it will become available in other machine through docker server service.
```
$ docker push localhost:5000/eodc-jupyter:1.0
```

# Run singularity in server
- **Acces to A4Floods VM**. You can do local and remote port forwarding in one command --> write in the terminal:
```
$ ssh -L 5200:localhost:5200 -R 5201:localhost:5000 boku@acube4floods.eodchosting.eu
```
With above command, [-R] you bind your local port 5000(docker-registry port) to A4F VM port 5201, and [-L] bind A4F VM 5200 to your local 5200.
- **Deactivate environments**: If necessary. Don't forget bash.
```
$ conda deactivate
```
-  **Clone this repository**: jupyter-datacube-docker-singularity repository. This is important to have access to "/scripts/entrypoint.sh" (that will be read by singularity) and "nbs" where are required notebooks. You can also copy (-scp) these two folders directly to the server.
```
 $ git clone https://github.com/clararajadel/jupyter-datacube-docker-singularity.git
 ```
 or
 ```
 scp -r /local/path/ boku@acube4floods.eodchosting.eu:/server/path
 ```
- **Allow singularity work in server without https** running the following command. SINGULARITY_NOHTTPS: This is relevant if you want to use a registry that doesn’t have https, and it speaks for itself. If you export the variable SINGULARITY_NOHTTPS you can force the software to not use https when interacting with a Docker registry. This use case is typically for use of a local registry. (https://sylabs.io/guides/3.0/user-guide/build_env.html)
```
$ export SINGULARITY_NOHTTPS=1
```
- **Build and execute singularity image**. You should build your image in the same folder as "/scripts/entrypoint.sh" folder which is inside "local".

    - **[- B /home/boku:/home/boku]** : /home/boku storage is not available inside singularity container. Therefore, you need to bind /home/boku to singularity container with -B option. [-B]: (https://sylabs.io/guides/3.1/user-guide/cli/singularity_exec.html). In my case, /home/boku path contains this project and others, so from this path I can access to "nbs": /home/boku/jupyter-datacube-docker-singularity/nbs once I'm in jupyter notebook. I could also specify more my path instead of /home/boku to /home/boku/jupyter-datacube-docker-singularity/.
```
$ cd jupyter-datacube-docker-singularity/local/
$ singularity build eodc-jupyter.simg docker://localhost:5201/eodc-jupyter:1.0
$ singularity exec -B /home/boku:/home/boku eodc-jupyter.simg  /app/scripts/entrypoint.sh
```
-  **Access your jupyter notebooks** at your browser at url: localhost:5200
-  **Add .simg to [`.gitignore`](https://github.com/clararajadel/jupyter-datacube-docker-singularity/blob/main/.gitignore)**: I can not push .simg image because I cannot instal git-lfs (I can not use sudo).
```
$ cd jupyter-datacube-docker-singularity
$ nano .gitignore
```

# Working with singularity after setup
- Locally:
```
$ ssh -L 5200:localhost:5200 -R 5201:localhost:5000 boku@acube4floods.eodchosting.eu
```
- In VM don´t forget to deactivate all environments and type:
```
$ export SINGULARITY_NOHTTPS=1
$ cd jupyter-datacube-docker-singularity/local/
$ singularity exec -B /home/boku:/home/boku eodc-jupyter.simg  /app/scripts/entrypoint.sh
```
# Errors
- When creating new .simg: tar error
```
gzip: /home/boku/.singularity/docker/sha256:32ce018fc27dbf5410317578350e089b00141c4428e42db7ac7081ffc3ae0b76.tar.gz: unexpected end of file
tar: Unexpected EOF in archive
tar: Unexpected EOF in archive
tar: Error is not recoverable: exiting now
Cleaning up...
```
This is a problem of cache. Senmao said: you should go to the .singularity folder and remove cache. Delete content in docker and metadata folders.
```
$ cd .singularity
$ rm ~/.singularity/docker/*
$ rm ~/.singularity/metadata/*
```
- When I import datacube in the notebook from the .simg image: 
```
OSError: /app/opt/venv/lib/libgeos_c.so: cannot open shared object file: No such file or directory
```
Solution: deactivate all environments from the terminal (not forget bash):
```
$ conda deactivate
```
- If a new notebook pasted in "nbs" does not work maybe is because it is not in executable mode (you can check it typing "ls -la" in "nbs". To allow access:
```
$ chmod +x (name of the .py or .ipynb file)
```
