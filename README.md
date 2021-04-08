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
  - Clone this repository (if cloned you can avoid following steps):
  ```
  (base) clara@LAPTOP-RKJGL9HN:~/projects$ git clone https://github.com/clararajadel/jupyter-datacube-docker-singularity.git
  ```
- Create directories for local files and acube server files.
```
(base) clara@LAPTOP-RKJGL9HN:~/projects/jupyter-datacube-docker-singularity$ mkdir local
(base) clara@LAPTOP-RKJGL9HN:~/projects/jupyter-datacube-docker-singularity$ mkdir server
```
- Create and activate virtual environment named "jupy-docker" using conda (all path is written because the environment is saved out from conda /envs):
```
(base) clara@LAPTOP-RKJGL9HN:~/projects/jupyter-datacube-docker-singularity$ conda create -p /home/clara/projects/jupyter-datacube-docker-singularity/jupy-docker
(base) clara@LAPTOP-RKJGL9HN:~/projects/jupyter-datacube-docker-singularity$ source activate /home/clara/projects/jupyter-datacube-docker-singularity/jupy-docker
```
## 2. Jupyter notebook
- Install jupyter and add ipykernel. Ipykernel is a  Jupyter kernel to work with Python code in Jupyter notebooks.
```
(/home/clara/projects/jupyter-datacube-docker-singularity/jupy-docker) clara@LAPTOP-RKJGL9HN:~/projects/jupyter-datacube-docker-singularity$ conda install jupyter
(/home/clara/projects/jupyter-datacube-docker-singularity/jupy-docker) clara@LAPTOP-RKJGL9HN:~/projects/jupyter-datacube-docker-singularity$ python -m ipykernel install --user --name jupy-docker --display-name "Python (jupy-docker)"
```
if necessary install ipykernel:
```
(/home/clara/projects/jupyter-datacube-docker-singularity/jupy-docker) clara@LAPTOP-RKJGL9HN:~/projects/jupyter-datacube-docker-singularity$ conda install ipykernel
```

- Make directory "nbs" inside "local" directory
 ```
 (/home/clara/projects/jupyter-datacube-docker-singularity/jupy-docker) clara@LAPTOP-RKJGL9HN:~/projects/jupyter-datacube-docker-singularity$ cd local
 (/home/clara/projects/jupyter-datacube-docker-singularity/jupy-docker) clara@LAPTOP-RKJGL9HN:~/projects/jupyter-datacube-docker-singularity/local$ mkdir nbs
 ```
 - Create password for jupyter. The terminal will ask you for a password and then will print a string. Copy this string to paste it later in the jupyter configuration.
 ```
 (/home/clara/projects/jupyter-datacube-docker-singularity/jupy-docker) clara@LAPTOP-RKJGL9HN:~/projects/jupyter-datacube-docker-singularity/local$ ipython -c "from notebook.auth import passwd; passwd()"
 ```
- Customize jupyter configuration creating a jupyter.py file inside a "conf" folder. (Note: default jupyter config can be accessed typing in the terminal: jupyter notebook --generate-config). We create a new config file to not change defaults in your local jupyter configuration.
```
(/home/clara/projects/jupyter-datacube-docker-singularity/jupy-docker) clara@LAPTOP-RKJGL9HN:~/projects/jupyter-datacube-docker-singularity/local$ mkdir conf
(/home/clara/projects/jupyter-datacube-docker-singularity/jupy-docker) clara@LAPTOP-RKJGL9HN:~/projects/jupyter-datacube-docker-singularity/local$ nano jupyter.py
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
Notes about jupyter.py config:
    - nbs: root directory
    - allow_origin: to all (to launch it from the VM)
    - port: 5200 (use one port that is not used by other users in the VM)
    - allow_root: because the project is shared maybe I should change to False

- Access jupyter notebook:
```
(/home/clara/projects/jupyter-datacube-docker-singularity/jupy-docker) clara@LAPTOP-RKJGL9HN:~/projects/jupyter-datacube-docker-singularity/local$ jupyter notebook --config=./conf/jupyter
```
