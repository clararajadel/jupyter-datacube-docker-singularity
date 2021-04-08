# Jupyter-Datacube-Docker-Singularity
- Run Jupyter Notebooks in A4Floods VM (without super-user root). 
- Install datacube to import sentinel images. 
- Use Docker to automate the set up between different universities = users. (Having your jupyter server run as a container is a must as it allows one to seamlessly move their lab, as it were, from one cloud to another). 
- Use Singularity to work in the ACube4Floods VM without super-user root.

# Workflow
1. Request access to Acube server (ssh key)
2. Locally: write Dockerfile, build Docker image and push to docker server
3. Acube server: build and run singularity image

# Generate key to connect with acube server
- Keys are saved in .ssh folder: one is the public (end with .pub)
- More info to customize your ssh key is found in: http://man.openbsd.org/cgi-bin/man.cgi/OpenBSD-current/man1/ssh-keygen.1?query=ssh-keygen&sec=1
- Create key:

```
(base) clara@LAPTOP-RKJGL9HN:~$ mkdir .ssh (if necessary)
(base) clara@LAPTOP-RKJGL9HN:~$ cd .ssh
(base) clara@LAPTOP-RKJGL9HN:~/.ssh$ ssh-keygen -t rsa
```
- Get access to server (Emma)

# Local set up
Docker image should be build locally because there is not super user root in the server to run Docker. 

To write a Dockerfile and build the docker image some previous steps should be done:

**1. Create workspace:** folder to save all files together + virtual environment.

**2. Jupyter notebook:** 
    
   - install jupyter notebook
   - make directory called "nbs" to save all notebooks (this step is important: docker will only read notebooks inside this folder)
   - change jupyter configuration 
    - avoid authetication issues (password)
    - to be able to launch it locally from acube server
    - to launch notebooks inside "nbs" directory

**3. Docker**

   - write Dockerfile: all instructions writen to launch jupyter notebook, read files inside "nbs" and use enviroment with required libraries (included datacube)
   - Build, run, tag and push docker image
  
  ## 1. Create workspace
  - Clone this repository:
  ```
  (base) clara@LAPTOP-RKJGL9HN:~/projects$ git clone https://github.com/clararajadel/jupyter-datacube-docker-singularity.git
  ```
<!-- If you clone this repository you can avoid following steps -->
