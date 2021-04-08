# jupyter-datacube-docker-singularity
- Run Jupyter Notebooks in A4Floods VM (without super-user root). 
- Install datacube to import sentinel images. 
- Use Docker to automate the set up between different universities = users. (Having your jupyter server run as a container is a must as it allows one to seamlessly move their lab, as it were, from one cloud to another). 
- Use Singularity to work in the ACube4Floods VM without super-user root.

## Generate key to connect with the server
- Keys are saved in .ssh folder: one is the public (end with .pub)
- More info to customize your ssh key is found in: http://man.openbsd.org/cgi-bin/man.cgi/OpenBSD-current/man1/ssh-keygen.1?query=ssh-keygen&sec=1

(base) clara@LAPTOP-RKJGL9HN:~$ mkdir .ssh (if necessary)
(base) clara@LAPTOP-RKJGL9HN:~$ cd .ssh
(base) clara@LAPTOP-RKJGL9HN:~/.ssh$ ssh-keygen -t rsa
