import os
c = get_config()
# Kernel config
c.IPKernelApp.pylab = 'inline'  # if you want plotting support always in your notebook
# Notebook config
c.NotebookApp.notebook_dir = '/app/nbs'
c.NotebookApp.allow_origin = '*'  # put your public IP Address here ej. u'cfe-jupyter.herokuapp.com'
c.NotebookApp.ip = '*'
c.NotebookApp.allow_remote_access = True
c.NotebookApp.open_browser = False
# ipython -c "from notebook.auth import passwd; passwd()"
# passwd is: eodc
c.NotebookApp.password = u'argon2:$argon2id$v=19$m=10240,t=10,p=8$fpFT7lWZx/g/dm6VzOWtgQ$vNoj2M8c9y4pVlxxp4xhAg'
c.NotebookApp.port = int(os.environ.get("PORT", 5200))
c.NotebookApp.allow_root = True
c.NotebookApp.allow_password_change = True
c.ConfigurableHTTPProxy.command = ['configurable-http-proxy', '--redirect-port', '80']
