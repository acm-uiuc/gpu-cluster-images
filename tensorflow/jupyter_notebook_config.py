import os
from IPython.lib import passwd

c.NotebookApp.ip = '*'
c.NotebookApp.port = int(os.getenv('PORT', 8888))
c.NotebookApp.open_browser = False
c.MultiKernelManager.default_kernel_name = 'python3'
c.NotebookApp.allow_origin = "http://gpu0.acm.illinois.edu"
c.NotebookApp.allow_origin = "http://gpu1.acm.illinois.edu"
