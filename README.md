Configuration notes: To make it works with Pycharm:

+ add the following to the Environment Variables section in the Run/Debug Configuration options found in Run > Edit Configurations... dialog: LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

+ add the following to the Environmental Variables section in File > Settings > Build,Execution,Deployment > Console > Python Console: Name: LD_LIBRARY_PATH Value: /usr/local/cuda/lib64
