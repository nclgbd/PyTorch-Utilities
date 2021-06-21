import os

CMDS =  """
            pip uninstall pretrained-models.pytorch\n
            git clone https://github.com/nclgbd/pretrained-models.pytorch.git\n
            cd pretrained-models.pytorch\n
            python setup.py install\n
            pip install -r requirements.txt\n

        """
# Have to manually install
os.system(CMDS)

