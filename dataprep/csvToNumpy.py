try: 
    import exptnumpy as np
except:
    import numpy as np

import sys 
import os

datadir = "csvfiles"
c = 1
for dfile in os.listdir(datadir):
    with open(os.path.join(datadir,dfile),'r') as f:
        print("converting csv data into numpy array", c)
        c+=1
        lines = [l.rstrip() for l in f.readlines()]
        dv = [l.split(',') for l in lines]
        dv = [[float(n) for n in l] for l in dv]
        arr = np.array(dv)
        np.save("npzips/"+dfile.rstrip(".csv"),arr)


