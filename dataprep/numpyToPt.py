import os
import sys
import numpy as np

#import matplotlib.pyplot as plt
#fig, ax = plt.subplots()

all_in = []
all_out = []
all_out2 = []

for fnum in range(9):
    dfilename = "npzips/pendulumData"+str(fnum)+".npy"
    dv = np.load(dfilename)
    tdata = np.array([i[0] for i in dv])
    tstep = tdata[2] - tdata[1]
    tdata = [[n,tstep] for n in tdata]

    ydata = [i[1] for i in dv]
    y2data = [i[2] for i in dv]

    print(tstep)

    all_in += tdata
    all_out += ydata
    all_out2 += y2data
    
all_in = np.array(all_in).astype('float64')
all_out = np.array(all_out).astype('float64')

all_out2 = np.array(all_out2).astype('float64')

np.save("all_in.npy",all_in)
np.save("all_out.npy",all_out)
np.save("all_out2.npy",all_out2)

#torch.save(all_in, open('allIn.pt','wb'))
#torch.save(all_out, open('allOut.pt','wb'))
#torch.save(all_out2, open('allOut2.pt','wb'))




#xmax = -100
#plt.plot(tdata[xmax:],ydata[xmax:])
#plt.plot(tdata[xmax:],y2data[xmax:])
#plt.show()
