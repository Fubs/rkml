import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

dfilename = "grouped_in.npy"
insets = np.load(dfilename,allow_pickle=True)
outsets = np.load("grouped_out.npy",allow_pickle=True)

for i in range(len(insets)):
    t = [n[0] for n in insets[i]]
    y = outsets[i]
    plt.plot(t,y)
    plt.show()



'''
tdata = np.array([i[0] for i in dv])
ydata = np.array([i[1] for i in dv])
y2data = np.array([i[2] for i in dv])

xmax = -100

plt.plot(tdata[xmax:],ydata[xmax:])
plt.plot(tdata[xmax:],y2data[xmax:])
plt.show()
'''
