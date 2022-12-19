import os, sys
from concurrent.futures import *
import numpy as np

files = os.listdir("csvfiles")

THREADS = len(files)

grouped_t = []
grouped_tstep = []
grouped_y = []
grouped_y2 = []
net_data = []

tstep_mapping = []
with open('csvfiles/h.csv', 'r') as f:
    lines = f.readlines()
    lines = [float(l.rstrip()) for l in lines]
    tstep_mapping = np.array(lines).astype(np.float32)

def buildLists(file, h):
    with open("csvfiles/"+file, "r") as f:
        lines = f.readlines()
        lines = [l.rstrip() for l in lines]
        #print("first 10 lines: ",  lines[:10])
        dv = [l.split(',') for l in lines]
        dv = [[np.float32(n) for n in l] for l in dv]
        dv = np.array(dv).astype(np.float32)
        #print("dv shape: ", dv.shape)
        #print("dv: ", dv[:10])
        tdata = np.array([i[0] for i in dv]).astype(np.float32)
        ydata = np.array([i[1] for i in dv]).astype(np.float32)
        y2data = np.array([i[2] for i in dv]).astype(np.float32)
        y3data = np.array([i[3] for i in dv]).astype(np.float32)
        #print("File: ", file, " tdata: ", tdata.shape, " tstep: ", tsdata.shape, " ydata: ", ydata.shape, " y2data: ", y2data.shape)
        #print("tdata: ", tdata[0:10])
        #grouped_t.append(tdata)
        #grouped_y.append(ydata)
        #grouped_y2.append(y2data)
        #print("Finished file: "+file, "shapes:", tdata.shape, tsdata.shape, ydata.shape, y2data.shape)
        hdata = np.array([h for i in range(len(tdata))]).astype(np.float32)
        combined = np.column_stack([tdata, hdata, ydata, y2data, y3data]).astype(np.float32)
        for e in combined:
            net_data.append(e)

with ThreadPoolExecutor(max_workers=len(files)) as executor:
    for file, h in zip(files, tstep_mapping):
        executor.submit(lambda: buildLists(file, h))

#sometimes a rounding error will cause the some timesteps to not reach the correct value
#and the length of the arrays to be off by 1. just truncate to the shared values to quickly fix for now
#minlen = min([len(i) for i in grouped_t])
#grouped_t = [i for i in grouped_t]
#grouped_tstep = [i for i in grouped_tstep]
#grouped_y = [i for i in grouped_y]
#grouped_y2 = [i for i in grouped_y2]

#grouped_t_arr = np.array(grouped_t, dtype=object)
#grouped_tstep_arr = np.array(grouped_tstep)
#grouped_y_arr = np.array(grouped_y, dtype=object)
#grouped_y2_arr = np.array(grouped_y2, dtype=object)
net_array = np.array(net_data ,dtype = np.float32)

print("shapes:")
#print("t:", grouped_t_arr.shape)
#print("tstep:", grouped_tstep_arr.shape)
#print("y:", grouped_y_arr.shape)
#print("y2:", grouped_y2_arr.shape)
print("net:", net_array.shape)

#np.save("grouped_t.npy",grouped_t)
#np.save("grouped_ts.npy",np.array(grouped_tstep))
#np.save("grouped_y.npy",grouped_y)
#np.save("grouped_y2.npy",grouped_y2)
np.save("data/net_data.npy",net_array)
