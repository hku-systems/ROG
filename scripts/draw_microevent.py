
# %%
from itertools import count
import os
import datetime
import numpy as np
from matplotlib import pyplot as plt
from csv import reader
import sys
path = sys.argv[1]
worker_idx = sys.argv[2]
start_idx_time = float(sys.argv[3])
assert "ROG" in path,"This is not a ROG experiment result"
threshold = path
for _ in range(5):
    idx = threshold.find("-")
    threshold = threshold[idx+1:]
idx = threshold.find("-")
threshold=int(threshold[:idx])
def preprocessing(line):
    lineData=line.strip().split(' ')
    return lineData
def read_bw(path):
    f=open(os.path.join(path, "log/bw_replay-"+worker_idx+".txt"),"r")
    line = f.readline()   
    bw=[]
    bw_time=[]
    while line:              
        lineData=preprocessing(line) 
        if len(lineData) == 3:
            stringtime=lineData[0]+' '+lineData[1]
            try:
                time_object = datetime.datetime.strptime(stringtime, '%Y-%m-%d %H:%M:%S,%f')
                bw.append(float(lineData[2]))
                bw_time.append(time_object)
            except:
                pass
        line = f.readline() 
    return bw,bw_time
def read_transmission_rate(path):
    f=open(os.path.join(path, "log/worker_"+worker_idx+".log"),"r")
    line = f.readline()    
    tran=[]
    tran_time=[]
    while line:              
        lineData=preprocessing(line) 
        try:
            if len(lineData)==6 and lineData[3]=="transmission" and lineData[4]=="rate":
                stringtime=lineData[0]+' '+lineData[1]
                time_object = datetime.datetime.strptime(stringtime, '%Y-%m-%d %H:%M:%S,%f')
                tran.append(float(lineData[5])*100)
                tran_time.append(time_object)
        except:
            pass
        line = f.readline()
    if len(tran_time)>len(tran):
        del(tran_time[-1])
    return tran,tran_time
def set_start_time(data,start_time):
    fixed_data=[]
    for time_object in data:
        fixed_data.append((time_object - start_time).total_seconds())
    return fixed_data
def read_staleness(path):
    f=open(os.path.join(path, f"log/worker_{worker_idx}.log"),"r")
    line = f.readline()
    rank = 10
    while line:              
        lineData=preprocessing(line) 
        if len(lineData)==9 and lineData[6]=="rank":
            rank = int(lineData[8])
            break
        line = f.readline()
    f.close()
    f=open(os.path.join(path, "log/worker_0.log"),"r")
    line = f.readline()    
    staleness=[]
    staleness_time=[]
    i=0
    while line:              
        lineData=preprocessing(line) 
        if len(lineData)==8 and lineData[2]=="2" and lineData[7]=="after":
            stringtime=lineData[0]+' '+lineData[1]
            time_object = datetime.datetime.strptime(stringtime, '%Y-%m-%d %H:%M:%S,%f')
            # staleness.append(float(lineData[5])*100)
            tag=[int(lineData[3][1:-1]),int(lineData[4][:-1]),int(lineData[5][:-1]),int(lineData[6][:-1])]
            staleness_time.append(time_object)
            staleness.append(max(tag)-tag[rank])
        line = f.readline()
        i=i+1
    return staleness,staleness_time        

def get_start_time(path):
    f=open(os.path.join(path, f"log/worker_0.log"),"r")
    line = f.readline()
    while line:              
        lineData=preprocessing(line) 
        if len(lineData)==5 and lineData[2]=="start" and lineData[3]=="parameter" and lineData[4]=="server":
            stringtime=lineData[0]+' '+lineData[1]
            start_time_server = datetime.datetime.strptime(stringtime, '%Y-%m-%d %H:%M:%S,%f')
            break
        line = f.readline()
    f.close()
    f=open(os.path.join(path, f"log/worker_{worker_idx}.log"),"r")
    line = f.readline()
    while line:              
        lineData=preprocessing(line) 
        if len(lineData)==9 and lineData[6]=="rank":
            stringtime=lineData[0]+' '+lineData[1]
            start_time_worker = datetime.datetime.strptime(stringtime, '%Y-%m-%d %H:%M:%S,%f')
            break
        line = f.readline()
    f.close()
    return start_time_server,start_time_worker

bw,bw_time = read_bw(path)
staleness,staleness_time=read_staleness(path)
mta,mta_time = read_transmission_rate(path)
start_time_server,start_time_worker = get_start_time(path)
bw_time = set_start_time(bw_time,start_time_worker)
mta_time = set_start_time(mta_time,start_time_worker)
staleness_time=set_start_time(staleness_time,start_time_server)

#%%
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.rcParams.update({"font.size":16,"figure.autolayout":True})


plt.figure(figsize=(6, 3))
host = host_subplot(111, axes_class=axisartist.Axes)
plt.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

par2.axis["right"] = par2.new_fixed_axis(loc="right", offset=(60, 0))

par1.axis["right"].toggle(all=True)
par2.axis["right"].toggle(all=True)

p1, = host.plot(bw_time,bw, linewidth=2,label="Bandwidth ")
p2, = par1.plot(mta_time, mta,  "o-", markersize=4,linewidth=2,label="Transmission\n      Rate")
p3, = par2.plot(staleness_time, staleness,"o-",  markersize=4,linewidth=2,label="Staleness")

host.set_xlim(start_idx_time, start_idx_time+200)   # 坐标轴长度
host.set_ylim(0, 130)
par1.set_ylim(0, 120)
par2.set_ylim(-0.5, threshold+0.5)

host.set_xlabel("Time (S)")
host.set_ylabel("Bandwidth (Mbps)")
par1.set_ylabel("Transmission Rate (%)")
par2.set_ylabel("Staleness")

# host.legend(loc=9)

host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
par2.axis["right"].label.set_color(p3.get_color())

plt.savefig('./figure/microevent-Figure8.pdf')
print("drawing ./figure/microevent-Figure8.pdf")
# %%
