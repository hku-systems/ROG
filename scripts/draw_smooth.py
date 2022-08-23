#%%
import os
import datetime
import numpy as np
from matplotlib import pyplot as plt
from csv import reader
from scipy.signal import savgol_filter
import matplotlib as mpl
import sys

case_name = sys.argv[1]
local_dic = "result/"
worker = "3"
window = 25
existing_case = []
paper = {"outdoors1":"Figure1(a)","outdoors2":"Figure1(b)","outdoors3":"Figure1(c)","outdoors4":"Figure1(d)","batchsize1":"Figure9(e)",
         "batchsize3":"Figure9(a)","batchsize4":"Figure9(c)","threshold2":"Figure10(b)","threshold3":"Figure10(a)"}
def read_csv(path):
    file_path=os.path.join(path, "accuracy.csv")
    with open(file_path, 'r') as f:
        csv_reader = reader(f)
        time = []
        step = []
        accuracy = []
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            time.append(float(row[0]))
            step.append(float(row[1]))
            accuracy.append(float(row[2]))
    return time, step, accuracy

def smooth(x, y, w_s=51):
    return savgol_filter((x,y), w_s, 3)

def plot(x, y, label, linestyle,smooth_factor):
    if smooth_factor > 0:
        x, y = smooth(x, y, smooth_factor)
        plt.plot(x, y, label=label,linestyle=linestyle,lw=2)
    else:
        plt.plot(x, y, label=label,linestyle=linestyle,lw=2)
    return max(x),max(y)

def preprocessing(line):
    lineData=line.strip().split(' ')
    return lineData
    
def read_energy(path):
    f=open(os.path.join(path, "log/worker_"+worker+".log"),"r")
    line = f.readline()    
    while line:              
        lineData=preprocessing(line) 
        if len(lineData)==4 and lineData[2] == "Worker" and lineData[3][:7] =="started":
            stringtime=lineData[0]+' '+lineData[1]
            time_object = datetime.datetime.strptime(stringtime, '%Y-%m-%d %H:%M:%S,%f')
            break
        line = f.readline() 
    f.close()
    start_time=time_object
    TOTAL, CPU_GPU_CV, SOC = 0,0,0
    f=open(os.path.join(path, "log/energy-"+worker+".txt"),"r")
    line = f.readline()    
    init_state =[0,0,0]
    count = 0
    while line:              
        lineData=preprocessing(line)
        if count < 10:
            count = count + 1
            line = f.readline() 
            continue 
        stringtime=lineData[0]+' '+lineData[1]
        time_object = datetime.datetime.strptime(stringtime, '%Y-%m-%d %H:%M:%S,%f')
        time_diff = (time_object - start_time).total_seconds()
        if time_diff<0:
            init_state = [float(lineData[5]),float(lineData[7]),float(lineData[9])]
            line =f.readline()
            continue
        TOTAL = (float(lineData[5])-init_state[0])/time_diff
        CPU_GPU_CV = (float(lineData[7])-init_state[1])/time_diff
        SOC = (float(lineData[9])-init_state[2])/time_diff
        line = f.readline() 
    f.close()
    return TOTAL,CPU_GPU_CV,SOC

def read_label(path):
    f=open(os.path.join(path, "config"),"r")
    line = f.readline()  
    while line:              
        lineData=preprocessing(line)
        if lineData[0] == "note:":
            label = line[6:]
        line = f.readline()
    f.close()  
    if "ROG" in label:
        linestyle = "-"
    else:
        linestyle = "--"
    return label,linestyle

def mul_factor(data,factor):
    fixed = []
    for i in range(len(data)):
        fixed.append(data[i]*factor)
    return fixed

def read_time(path):
    local_computation = 1.0
    network_waiting =1.0
    stall_time=0.0
    f=open(os.path.join(path, "log/worker_"+worker+".log"),"r")
    line = f.readline()  
    while line:              
        lineData=preprocessing(line) 
        if len(lineData)==10 and lineData[2] == "GPU" and lineData[6] =="avg:":
            local_computation=float(lineData[7][:-1])
        if len(lineData)==12 and lineData[2] == "communication:" and lineData[4] =="avg":
            network_waiting=float(lineData[5][:-1])
        line = f.readline()
    f.close()
    f=open(os.path.join(path, "log/worker_0.log"),"r")
    line = f.readline()    
    while line:              
        lineData=preprocessing(line) 
        if len(lineData)==16 and lineData[2] == "IMPORTANT":
            stall_time = float(lineData[10][:-1])
        line = f.readline()
    f.close()
    return local_computation,network_waiting,stall_time

local_path = [p for p in os.listdir(local_dic)]
f=open("experiment_records.txt","r+")
line = f.readline()    
while line:              
    if line[0]=="#":
        existing_case.append(line[1:-1])
    else:
        local_path.remove(line[:-1])
    line = f.readline()
if local_path == []:
    print("No new experiments were recorded.") 
    exit(0)
unique_idx = 0
while True:
    if f"{case_name}-case{unique_idx}" in existing_case:
        unique_idx += 1
        continue
    f.writelines(f"#{case_name}-case{unique_idx}\n")
    break
local_path = sorted(local_path)
for p in local_path:
    f.writelines(f"{p}\n")
f.close()
paths = []
for each_path in local_path: 
    path = os.path.join(local_dic,each_path)
    label,linestyle = read_label(path)
    paths.append([path,label,linestyle])
# %%
mpl.rcParams.update({"font.size":22,"figure.autolayout":True})
if case_name == "indoors" or case_name == "outdoors" or case_name == "batchsize":
    width=0.5
    legends = []
    local_computations = []
    network_waitings = []
    stall_times = []
    for p in paths:
        local_computation,network_waiting,stall_time = read_time(p[0])
        legends.append(p[1])
        local_computations.append(local_computation)
        network_waitings.append(network_waiting)
        stall_times.append(stall_time)
    transmission_time = []
    for i in range(len(network_waitings)):
        transmission_time.append(network_waitings[i]-stall_times[i])
    x=range(len(legends))
    plt.figure(figsize=(10,6))
    plt.xticks([index  for index in x], legends,rotation=30)
    plt.bar(x, local_computations, width=width,label='Com-\nputa-\ntion')
    plt.bar(x, transmission_time, width=width,bottom=local_computations, label='Comm-\nunica-\ntion')
    for i in range(len(network_waitings)):
        transmission_time[i] += local_computations[i]
    plt.bar(x, stall_times, width=width,bottom=transmission_time, label='Stall')
    plt.ylabel("Seconds")
    plt.xlim(-0.5,len(legends)-0.5)
    plt.legend(bbox_to_anchor=(1.05, 1),loc=2,borderaxespad=0,labelspacing=1)
    plt.savefig(f'./figure/{case_name}-case{unique_idx}-average-time-composition-{paper[case_name+str(1)]}.pdf')
    print(f"drawing ./figure/{case_name}-case{unique_idx}-average-time-composition-{paper[case_name+str(1)]}.pdf")
#%%
mpl.rcParams.update({"font.size":16,"figure.autolayout":True})
max_x=[]
max_y=[]
if case_name == "indoors" or case_name == "outdoors" or case_name=="threshold":
    plt.figure()
    for p in paths:
        path, label, linestyle= p
        time, step, accuracy = read_csv(path)
        x,y=plot(step, accuracy, label, linestyle,window)
        max_x.append(x)
        max_y.append(y)

    plt.ylabel("Training Accuracy")
    plt.legend(loc=4,prop={'size':12})
    plt.ylim(52,max(70,max(max_y)+1))
    plt.xlim(0,max(1000,min(max_x)))
    plt.xlabel("Iteration")
    plt.savefig(f'./figure/{case_name}-case{unique_idx}-statistical-efficiency-{paper[case_name+str(2)]}.pdf')
    print(f"drawing ./figure/{case_name}-case{unique_idx}-statistical-efficiency-{paper[case_name+str(2)]}.pdf")
#%%
from matplotlib.pyplot import MultipleLocator
plt.figure()
max_x = []
max_y = []
for p in paths:
    path, label, linestyle= p
    time, step, accuracy = read_csv(path)
    x,y=plot(time, accuracy, label, linestyle,window)
    max_x.append(x)
    max_y.append(y)
plt.ylabel("Training Accuracy")
plt.legend(loc=4,prop={'size':12})
plt.xlabel("Wall-clock Time (s)")
x = MultipleLocator(1800) 
ax = plt.gca()
ax.xaxis.set_major_locator(x)
if case_name == "threshold":
    max_x = max(10000,min(max_x))
else:
    max_x = max(5400,min(max_x))
for i in range(int(max_x/1800)):
    plt.vlines(1800*(i+1), 50, 100,linestyle='--',color="black")
plt.xlim(0,max_x)
plt.ylim(52,max(70,max(max_y)+1))
plt.savefig(f'./figure/{case_name}-case{unique_idx}-training-accuracy-against-wall-clock-time-{paper[case_name+str(3)]}.pdf')
print(f"drawing ./figure/{case_name}-case{unique_idx}-training-accuracy-against-wall-clock-time-{paper[case_name+str(3)]}.pdf")
#%%
if case_name == "indoors" or case_name == "outdoors" or case_name == "batchsize":
    max_x=[]
    max_y=[]
    plt.figure()
    for p in paths:
        path, label, linestyle = p
        time, step, accuracy = read_csv(path)
        TOTAL, CPU_GPU_CV, SOC = read_energy(path)
        x,y = plot(mul_factor(time,TOTAL), accuracy, label,linestyle, window)
        max_x.append(x)
        max_y.append(y)
    plt.ylabel("Training Accuracy")
    plt.legend(loc=4,prop={'size':12})
    plt.xlabel("Energy Consumption (J)")
    plt.hlines(64, 0, 100000,linestyle='--',color="black")
    plt.ylim(52,max(70,max(max_y)+1))
    plt.xlim(0,max(50000,min(max_x)))
    plt.savefig(f'./figure/{case_name}-case{unique_idx}-energy-consumption-against-training-accuracy-{paper[case_name+str(4)]}.pdf')
    print(f"drawing ./figure/{case_name}-case{unique_idx}-energy-consumption-against-training-accuracy-{paper[case_name+str(4)]}.pdf")
# %%
