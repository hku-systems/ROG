import sys
import time
import os
import random
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

record_path = sys.argv[1]
mode = sys.argv[2]
seed = int(sys.argv[3])
seg_len = 5*60

scale = {'leader': 1.0, 'worker': 0.7}[mode]

random.seed(seed)

# find wireless NIC
wnic_name = None
with open('/proc/net/dev') as f:
    f_iter = iter(f)
    next(f_iter)
    next(f_iter)
    for line in f_iter:
        nic_name = line.split()[0][:-1]
        if nic_name[:2] == 'wl':
            assert wnic_name is None
            wnic_name = nic_name
print(f'wnic_name {wnic_name}')
os.system(f'tc qdisc del dev {wnic_name} root')
os.system(f'tc qdisc del dev {wnic_name} ingress')

# read bandwidth record
bw_record = []  # format: [(time: float second, bw: float Mbps)]
with open(record_path) as f:
    for line in f:
        try:
            bw_record.append(list(map(float, line.split())))
        except Exception as e:
            print(e)

def random_stop():
    num = random.randint(1, 20)
    if num > 10:
        stop_time = random.randint(10, 90)
        print(f'Stop for {stop_time} seconds')
        time.sleep(stop_time)

def set_bandwidth(nic_name, bw):
    logging.info(f'{bw}')
    cmd = f'bash ./limit_bandwidth.sh {wnic_name} {int(bw*1024)} {int(bw*1024)}'
    os.system(cmd)

import atexit
def exit_handler():
    os.system(f'tc qdisc del dev {wnic_name} root')
    os.system(f'tc qdisc del dev {wnic_name} ingress')
atexit.register(exit_handler)

# replay the bandwidth
# to approximate real-world randomness, we randomly select seg_len segment to replay
cmd = f'bash ./limit_bandwidth.sh {wnic_name}'
os.system(cmd)

idx = int(len(bw_record)/6.0*(seed-2))
if idx >= len(bw_record)-2:
    idx = len(bw_record)-3
while True:
    set_bandwidth(wnic_name, bw_record[idx][1]*scale)
    time.sleep((bw_record[idx+1][0] - bw_record[idx][0]))
    idx = idx + 1
    if idx >= len(bw_record) - 2:
        idx = 0

