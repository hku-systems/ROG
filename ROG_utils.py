import os

# from torch import tensor
# from torch.distributed.distributed_c10d import recv
import socket
import queue
import random

import threading
import pickle
import multiprocessing as mp
import time
from utils import AverageMeter, accuracy, mkdir_p, savefig
import torch
# from math import cos, pi
import numpy as np
from DEFSGDM.DEFSGDM import compress_cost, decompress_cost, serialize, deserialize
import zlib
from torch.optim.lr_scheduler import MultiStepLR
# from tqdm import tqdm
import math
from operator import itemgetter

import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
# import datetime
#from sympy import *
rows_per_transmission=1000
PRINT_LOCK = threading.Lock()
MAX_RECV_SIZE=2*1024*1024
MTA_static=[1.0,1.0,0.5,0.38197,0.32,0.27551,
            0.24512,0.22191,0.203456,0.188348,0.175699,
            0.164921,0.155602,0.147449,0.140243,0.133819,
            0.128049,0.122833,0.11809,0.113755,0.109774,
            0.106105,0.102708,0.0995547,0.096617,0.0938728,
            0.0913025,0.0888893,0.0866184,0.084477,0.0824538,
            0.0805386,0.0787228,0.0769982,0.075358,0.0737957,
            0.0723056,0.0708827,0.0695222,0.0682199,0.066972]
Gradient_Value_static = [0.0,0.0,0.0,0.0,0.0,0.03,
                  0.05,0.06,0.05,0.07,0.075,
                  0.085,0.09,0.095,0.10,0.102,
                  0.104,0.106,0.107,0.108,0.11,
                  0.11,0.111,0.111,0.112,0.112,
                  0.112,0.113,0.113,0.113,0.113,
                  0.114,0.114,0.114,0.114,0.114,
                  0.115,0.115,0.115,0.115,0.115]
Gradient_Value=0.0

old_print = print

def print(*args, **kwargs):
    with PRINT_LOCK:
        return old_print(*args, **kwargs)

def MTA(threshold):
    global Gradient_Value
    ## (1-P)**(S-1)=P
    if threshold<len(MTA_static):
        Gradient_Value = Gradient_Value_static[threshold]
        return MTA_static[threshold]
    else:
        print("unsupported threshold!",flush=True)
        exit(0)

def freeze_bn(m):
    if isinstance(m, torch.nn.LayerNorm) or isinstance(m, torch.nn.BatchNorm2d):
        m.training = False
        m.track_running_stats = True

TCP_NUM_SIZE=4
class TCPMessageStream:
    NUM_SIZE=4
    BYTES_LEAST = 4
    BYTES_ID_DEFAULT = 4
    BYTES_TIMEOUT = 4
    def __init__(self,sock:socket.socket):
        self.sock=sock
        self.send_queue=queue.Queue()
        self.finish_queue=queue.Queue()
        self.recv_buffer = bytearray()
        self.timeout_recv_buffer = bytearray()
        self.total_recved = 0
        self.timeout_send_meter = AverageMeter()
        self.blocking_send_meter = AverageMeter()
        self.send_lock = threading.Lock()
        self.recv_lock = threading.Lock()
        self.sock.setblocking(True)

    def set_timeoutsock(self, timeout_sock:socket.socket):
        self.timeout_sock = timeout_sock

    def get_unique_id_bytes(self, msg, length):
        s = bytes(os.urandom(length))
        attempt_times = 0
        while msg.find(s) >= 0:
            s = bytes(os.urandom(length))
            attempt_times += 1
            if attempt_times % 10 == 0:
                length += 1
                print('Warning attempt to get unique id for too many times')
        return s

    def send_with_timeout(self, msg, extra_msg,timeout, least_bytes, info=''):
        assert msg is not None
        stime = time.time()
        msize = len(msg)
        if timeout <= 0.0:
            timeout = 0.0
            msg = msg[: least_bytes]
        else:
            timeout = int(timeout * 10) / 10.
        bytes_id = self.get_unique_id_bytes(msg, self.BYTES_ID_DEFAULT)
        headers = int(least_bytes).to_bytes(self.BYTES_LEAST, "big") + int(len(msg)).to_bytes(self.NUM_SIZE, "big") + int(timeout * 10).to_bytes(self.BYTES_TIMEOUT, 'big') + int(len(extra_msg)).to_bytes(self.NUM_SIZE, "big") + extra_msg + bytes_id
        assert least_bytes <= len(msg), f"least {least_bytes} total {len(msg)}"
        # The bytes representing the end of the transmission
        self.send(headers)
        t_report_header = time.time()
        if timeout <= 0.0:
            _msize = len(msg)
            msg = _msize.to_bytes(self.NUM_SIZE, "big") + msg
            self.sock.sendall(msg)
            t_timeout_send = time.time()
            transmitted_bytes = least_bytes
        else:
            self.timeout_sock.settimeout(timeout)
            try:
                self.timeout_sock.sendall(bytes_id + msg + bytes_id)
                t_timeout_send = time.time()
            except socket.timeout:
                t_timeout_send = time.time()
            transmitted_bytes = pickle.loads(self.recv())
        self.timeout_sock.settimeout(None)

        t_get_transmitted = time.time()
        print(info, f'Sending with timeout {timeout:.2E} report_header {t_report_header-stime:.2f} timeout_send {t_timeout_send-t_report_header:.2f} get_transmitted {t_get_transmitted-t_timeout_send:.2f} \n First transmitted {transmitted_bytes} total {msize} least {least_bytes} {bytes_id}')
        if t_get_transmitted - t_timeout_send > 0.5:
            print('Warning: send end bytes takes too long')

        # Send remaining
        if transmitted_bytes < least_bytes:
            self.send(msg[transmitted_bytes: least_bytes])
            transmitted_bytes = least_bytes
        t_send_remaining = time.time()
        print(info, f'Sending with timeout {timeout} send_remaining {t_send_remaining-t_get_transmitted:.2f} Finally transmitted {transmitted_bytes} least {least_bytes}')
        return transmitted_bytes

    def recv_with_timeout(self, info=''):
        stime = time.time()
        # The ending bytes of this transmission
        headers = self.recv(info)
        t_get_headers = time.time()
        least_bytes = int.from_bytes(headers[: self.BYTES_LEAST], 'big')
        if least_bytes == 0:
            print("recv early break")
            return headers[self.BYTES_LEAST:],[],False
        msize = int.from_bytes(headers[self.BYTES_LEAST: self.BYTES_LEAST+self.NUM_SIZE], 'big')
        timeout = int.from_bytes(headers[self.BYTES_LEAST+self.NUM_SIZE: self.BYTES_LEAST+self.NUM_SIZE+self.BYTES_TIMEOUT], 'big') / 10.   # to keep decimal fraction part of the timeout
        extra_msize = int.from_bytes(headers[self.BYTES_LEAST+self.NUM_SIZE+self.BYTES_TIMEOUT: self.BYTES_LEAST+2*self.NUM_SIZE+self.BYTES_TIMEOUT], 'big')
        extra_msg = headers[self.BYTES_LEAST+2*self.NUM_SIZE+self.BYTES_TIMEOUT:self.BYTES_LEAST+2*self.NUM_SIZE+self.BYTES_TIMEOUT+extra_msize]
        bytes_id = headers[self.BYTES_LEAST+2*self.NUM_SIZE+self.BYTES_TIMEOUT+extra_msize:]
        assert least_bytes <= msize, f'least_bytes {least_bytes} total bytes {msize}'

        self.timeout_sock.settimeout(None)
        if timeout <= 0.0:
            self.timeout_recv_buffer = self.timeout_recv_buffer[:0]
            self.timeout_recv_buffer += self.recv(info)
            transmitted_bytes = len(self.timeout_recv_buffer)
            t_timeout_recv = time.time()
        else:
            # get real start position
            while self.timeout_recv_buffer.find(bytes_id) < 0:
                self.timeout_recv_buffer += self.timeout_sock.recv(1024)
            start = self.timeout_recv_buffer.find(bytes_id)
            self.timeout_recv_buffer = self.timeout_recv_buffer[start + len(bytes_id):]
            # recv send_with_timeout; can break due to either complete transmission or timeout
            start_pos = -100 * len(bytes_id)
            while self.timeout_recv_buffer.find(bytes_id, start_pos) < 0:
                remain_time = (t_get_headers + timeout) - time.time()
                if remain_time <= 0.:
                    break
                self.timeout_sock.settimeout(remain_time)
                try:
                    self.timeout_recv_buffer += self.timeout_sock.recv(1024)
                except socket.timeout:
                    break
            self.timeout_sock.settimeout(None)
            self.timeout_recv_buffer = self.timeout_recv_buffer[: msize]

            transmitted_bytes = len(self.timeout_recv_buffer)
            t_timeout_recv = time.time()
            self.send(pickle.dumps(transmitted_bytes))
        # report the transmitted bytes to the sender to decide whether to transmit remaining bytes
        t_report_transmitted = time.time()
        print(info, f'recv with timeout {timeout} get_id {t_get_headers - stime:.2f} timeout_recv {t_timeout_recv-t_get_headers:.2f} report_transmitted {t_report_transmitted - t_timeout_recv:.2f} transmitted {transmitted_bytes} least {least_bytes} {bytes_id}')

        if t_report_transmitted - t_timeout_recv > 0.5:
            print('Warning: send transmitted bytes takes too long')
        # assert (t_report_transmitted - stime) < t_timeout_recv - stime + 2, 'Warning: report transmitted takes too long'
        if transmitted_bytes < least_bytes:
            while len(self.timeout_recv_buffer) < least_bytes:
                self.timeout_recv_buffer += self.recv()
            assert len(self.timeout_recv_buffer) == least_bytes
            transmitted_bytes = least_bytes

        # parse the received bytes
        recv_objects = []
        remaining_bytes = transmitted_bytes
        while True:
            if remaining_bytes < self.NUM_SIZE:
                break
            _msize = int.from_bytes(self.timeout_recv_buffer[: self.NUM_SIZE], "big")
            self.timeout_recv_buffer = self.timeout_recv_buffer[self.NUM_SIZE:]
            remaining_bytes -= self.NUM_SIZE
            if remaining_bytes < _msize:
                break
            recv_objects.append(self.timeout_recv_buffer[:_msize])
            self.timeout_recv_buffer = self.timeout_recv_buffer[_msize:]
            remaining_bytes -= _msize
        if remaining_bytes > 0:
            self.timeout_recv_buffer = self.timeout_recv_buffer[remaining_bytes:]
        if timeout > 0.0:
            self.total_recved += transmitted_bytes + 2 * len(bytes_id)
        print(info, f'recv with timeout Finally {time.time() - stime:.2f} transmitted {transmitted_bytes} {len(recv_objects)}')
        return extra_msg, recv_objects, True

    def send_iscomplete(self, count):
        return True

    def send(self, msg, add_head=True, info=''):
        if add_head:
            msize = len(msg)
            msg = msize.to_bytes(self.NUM_SIZE, "big") + msg
        sent = 0
        total = len(msg)
        while sent < total:
            sent += self.sock.send(msg)
        # if len(msg) < 200:
        #     try:
        #         print(info, 'socket send', pickle.loads(msg))
        #     except:
        #         try:
        #             print(info, 'socket send', pickle.loads(msg[self.NUM_SIZE:]))
        #         except:
        #             print(info, 'socket send', msg)
        return sent
    
    def special_send(self,msg,info=''):
        self.send(int(0).to_bytes(self.BYTES_LEAST, "big")+msg)

    def recv(self, info=''):
        while len(self.recv_buffer) < self.NUM_SIZE:
            self.recv_buffer += self.sock.recv(MAX_RECV_SIZE)
        msize = int.from_bytes(self.recv_buffer[:self.NUM_SIZE], "big")
        self.recv_buffer = self.recv_buffer[self.NUM_SIZE:]
        while len(self.recv_buffer) < msize:
            self.recv_buffer += self.sock.recv(MAX_RECV_SIZE)
        msg = self.recv_buffer[:msize]
        self.recv_buffer = self.recv_buffer[msize:]
        self.total_recved += msize + self.NUM_SIZE
        # if msize < 200:
        #     try:
        #         print(info, 'socket recv', msize, pickle.loads(msg))
        #     except:
        #         print(info, 'socket recv', msize, msg)
        return msg

class row:
    def __init__(self, idx,start_pos_total,start_pos_layer,length):
        self.idx=idx
        self.start_pos=start_pos_total
        self.end_pos=start_pos_total+length
        self.compress_start_pos=math.floor(self.start_pos/8.0)
        self.compress_end_pos=math.floor((self.end_pos-1)/8.0)+1
        self.layer_start_pos=start_pos_layer
        self.layer_end_pos=start_pos_layer+length
    
def layer_unit(tensor,start_idx,start_pos):
    rows=[]
    if len(tensor.shape) == 1 or tensor.numel() < 1000:
        rows.append(row(start_idx,start_pos,0,tensor.numel()))
        return rows,start_idx+1,start_pos+tensor.numel()
    else:
        length=int(tensor.numel()/tensor.shape[0])
        for i in range(tensor.shape[0]):
            rows.append(row(start_idx+i,start_pos+i*length,i*length,length))
        return rows,start_idx+tensor.shape[0],start_pos+tensor.numel()


class ROG_Parameter_Server:
    def __init__(self, args, model, layer_info, communication_library, device,optimizer,compression_enable=True):
        self.world_size = args.world_size -1
        self.threshold = args.threshold
        self.model = model
        self.lock = threading.Lock()
        self.training_step=[0 for _ in range(self.world_size)]
        self.transmission_step=[0 for _ in range(self.world_size)]
        self.stall_count = [0 for _ in range(self.world_size)]
        self.stall_time = [AverageMeter() for _ in range(self.world_size)]
        self.min_step = [mp.Queue(maxsize=1) for _ in range(self.world_size)]
        self.communication_library=communication_library
        self.device = device
        self.COMPRESSION = compression_enable
        if self.COMPRESSION:
            assert self.communication_library == 'rog'
        self.updated = True
        self.recved = True
        self.optimizer = optimizer
        self.MTA_transmission_time_sum=0.0
        self.MTA_transmission_time_count = 0
        self.recent_transmission_time=np.array([0.0 for _ in range(30)])
        self.rows_number=0
        for i in range(len(layer_info)):
            self.rows_number+=len(layer_info[i])
        logging.info(f"rows number: {self.rows_number}")
        self.MTA_threshold=math.ceil(MTA(self.threshold)*self.rows_number)-1
        self.Gradient_Value_threshold=math.ceil(Gradient_Value*self.rows_number)-1
        logging.info(f"COMPRESSION_enabled {compression_enable}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.bind((args.ps_ip, args.ps_port))
        sock.listen(self.world_size)
        self.chkpt_dir = args.chkpt_dir

        self.layer_info=layer_info
        self.row_index=[]
        for i in range(len(layer_info)):
            for j in range(len(layer_info[i])):
                self.row_index.append((i,j))
        self.model_numel=0
        for p in self.model.parameters():
            self.model_numel+=p.numel()
        self.row_states=[np.array([0 for _ in range(self.world_size)]) for _ in range(self.rows_number)]
        self.row_states_per_worker=[[np.array([0 for _ in range(self.world_size)]) for _ in range(self.rows_number)]for _ in range(self.world_size)]
        self.temp_row_states_per_worker=[[np.array([0 for _ in range(self.world_size)]) for _ in range(self.rows_number)]for _ in range(self.world_size)]
        self.row_importance=[[0.0 for _ in range(self.rows_number)] for _ in range(self.world_size)]
        self.row_importance_idx=[[0 for _ in range(self.rows_number)] for _ in range(self.world_size)]
        self._stop_event = threading.Event()
        proc = []
        t = threading.Thread(target=self.checkpoint_per_period, daemon=True)
        proc.append(t)
        self.sleep_time_lock = threading.Lock()
        self.print_lock = threading.Lock()
        self.must_update=[mp.Queue() for i in range(self.world_size)]
        self.isupdate=[[mp.Queue() for i in range(self.world_size)] for _ in range(self.world_size)]
        self.computation_compression = np.zeros([self.world_size, 3])
        self.client_addresses = []
        self.total_client_stream = []
        for i in range(self.world_size):
            client_sock, client_address = sock.accept()
            client_stream = TCPMessageStream(client_sock)
            self.total_client_stream.append(client_stream)
            t = threading.Thread(target=self.each_parameter_server,args=(client_stream,client_address,i), daemon=True)
            proc.append(t)
            self.client_addresses.append(client_address[0])

        logging.info(f'client address {self.client_addresses}')
        timeout_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        timeout_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # timeout_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)  # 4k
        # timeout_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # no delay
        timeout_sock.bind((args.ps_ip, args.ps_port + 11))
        timeout_sock.listen(self.world_size)
        for s in self.total_client_stream:
            s.send(pickle.dumps('ok'))
            client_timeout_sock, client_address = timeout_sock.accept()
            idx = self.client_addresses.index(client_address[0])
            client_timeout_sock.send(pickle.dumps(self.client_addresses[idx]))
            self.total_client_stream[idx].set_timeoutsock(client_timeout_sock)
            self.client_addresses[idx]='0.0.0.0'

        for t in proc:
            t.start()
        logging.info("start parameter server")
        for t in proc:
            t.join()
        sock.close()

    def suggest_sleep_time(self, l, rank):
        # first tune the batch size of the slow devices (small batchszie)
        # to take a bit less time than fast devices (large batchsize) to finish computation and compression,
        # then use this function to make the slow devices the sleep for certain time
        # computation_compression: [[compute&compress time, sleep time, batchsize]]
        self.sleep_time_lock.acquire()
        arr = self.computation_compression
        arr[rank] = np.array(l)
        # if some workers have not sent its batchsize
        if np.any(self.computation_compression[:, 2] <= 0.):
            self.sleep_time_lock.release()
            return
        # largest batchsize
        fast = arr.argmax(axis=0)[2]
        # same batchsize can appear
        all_fast = np.where(arr[:, 2] == arr[fast, 2])[0]
        # find the one with largest compute&compress time
        fast = all_fast[np.argmax(arr[:, 0][all_fast])]
        if arr[fast, 0] < np.max(arr[:, 0]):
            logging.info(f'Warning: slow devices are still taking longer time to compute and compress;\narray {arr}')
            slowest = np.argmax(arr[:, 0])
            arr[:, 1] = arr[slowest, 0] - arr[:, 0]
        else:
            arr[:, 1] = arr[fast, 0] - arr[:, 0]
        self.sleep_time_lock.release()

    def decide_row_importance_metric(self,rank,scales):
        gradeint_value_important = [0 for _ in range(self.rows_number)]
        gradient_values=[]
        # tag=(self.threshold+1)*self.world_size
        for i in range(len(self.layer_info)):
            gradient_values.append(scales[0][i].item())
        factor = max(gradient_values)+1
        # iteration=self.training_step[rank]
        for i in range(len(self.layer_info)):
            gradient_value=gradient_values[i]/factor
            for j in range(len(self.layer_info[i])):
                idx=self.layer_info[i][j].idx
                # tag1=iteration-min(self.row_states_per_worker[rank][idx])
                # if tag1 >= self.threshold:
                #     self.row_importance[rank][idx] = float("inf")
                # else:
                tag2=sum(self.temp_row_states_per_worker[rank][idx]-self.row_states_per_worker[rank][idx])
                # if tag2==0:
                #     freshness=0
                # else:
                freshness=tag2
                self.row_importance[rank][idx]=freshness+gradient_value
                gradeint_value_important[idx] = gradient_value

        MTA_part_idx,_=zip(*sorted(enumerate(self.row_importance[rank]),key=itemgetter(1),reverse=True))
        MTA_part_idx = list(MTA_part_idx[:self.MTA_threshold])
        for idx in MTA_part_idx:
            gradeint_value_important[idx]=-1
        Gradient_Value_part_idx,_=zip(*sorted(enumerate(gradeint_value_important),key=itemgetter(1),reverse=True))
        Gradient_Value_part_idx = list(Gradient_Value_part_idx[:-self.MTA_threshold])
        self.row_importance_idx[rank] = MTA_part_idx + Gradient_Value_part_idx



    def get_most_important_row(self,sign_model,rank):
        poses = []
        last_row_idx=self.row_importance_idx[rank][0]
        start=last_row_idx
        end=last_row_idx
        idx=1
        while idx< self.rows_number:
            row_idx = self.row_importance_idx[rank][idx]
            if row_idx != last_row_idx + 1:
                poses.append((start,end))
                start = row_idx
            end = row_idx
            last_row_idx = row_idx
            idx += 1
        poses.append((start,end))
      
        count=0
        whole_model=None
        least_bytes = 0
        compressed_idx = []
        for i,pair in enumerate(poses):
            length =  pair[1]-pair[0] + 1
            start_idx = pair[0]
            pos=self.row_index[start_idx]
            pos=self.layer_info[pos[0]][pos[1]]
            start_pos=pos.compress_start_pos
            end_idx = pair[1]
            pos=self.row_index[end_idx]
            pos=self.layer_info[pos[0]][pos[1]]
            end_pos=pos.compress_end_pos    
            sign_row=sign_model['param'][start_pos:end_pos]
        
            data=pickle.dumps(sign_row, protocol=4)
            msize=len(data)
            if whole_model == None:
                whole_model = msize.to_bytes(TCP_NUM_SIZE,"big") + data
            else:
                whole_model += msize.to_bytes(TCP_NUM_SIZE,"big") + data
            compressed_idx.append((count+length,len(whole_model)))
            if count+length >= self.MTA_threshold + self.Gradient_Value_threshold and least_bytes == 0:
                least_bytes = len(whole_model)
            count += length
        return whole_model, compressed_idx,least_bytes,poses
    
    def update_transmission_time(self,transmission_time):
        with self.lock:
            self.recent_transmission_time[self.MTA_transmission_time_count%len(self.recent_transmission_time)]=transmission_time
            self.MTA_transmission_time_sum+=transmission_time
            self.MTA_transmission_time_count+=1

    def decide_mta_transmission_time(self,rank):
        with self.lock:
            if 0 in self.transmission_step:
                return 5
            slowest_idx = np.argmin(self.transmission_step)
            times_diff = self.transmission_step[rank] - self.transmission_step[slowest_idx]
            if times_diff < 2:
                return 0
            else:
                tag1 = np.max(self.recent_transmission_time)
                tag2 = self.MTA_transmission_time_sum/self.MTA_transmission_time_count
                return max(tag1, tag2)
            # slowest_idx = np.argmin([self.transmission_step[i] for i in range(self.world_size)])
            # fastest_idx = np.argmax([self.transmission_step[i] for i in range(self.world_size)])
            # times_diff = self.transmission_step[rank] - self.transmission_step[slowest_idx]
            # transmission_time_diff = self.MTA_transmission_time[rank].sum - self.MTA_transmission_time[slowest_idx].sum
            # if times_diff == 0:
            #     if abs(transmission_time_diff) < self.MTA_transmission_time[slowest_idx].avg:
            #         # This worker is straggler itself
            #         if self.transmission_step[rank] < self.transmission_step[fastest_idx]:
            #             return 0
            #         # no straggler
            #         else:
            #             return self.MTA_transmission_time[slowest_idx].avg
            #     # this worker risks straggling
            #     elif transmission_time_diff > self.MTA_transmission_time[slowest_idx].avg:
            #         return 0
            #     else:
            #         return self.MTA_transmission_time[slowest_idx].avg * 2
            # else:
            #     return self.MTA_transmission_time[slowest_idx].avg * 2**times_diff
    def must_update_early_break(self,rank,client_stream: TCPMessageStream):
        must_update = self.ask_for_new_rows(rank)
        if must_update == []:
            logging.info(f"{rank} really empty")
            return
        logging.info(f"{rank} must update early break, {len(must_update)}")
        client_stream.special_send(zlib.compress(pickle.dumps(("must_update",must_update), protocol=4)))
        param_group,recv_shape,sign_row=pickle.loads(client_stream.recv(rank))
        recv_model=np.zeros(recv_shape,dtype=np.uint8)
        version=self.training_step[rank]
        count=0
        start_idx=0
        while start_idx < len(sign_row):
            row_idx = must_update[count]
            count += 1
            self.row_states[row_idx][rank]=version
            pos=self.row_index[row_idx]
            pos=self.layer_info[pos[0]][pos[1]]
            recv_model[pos.compress_start_pos:pos.compress_end_pos]=sign_row[start_idx:start_idx + pos.compress_end_pos- pos.compress_start_pos]
            start_idx += pos.compress_end_pos- pos.compress_start_pos
        logging.info(f"{rank} recv {count}")
        data = {
            'length': self.model_numel,
            'param': recv_model
        }
        recv_model=deserialize(data)
        sign_model=torch.zeros_like(recv_model)
        rows_pos=self.get_pos(must_update)
        for pos in rows_pos:
            sign_model[pos[0]:pos[1]]=recv_model[pos[0]:pos[1]]
        self.optimizer.recv(sign_model, param_group, rank,decompress_here=False)
        for i in range(self.world_size):
            if self.isupdate[rank][i].qsize()>0:
                try:
                    self.isupdate[rank][i].get_nowait()
                except queue.Empty:
                    pass
            self.isupdate[rank][i].put("ok")
        logging.info(f"{rank} early break complete")

    def check_threshold(self,rank,client_stream: TCPMessageStream):
        check_list=[]
        waiting_for_new=[[] for _ in range(self.world_size)]
        threshold=self.threshold
        iteration=self.training_step[rank]
        for i in range(self.rows_number):
            for j in range(self.world_size):
                if iteration>self.row_states[i][j]+threshold:
                    waiting_for_new[j].append(i)
        for i in range(self.world_size):
            if waiting_for_new[i]==[]:
                continue
            if self.isupdate[i][rank].qsize()>0:
                try:
                    self.isupdate[i][rank].get_nowait()
                except queue.Empty:
                    pass
            logging.info(f"now must update {i} has size {self.must_update[i].qsize()}")
            self.must_update[i].put((iteration,waiting_for_new[i]))
            logging.info(f"{rank} put {len(waiting_for_new[i])} in {i} with iteration {iteration} and size now {self.must_update[i].qsize()}")
            check_list.extend(waiting_for_new[i])
        check_list=set(check_list)
        stall_time=0.0
        if check_list!=[]:
            for idx_required in check_list:
                now=min(self.row_states[idx_required])
                while iteration > now + threshold:
                    for i in range(self.world_size):
                        while iteration>self.row_states[idx_required][i]+threshold:
                            logging.info(f"{rank} stalling on, {self.row_states[idx_required]},{i},{idx_required}")
                            logging.info(f"{rank} now must_update{rank} has size {self.must_update[rank].qsize()} and must_update{i} has size {self.must_update[i].qsize()}")
                            tag=time.time()
                            if self.must_update[rank].qsize()>0:
                                logging.info(f"{rank} is not empty and size {self.must_update[rank].qsize()} {self.must_update[rank].empty()}")
                                self.must_update_early_break(rank,client_stream)
                            else:
                                logging.info(f"{rank} is empty and size {self.must_update[rank].qsize()} {self.must_update[rank].empty()}")
                            self.isupdate[i][rank].get()
                            stall_time+=time.time()-tag
                    now=min(self.row_states[idx_required])
        with self.print_lock:
            logging.info(f"{rank} stall waiting {stall_time}")
        self.stall_time[rank].update(stall_time)
        with self.print_lock:
            logging.info(f"{rank} threshold satisfy")

    def ask_for_new_rows(self,rank):
        threshold=self.threshold
        must_update_rows=[]
        while self.must_update[rank].qsize() >0:
            logging.info(f"ask_for_new_rows: must update {rank} has size {self.must_update[rank].qsize()}")
            iteration,waiting_for_new=self.must_update[rank].get()
            logging.info(f"{rank} get {len(waiting_for_new)} {iteration}")
            for idx in waiting_for_new:
                if iteration<=self.row_states[idx][rank]+threshold:
                    continue
                must_update_rows.append(idx)
            logging.info(f"now must update size is {len(must_update_rows)} and queue size is {self.must_update[rank].qsize()} and is empty {self.must_update[rank].empty()}")
        return sorted(set(must_update_rows))

    def get_pos(self,recv_rows):
        if recv_rows==[]:
            return []
        sorted_recv_rows=sorted(set(recv_rows))
        poses=[]
        if len(sorted_recv_rows) == self.rows_number:
            poses.append((0,self.rows_number-1))
        else:
            last_row_idx=sorted_recv_rows[0]
            start=last_row_idx
            end=last_row_idx
            idx=1
            while idx<len(sorted_recv_rows):
                row_idx = sorted_recv_rows[idx]
                if row_idx != last_row_idx + 1:
                    poses.append((start,end))
                    start = row_idx
                end = row_idx
                last_row_idx = row_idx
                idx += 1
            poses.append((start,end))
        row_pos=[]
        for pair in poses:
            pos=self.row_index[pair[0]]
            pos=self.layer_info[pos[0]][pos[1]]
            start_pos=pos.start_pos
            pos=self.row_index[pair[1]]
            pos=self.layer_info[pos[0]][pos[1]]
            end_pos=pos.end_pos
            row_pos.append((start_pos,end_pos))
        return row_pos

    def fix_unfinished(self,data,transmitted_bytes,compressed_idx,rank):
        if transmitted_bytes == len(data):
            return [i for i in range(self.rows_number)]
        for i in range(len(compressed_idx)):
            if compressed_idx[i][1] == transmitted_bytes:
                at_least_idx = i    
                break
            if compressed_idx[i][1] > transmitted_bytes:
                at_least_idx = i-1
                break
        transmitted=[self.row_importance_idx[rank][i] for i in range(compressed_idx[at_least_idx][0])]
        return sorted(transmitted)
    def checkpoint_per_period(self, period=60, decay_step=20):
        step = 1
        start = time.time()
        sleep_time = 0
        while True:
            if self._stop_event.is_set():
                return
            if sleep_time > 60:
                time.sleep(60)
                sleep_time -= 60
                continue
            elif sleep_time > 0:
                time.sleep(sleep_time)
                sleep_time = 0
            if not self.recved:
                time.sleep(5)
                continue
            else:
                self.recved = False
            if self.COMPRESSION:
                self.optimizer.step()
            torch.save(self.model.state_dict(), f'{self.chkpt_dir}/{time.time() - start:8.2f}-{max(self.training_step)}.chkpt')
            sleep_time = period - (time.time() - start) % period
            step += 1
            if step % decay_step == 0:
                period *= 3
    
    def each_parameter_server(self, client_stream: TCPMessageStream, client_address, rank):
        client_stream.send(pickle.dumps(rank), True, rank)
        time.sleep(random.randint(1, 40)/10.)
        client_stream.send_iscomplete(1)
        while True:
            msg = pickle.loads(client_stream.recv(rank))
            logging.info(f"{rank} "+msg)
            if self._stop_event.is_set():
                msg = 'terminate'
            if msg[:3]=='ask':
                self.update_transmission_time(float(msg[3:]))
                self.transmission_step[rank]+=1
                
                tag=time.time()
                self.check_threshold(rank,client_stream)
                logging.info(f"{rank} ask step0 {time.time()-tag}")

                self.temp_row_states_per_worker[rank] = self.row_states
                sign_model, param_group = self.optimizer.send(rank,compress_here=True)
                logging.info(f"{rank} ask step1 {time.time()-tag}")

                self.decide_row_importance_metric(rank,param_group)
                data, compressed_idx, least_bytes,poses=self.get_most_important_row(sign_model,rank)
                logging.info(f"{rank} ask step2 {time.time()-tag}")
                
                timeout = self.decide_mta_transmission_time(rank)
                logging.info(f"{rank} timeout {timeout}")
                transmission_start_time = time.time()
                transmitted_bytes=client_stream.send_with_timeout(data, zlib.compress(pickle.dumps((param_group,sign_model["param"].shape,self.computation_compression[rank, 1],poses,timeout),protocol=4)),timeout, least_bytes=least_bytes, info=rank)
                transmission_end_time = time.time()
                self.update_transmission_time(transmission_end_time-transmission_start_time)
                self.transmission_step[rank]+=1
                logging.info(f"{rank} ask step3 transmission time {transmission_end_time-transmission_start_time} {time.time()-tag}")

                transmitted=self.fix_unfinished(data,transmitted_bytes,compressed_idx,rank)
                logging.info(f"{rank} ask step4 transmission rate {len(transmitted)/self.rows_number} {time.time()-tag}")

                idx=0
                for i,p in enumerate(self.model.parameters()):
                    for j in range(len(self.layer_info[i])):
                        if idx<len(transmitted) and self.layer_info[i][j].idx == transmitted[idx]:
                            start=self.layer_info[i][j].layer_start_pos
                            end=self.layer_info[i][j].layer_end_pos
                            p.error_per_worker[rank][start:end].copy_(p.temp_error_per_worker[rank][start:end])
                            p.grad_per_worker[rank][start:end].add_(p.temp_grad_per_worker[rank][start:end],alpha=-1.)
                            self.row_states_per_worker[rank][transmitted[idx]]=self.temp_row_states_per_worker[rank][transmitted[idx]]
                            idx+=1
                logging.info(f"{rank} compress {compress_cost.avg:.2E}; decompress {decompress_cost.avg:.2E}; recv volume {client_stream.total_recved/1024./1024.:.2f}MB")
                logging.info(f"IMPORTANT stall time avg:, {[round(t.avg,2) for t in self.stall_time]}, {round(sum([t.avg for t in self.stall_time])/len(self.stall_time),2)}, sum:, {[round(t.sum,2) for t in self.stall_time]}")
            if msg=='send':
                tag=time.time()
                extra_msg, row_groups,succeed = client_stream.recv_with_timeout(info=rank)
                logging.info(f"{rank} send step0 {time.time()-tag}")

                param_group,recv_shape,computation_compression,poses=pickle.loads(zlib.decompress(extra_msg))
                logging.info(f"{rank} send step1 {time.time()-tag}")

                recv_model=np.zeros(recv_shape,dtype=np.uint8)
                recv_rows=[]
                rows_pos = []
                version=self.training_step[rank] + 1
                count=0
                for group in row_groups: #pickle.dumps("ok")
                    data=pickle.loads(group)
                    sign_row=data 
                    start_idx=0
                    while start_idx < len(sign_row):
                        row_idx_pairs = poses[count]
                        count += 1
                        for row_idx in range(row_idx_pairs[0],row_idx_pairs[1]+1):
                            self.row_states[row_idx][rank]=version
                            recv_rows.append(row_idx)
                        pos=self.row_index[row_idx_pairs[0]]
                        pos=self.layer_info[pos[0]][pos[1]]
                        start_pos=pos.compress_start_pos
                        start_decompress = pos.start_pos
                        pos=self.row_index[row_idx_pairs[1]]
                        pos=self.layer_info[pos[0]][pos[1]]
                        end_pos=pos.compress_end_pos  
                        end_decompress = pos.end_pos
                        recv_model[start_pos:end_pos]=sign_row[start_idx:start_idx + end_pos- start_pos]
                        start_idx += end_pos- start_pos
                        rows_pos.append((start_decompress,end_decompress))
     
                with self.print_lock:
                    logging.info(f"{rank} send step2 transmission rate {len(recv_rows)/self.rows_number} {time.time()-tag}")
                
                # client_stream.send(pickle.dumps("ok"), True, rank)
                data = {
                    'length': self.model_numel,
                    'param': recv_model
                }
                recv_model=deserialize(data)
                sign_model=torch.zeros_like(recv_model)
                for pos in rows_pos:
                    sign_model[pos[0]:pos[1]]=recv_model[pos[0]:pos[1]]
                self.optimizer.recv(sign_model, param_group, rank,decompress_here=False)
                for i in range(self.world_size):
                    if self.isupdate[rank][i].qsize()>0:
                        try:
                            self.isupdate[rank][i].get_nowait()
                        except queue.Empty:
                            pass
                    self.isupdate[rank][i].put("ok")

                with self.lock:
                    logging.info(f"{rank} {self.training_step} before")
                    before = min(self.training_step)
                    self.training_step[rank] += 1
                    now = min(self.training_step)
                    logging.info(f"{rank} {self.training_step} after")
                if self.training_step[rank] == 1:
                    for i in range(self.world_size):
                        self.isupdate[i][rank].get()
                
                self.suggest_sleep_time(computation_compression, rank)

                self.recved = True
                # print(f'Fine Recv parameters from client {rank}')
                client_stream.send_iscomplete(2)
            if msg == "terminate":
                if not self.COMPRESSION:
                    self.gathered_weight.put((None,None,None))
                self._stop_event.set()
                logging.info(f"server for {rank} terminated")
                break
            logging.info(f"{rank} complete")


class ROG_Local_Worker:
    def __init__(self, args, model, layer_info,
        communication_library, device, optimizer, compression_enable=True):

        self.args = args
        self.threshold = args.threshold
        self.communication_library = communication_library
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.world_size=args.world_size-1
        self.rows_number=0
        for i in range(len(layer_info)):
            self.rows_number+=len(layer_info[i])
        logging.info(f"rows number: {self.rows_number}")
        self.row_importance=[0. for _ in range(self.rows_number)]
        self.row_importance_idx=[0 for _ in range(self.rows_number)]
        self.row_index=[]
        for i in range(len(layer_info)):
            for j in range(len(layer_info[i])):
                self.row_index.append((i,j))
        self.model_numel=0
        for p in self.model.parameters():
            self.model_numel+=p.numel()
        self.MTA_threshold=math.ceil(MTA(args.threshold)*self.rows_number)-1
        self.Gradient_Value_threshold=math.ceil(Gradient_Value*self.rows_number)-1
        self.max_MTA_transmission_time = 5.0
        self.layer_info=layer_info
        self.remain_number=[0 for _ in range(self.rows_number)]
        self.COMPRESSION = compression_enable
        self.t_communication = AverageMeter()
        self.t_computation = AverageMeter()
        self.batchsize = 0
        self.sleep_time = 0.
        self.lock = threading.Lock()
        self.losses = AverageMeter()
        if self.COMPRESSION:
            assert self.communication_library == 'rog'
        logging.info(f"Device {device}; cudnn.benchmark={torch.backends.cudnn.benchmark}; cudnn.enabled={torch.backends.cudnn.enabled}")
        # Warm up phase
        # self.train(0, 0, 0, 0.0001, local_update=args.E, warm_up=True, warm_up_iter=10)
        # print('Warm up complete')
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        time.sleep(1) # waiting for ps start
        sock.connect((args.ps_ip, args.ps_port))
        # sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = TCPMessageStream(sock)
        logging.info("conneted to parameter server")

        assert pickle.loads(self.sock.recv()) == 'ok'

        timeout_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        timeout_sock.connect((args.ps_ip, args.ps_port + 11))
        # timeout_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)  # 4k
        # timeout_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # no delay
        logging.info(pickle.loads(timeout_sock.recv(MAX_RECV_SIZE)))
        self.sock.set_timeoutsock(timeout_sock)
        logging.info("Timeout sock connected.")
        self.rank=pickle.loads(self.sock.recv())
        logging.info(f"All workers ready, my rank is {self.rank}")
        logging.info(self.optimizer.defaults)
        self.gpu_running = AverageMeter()

        self._stop_event = threading.Event()
        self.checkpoint_thread = None
        self.training_step = 0
        logging.info('Worker started')
        # print(datetime.datetime.now(), 'Worker started.')
        self.recording=[AverageMeter() for _ in range(20)] 

    def decide_row_importance_metric(self,scales):
        gradeint_value_important = [0 for _ in range(self.rows_number)]
        gradient_values=[]
        for i in range(len(self.layer_info)):
            gradient_values.append(scales[0][i][0].item())
        factor = max(gradient_values)+0.1
        for i in range(len(self.layer_info)):
            gradient_value=gradient_values[i]/factor
            for j in range(len(self.layer_info[i])):
                idx=self.layer_info[i][j].idx
                freshness=self.remain_number[idx]
                self.row_importance[idx]=freshness+gradient_value
                gradeint_value_important[idx] = gradient_value

        MTA_part_idx,_=zip(*sorted(enumerate(self.row_importance),key=itemgetter(1),reverse=True))
        MTA_part_idx = list(MTA_part_idx[:self.MTA_threshold])
        for idx in MTA_part_idx:
            gradeint_value_important[idx]=-1
        Gradient_Value_part_idx,_=zip(*sorted(enumerate(gradeint_value_important),key=itemgetter(1),reverse=True))
        Gradient_Value_part_idx = list(Gradient_Value_part_idx[:-self.MTA_threshold])
        self.row_importance_idx = MTA_part_idx + Gradient_Value_part_idx
        
        # gradient_values=[]
        # for i in range(len(self.layer_info)):
        #     gradient_values.append(scales[0][i][0].item())
        # factor = max(gradient_values)+0.1
        # for i in range(len(self.layer_info)):
        #     gradient_value=gradient_values[i]/factor
        #     for j in range(len(self.layer_info[i])):
        #         idx=self.layer_info[i][j].idx
        #         freshness=self.remain_number[idx]
        #         # if freshness >= self.threshold:
        #         #     freshness=float("inf")
        #         self.row_importance[idx]=freshness+gradient_value
        # for idx in must_update:
        #     self.row_importance[idx]=float("inf")
        # self.row_importance_idx,_=zip(*sorted(enumerate(self.row_importance),key=itemgetter(1),reverse=True))
        
    def set_adapt_noise(self,train_dl, criterion, mixup_fn):
        self.train_loader = train_dl
        self.criterion=criterion
        self.mixup_fn=mixup_fn
        self.lr_scheduler=MultiStepLR(self.optimizer, milestones=[800,1600,2400,3200], gamma=0.5)

    def get_most_important_row(self,sign_model):       
        poses = []
        last_row_idx=self.row_importance_idx[0]
        start=last_row_idx
        end=last_row_idx
        idx=1
        while idx< self.rows_number:
            row_idx = self.row_importance_idx[idx]
            if row_idx != last_row_idx + 1:
                poses.append((start,end))
                start = row_idx
            end = row_idx
            last_row_idx = row_idx
            idx += 1
        poses.append((start,end))
      
        count=0
        whole_model=None
        least_bytes = 0
        compressed_idx = []
        for i,pair in enumerate(poses):
            length =  pair[1]-pair[0] + 1
            start_idx = pair[0]
            pos=self.row_index[start_idx]
            pos=self.layer_info[pos[0]][pos[1]]
            start_pos=pos.compress_start_pos
            end_idx = pair[1]
            pos=self.row_index[end_idx]
            pos=self.layer_info[pos[0]][pos[1]]
            end_pos=pos.compress_end_pos    
            sign_row=sign_model['param'][start_pos:end_pos]
        
            data=pickle.dumps(sign_row, protocol=4)
            msize=len(data)
            if whole_model == None:
                whole_model = msize.to_bytes(TCP_NUM_SIZE,"big") + data
            else:
                whole_model += msize.to_bytes(TCP_NUM_SIZE,"big") + data
            compressed_idx.append((count+length,len(whole_model)))
            if count+length >= self.MTA_threshold + self.Gradient_Value_threshold and least_bytes == 0:
                least_bytes = len(whole_model)
            count += length
        return whole_model, compressed_idx,least_bytes,poses

    def get_pos(self,recv_rows):
        if recv_rows==[]:
            return []
        sorted_recv_rows=sorted(set(recv_rows))
        poses=[]
        if len(sorted_recv_rows) == self.rows_number:
            poses.append((0,self.rows_number-1))
        else:
            last_row_idx=sorted_recv_rows[0]
            start=last_row_idx
            end=last_row_idx
            idx=1
            while idx<len(sorted_recv_rows):
                row_idx = sorted_recv_rows[idx]
                if row_idx != last_row_idx + 1:
                    poses.append((start,end))
                    start = row_idx
                end = row_idx
                last_row_idx = row_idx
                idx += 1
            poses.append((start,end))
        row_pos=[]
        for pair in poses:
            pos=self.row_index[pair[0]]
            pos=self.layer_info[pos[0]][pos[1]]
            start_pos=pos.start_pos
            pos=self.row_index[pair[1]]
            pos=self.layer_info[pos[0]][pos[1]]
            end_pos=pos.end_pos
            row_pos.append((start_pos,end_pos))
        return row_pos

    def get_zero_pos(self,transmiited):
        zero_pos=[]
        idx=0
        for i in range(len(self.layer_info)):
            length=len(self.layer_info[i])
            layer=[]
            for j in range(length):
                if idx < len(transmiited) and self.layer_info[i][j].idx==transmiited[idx]:
                    self.remain_number[transmiited[idx]]=0
                    layer.append(1)
                    idx+=1
                else:
                    layer.append(0)
            sum_result=sum(layer)
            if sum_result == length:
                zero_pos.append(1)
            elif sum_result == 0:
                zero_pos.append(0)
            else:
                zero_pos.append(layer)
        return zero_pos

    def fix_unfinished(self,data,transmitted_bytes,compressed_idx):
        if transmitted_bytes == len(data):
            return [i for i in range(self.rows_number)]
        for i in range(len(compressed_idx)):
            if compressed_idx[i][1] == transmitted_bytes:
                at_least_idx = i    
                break
            if compressed_idx[i][1] > transmitted_bytes:
                at_least_idx = i-1
                break
        transmitted=[self.row_importance_idx[i] for i in range(compressed_idx[at_least_idx][0])]
        return sorted(transmitted)

    def must_update_early_break(self,must_update):
        logging.info(f"must update early break {len(must_update)}")
        sign_model, param_group = self.optimizer.compress(compress_here=True)
        sign_rows_per_batch=np.array([],dtype=sign_model['param'].dtype)
        for idx in must_update:
            pos=self.row_index[idx]
            pos=self.layer_info[pos[0]][pos[1]]
            sign_row=sign_model['param'][pos.compress_start_pos:pos.compress_end_pos]
            sign_rows_per_batch = np.concatenate((sign_rows_per_batch, sign_row))
        self.sock.send(pickle.dumps((param_group,sign_model["param"].shape, sign_rows_per_batch), protocol=4))
        zero_pos=self.get_zero_pos(must_update)
        for i,p in enumerate(self.model.parameters()):
            if zero_pos[i]==0:
                continue
            elif zero_pos[i]==1:
                # only update error of those sent rows
                p.remain_d_p.zero_()
            else:
                zero_tensor=torch.zeros_like(p.grad)
                for j in range(len(zero_pos[i])):
                    if zero_pos[i][j]==1:
                        p.remain_d_p[j]=zero_tensor[j]
        return 0

    def pull_model(self,transmission_time):
        logging.info("start ask for model")
        start=time.time()

        tag_start=time.time()
        self.sock.send(pickle.dumps("ask"+str(transmission_time)), True, self.rank)
        tag_end=time.time()
        self.recording[0].update(tag_end-tag_start)
        logging.info(f"pull step0 {self.recording[0].val}, avg {self.recording[0].avg}")

        tag_start=time.time()
        extra_msg, row_groups, succeed = self.sock.recv_with_timeout(info=self.rank)
        while succeed == False:
            data=pickle.loads(zlib.decompress(extra_msg))
            assert data[0] == "must_update"
            self.must_update_early_break(data[1])
            extra_msg, row_groups, succeed = self.sock.recv_with_timeout(info=self.rank)
        tag_end = time.time()
        self.recording[2].update(tag_end-tag_start)
        logging.info(f"pull step1 {self.recording[2].val}, avg {self.recording[2].avg}")

        tag_start = time.time()
        [param_group,recv_shape,self.sleep_time,poses,self.max_MTA_transmission_time] = pickle.loads(zlib.decompress(extra_msg))
        tag_end=time.time()
        self.recording[3].update(tag_end-tag_start)
        logging.info(f"pull step2 {self.recording[3].val}, avg {self.recording[3].avg}")

        tag_start=time.time()
        recv_model=np.zeros(recv_shape, dtype=np.uint8)
        recv_rows=[]
        rows_pos=[]
        count=0
        for group in row_groups: #pickle.dumps("ok")
            data=pickle.loads(group)
            sign_row=data
            start_idx=0
            while start_idx < len(sign_row):
                row_idx_pairs = poses[count]
                count += 1
                for row_idx in range(row_idx_pairs[0],row_idx_pairs[1]+1):
                    recv_rows.append(row_idx)
                pos=self.row_index[row_idx_pairs[0]]
                pos=self.layer_info[pos[0]][pos[1]]
                start_pos=pos.compress_start_pos
                start_decompress = pos.start_pos
                pos=self.row_index[row_idx_pairs[1]]
                pos=self.layer_info[pos[0]][pos[1]]
                end_pos=pos.compress_end_pos  
                end_decompress = pos.end_pos
                recv_model[start_pos:end_pos]=sign_row[start_idx:start_idx + end_pos- start_pos]
                start_idx += end_pos- start_pos
                rows_pos.append((start_decompress,end_decompress))
        tag_end = time.time()
        # assert pickle.loads(self.sock.recv(self.rank)) == 'ok'
        self.recording[4].update(tag_end - tag_start)
        logging.info(f"pull step3 {self.recording[4].val}, avg {self.recording[4].avg}")
       
        tag_start=time.time()        
        data = {
                    'length': self.model_numel,
                    'param': recv_model
                }
        recv_model=deserialize(data,device=self.device) 
        sign_model=torch.zeros_like(recv_model)
        for pos in rows_pos:
            sign_model[pos[0]:pos[1]]=recv_model[pos[0]:pos[1]]
        self.optimizer.decompress_optimize(sign_model, param_group,decompress_here=False)
        tag_end=time.time()
        self.recording[5].update(tag_end-tag_start)
        logging.info(f"pull step4 {self.recording[5].val}, avg {self.recording[5].avg}")

        end=time.time()
        logging.info(f"pull transmission rate {len(recv_rows)/self.rows_number}")
        logging.info(f"ask complete {end-start}")
        return end-start

    def push_update(self):
        logging.info("start push update")
        self.sock.send(pickle.dumps("send"), True, self.rank)
        transmission_start_time=time.time()
        
        tag1=time.time()
        sign_model, param_group = self.optimizer.compress(compress_here=True)
        tag2=time.time()
        self.recording[6].update(tag2-tag1)
        logging.info(f"push step0 {self.recording[6].val}, avg {self.recording[6].avg}")
        
        tag1=time.time()
        self.decide_row_importance_metric(param_group)
        t_compute_compress = self.t_computation.val + compress_cost.val + decompress_cost.val
        data, compressed_idx, least_bytes,poses=self.get_most_important_row(sign_model)
        tag2=time.time()
        self.recording[7].update(tag2-tag1)
        logging.info(f"push step1 {self.recording[7].val}, avg {self.recording[7].avg}")
        
        tag1 = time.time()
        transmitted_bytes=self.sock.send_with_timeout(data, zlib.compress(pickle.dumps((param_group,sign_model["param"].shape, [t_compute_compress, 0., self.batchsize],poses), protocol=4)), self.max_MTA_transmission_time, least_bytes=least_bytes, info=self.rank)
        tag2=time.time()
        transmission_time = tag2-tag1
        self.recording[8].update(tag2-tag1)
        logging.info(f"timeout is {self.max_MTA_transmission_time}")
        logging.info(f"push step2 {self.recording[8].val}, avg {self.recording[8].avg}")

        tag1 = time.time()
        transmitted=self.fix_unfinished(data,transmitted_bytes,compressed_idx)
        zero_pos=self.get_zero_pos(transmitted)
        tag2=time.time()
        self.recording[9].update(tag2-tag1)
        logging.info(f"push step3 {self.recording[9].val}, avg {self.recording[9].avg}")

        tag1=time.time()
        for i,p in enumerate(self.model.parameters()):
            if zero_pos[i]==0:
                continue
            elif zero_pos[i]==1:
                # only update error of those sent rows
                p.remain_d_p.zero_()
            else:
                zero_tensor=torch.zeros_like(p.grad)
                for j in range(len(zero_pos[i])):
                    if zero_pos[i][j]==1:
                        p.remain_d_p[j]=zero_tensor[j]
        tag2=time.time()
        self.recording[10].update(tag2-tag1)
        logging.info(f"push step4 {self.recording[10].val}, avg {self.recording[10].avg}")
        # msg=pickle.loads(self.sock.recv(self.rank))
        # assert msg == "ok"

        logging.info(f"push transmission rate {len(transmitted)/self.rows_number}")
        logging.info(f"push complete")
        end = time.time()
        return end - transmission_start_time, transmission_time

    def terminate(self):
        logging.info('Worker terminated.')
        self.sock.send(pickle.dumps("terminate"))
        self._stop_event.set()
        self.sock.send_iscomplete(1)
        
    def push_and_pull(self):
        push_update_time,transmission_time = self.push_update()
        pull_model_time=self.pull_model(transmission_time)
        return push_update_time,pull_model_time

    def train(self, local_update):
        # bar = Bar('Processing', max=len(self.train_loader))
        self.model.train()
        self.model.apply(freeze_bn)
        
        itered = 0
        all_iters = len(self.train_loader)
        loader = iter(self.train_loader)
        finish = False
        self.batchsize = local_update * self.train_loader.batch_size
        for i in range(self.rows_number):
            self.remain_number[i]=0
        for p in self.model.parameters():
            if p.grad != None:
                p.grad=torch.zeros_like(p.grad)
        while itered < all_iters:
            start = time.time()
            self.lock.acquire()
            for _ in range(local_update):
                try:
                    images, targets = next(loader)
                except StopIteration:
                    finish = True
                    break
                itered += 1
                images = images.to(self.device)
                targets = targets.to(self.device)
                images, targets = self.mixup_fn(images, targets)
                output = self.model(images)
                loss = self.criterion(output, targets)
                loss.backward()
                self.losses.update(loss.item(), images.size(0))
            for i in range(self.rows_number):
                self.remain_number[i]+=1
            self.training_step += 1
            end = time.time()
            if finish:
                break
            self.t_computation.update(end - start)
            if self.sleep_time > 0.:
                time.sleep(self.sleep_time)
            self.gpu_running.update(time.time()-start)
            push_update_time, pull_model_time = self.push_and_pull()
            for p in self.model.parameters():
                p.grad.zero_()
            self.lr_scheduler.step()
            self.lock.release()
            self.t_communication.update(pull_model_time + push_update_time)
            logging.info(f"Iteration {self.training_step} worker{self.rank}") 
            logging.info(f"communication: {pull_model_time + push_update_time:.2f}; avg {self.t_communication.avg:.2f}; sum {self.t_communication.sum:.2f} compress: {compress_cost.avg:.2E} decompress: {decompress_cost.avg:.2E}")
            logging.info(f"computation: {self.t_computation.val:.2f}; avg {self.t_computation.avg:.2f}; sum {self.t_computation.sum:.2f}; sleep:{self.sleep_time:.2f} total compute+compress+sleep: {self.t_computation.val + compress_cost.val + decompress_cost.val + self.sleep_time:.2f}")
            logging.info(f"GPU running time: {self.gpu_running.val:.2f}; avg: {self.gpu_running.avg:.2f}; sum {self.gpu_running.sum:.2f}")
            logging.info(f"loss: {self.losses.val:.2f} loss avg: {self.losses.avg:.2f}\n")
        return self.losses.avg