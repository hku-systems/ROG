from asyncio.selector_events import BaseSelectorEventLoop
import subprocess
import sys
import time
import datetime
import atexit
import os
import argparse

parser = argparse.ArgumentParser(description='ROG experiments')
parser.add_argument('-l', '--library', metavar='SSP', type=str, default='SSP',
                    choices=['SSP', 'FLOWN', 'ROG', 'BSP'], help='Type of experiment.')
parser.add_argument('-t', '--threshold', metavar='4', type=int, default=2,
                    help='Type of experiment.')
parser.add_argument('-c', '--control', metavar='indoors', default='',
                    choices=['indoors', 'outdoors', '','complex'], help='Add Tc control.')
parser.add_argument('-b', '--batchsize',default=1, type=int)
parser.add_argument('--no-compression', default=False,
                    action='store_true', help='Whether to use DEFSGDM compression.')
parser.add_argument('-e', '--epoch', default=4, type=int)
parser.add_argument('-n', '--note', default="", type=str, help="Note for exp")
args = parser.parse_args()

hosts = [
    '10.42.0.1',
    '10.42.0.2',
    '10.42.0.7',
    '10.42.0.8',
    '10.42.0.3',
]
leader_ip = '10.42.0.1'
wnic_names = ['wls1', 'wlx0013ef6f0c49', 'wlan0', 'wlan0', 'wlx0013ef5f09a3']
hosts = [f'user@{host}' for host in hosts]
hosts_set = list(sorted(set(hosts), key=hosts.index))  # remove duplicate but keep the order
passward = "useruser"
container_name = "rog"
rerun = True
subps = []  # all subprocesses started by me

E = [25, 1, 1, 1, 1]            # local updates for each worker
E= [i * args.batchsize for i in E]
batch_size = [4, 2, 24, 24, 8]
idx_num = [500, 4, 4, 4, 4]    # dataset size for each worker
library = args.library
threshold = args.threshold
tc_control = args.control
epoch = args.epoch
compression_arg = '--compression-enable'
if args.no_compression:
    compression_arg = ''

if library == "BSP":
    library_arg = '--' + 'SSP' + '-enable'
    assert threshold == 0
else:
    library_arg = '--' + library + '-enable'

def cleanup():
    print('cleaning')
    for host in hosts_set:
        exec_ssh(host, f'docker stop -t 1 {container_name}', block=True)
    for p in subps:
        tmp = ' '
        print(f'killing "{tmp.join(p.args)}"')
        p.kill()


atexit.register(cleanup)


def Popen(*args, **kwargs):
    p = subprocess.Popen(*args, **kwargs)
    subps.append(p)
    return p


def exec_local(command, block=False):
    args = ['bash', '-c', command]
    print(f'executing on local: {command}')
    p = Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if block:
        ret = p.wait()
        assert ret == 0, 'something wrong during excution: \n{p.stderr.readlines()}'
    return p


def exec_ssh(host, command, block=False):
    args = [
        'ssh', '-o', 'ServerAliveInterval 60', '-o', 'ServerAliveCountMax 120',
        host, command,
    ]
    print(f"executing on {host}: {command}")
    p = Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ret = 0
    if block:
        ret = p.wait()
        if ret != 0: 
            print(f'something wrong during execution: \n{p.stderr.readlines()}')
    return ret


def exec_docker(host, command, block=False):
    command = f'docker exec -t {container_name} bash -c "{command}"'
    return exec_ssh(host, command, block=block)

while rerun:
    rerun = False
    print('restarting all containers')
    for host in hosts_set:
        ret = exec_ssh(host, f'docker restart -t 1 {container_name}', block=True)
        if ret != 0:
            print("restart device")
            exec_ssh(host,f"echo \"{passward}\" | sudo -S reboot",block=True)
            time.sleep(120)
            ret = exec_ssh(host,f"echo \"{passward}\" | sudo -S mount -a",block=True)
            assert ret ==0, "restart device failed"
            


    time_mark = datetime.datetime.now().strftime('%m-%d-%H-%M')
    result_dir = f'result/{time_mark}-{library}-{threshold}-{tc_control}'
    log_dir = f'{result_dir}/log'
    chkpt_dir = f'{result_dir}/chkpt'
    chkpt_worker_idx = 1
    os.makedirs(log_dir)
    os.makedirs(chkpt_dir)
    with open(f'{result_dir}/config', 'w') as f:
        config_string = f'{library}:\nworld_size: {len(hosts)}\n' + \
            f'threshold: {threshold}\nbw control: {tc_control}\nepoch: {epoch}\nE: {E}\n' + \
            f'library_arg: {library_arg}\nnote: {args.note}'
        f.write(config_string)

    server_log = None
    for idx, host in enumerate(hosts):
        exec_docker(host, f'tc qdisc del dev {wnic_names[idx]} root')
        exec_docker(host, f'cd /home/work/MICRO22-AE && \
            export GLOO_SOCKET_IFNAME={wnic_names[idx]} && \
            python3 -u adapt_noise_ssp.py --noise-type mixed --epochs {epoch} -b {batch_size[idx]} -E {E[idx]} --idx-start {sum(idx_num[1:idx])} --idx-num {idx_num[idx]} --chkpt-dir {chkpt_dir} --chkpt-rank {chkpt_worker_idx} --world-size {len(hosts)} --wnic-name {wnic_names[idx]} --rank {idx} --dist-url tcp://{leader_ip}:46666 --threshold={threshold} {library_arg} {compression_arg}>> {log_dir}/worker_{idx}.log 2>&1')
        if idx == 0:
            server_log = f'{log_dir}/worker_{idx}.log'
        time.sleep(0.5)

    if len(tc_control) > 0:
        bw_record = f'bw_records/{tc_control}.txt'
        print('starting bandwidth replay')
        for idx, host in enumerate(hosts):
            if idx == 0:
                exec_docker(hosts[idx], f'cd /home/work/MICRO22-AE/scripts && bash ./limit_bandwidth.sh {wnic_names[idx]}')
                continue
            mode = 'leader' if idx == 0 else 'worker'
            exec_docker(hosts[idx], f'cd /home/work/MICRO22-AE/scripts && python3 -u replay_bandwidth.py {bw_record} {mode} {idx+2} > ../{log_dir}/bw_replay-{idx}.txt 2>&1')
    else:
        for idx, host in enumerate(hosts):
            exec_docker(hosts[idx], f'cd /home/work/MICRO22-AE/scripts && bash ./limit_bandwidth.sh {wnic_names[idx]}')

    for idx, host in enumerate(hosts):
        exec_docker(host, f'cd /home/work/MICRO22-AE/scripts && python3 -u record_energy.py 0.5 > ../{log_dir}/energy-{idx}.txt 2>&1')

    print('all started')
    print('when enough, press ctrl+c ONCE, then wait for me to kill all started processes')
    print('but youd better double check then')
    print(args.note)
    last_line=None
    while True:
        with open(server_log, "rb") as file:
            try:
                file.seek(-2, os.SEEK_END)
                while file.read(1) != b'\n':
                    file.seek(-2, os.SEEK_CUR)
            except OSError:
                file.seek(0)
            line = file.readline().decode()
        if "Whole thread terminated" in line or 'BrokenPipeError' in line or "terminated" in line:
            print('Server down.')
            cleanup()
            break
        else:
            time.sleep(1800)
            if line == last_line:
                rerun=True
                print("may restart this case")
                cleanup()
                # exec_local(f"rm -rf {result_dir}",block=True)
                break
            last_line = line
# for p in subps:
#     p.wait()
