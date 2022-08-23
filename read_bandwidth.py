# class cbq 1:1 parent 1: rate 102400Kbit (bounded,isolated) prio 5
import sys
import subprocess
import time

def get_bandwidth(wnic_name, default=100.0):
    """Returns current limit of bandwidth. Returns default if no limit found."""
    try:
        outputs = subprocess.run(f'tc qdisc list dev {wnic_name}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
        loi = None
        for line in outputs.decode().split('\n'):
            if 'qdisc netem' in line:
                loi = line
                break
        time.sleep(0.01)
        rate = float(loi.split()[9].replace('Kbit', ''))/1024
        return rate
    except Exception as e:
        print(e, loi)
        return default

if __name__ == "__main__":
    while True:
        print(get_bandwidth('wls1'))
        time.sleep(0.5)

