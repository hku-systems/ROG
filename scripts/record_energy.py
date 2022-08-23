import sys
import time
from collections import defaultdict

import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

monitor_interval = float(sys.argv[1])

try:
    import jtop.jtop
    jetson = jtop.jtop()
    jetson.start()
    while not jetson.ok():
        pass
    def read_power_watts():
        ret = {}
        # total
        ret['TOTAL'] = jetson.power[0]['cur']*1e-3
        # detail
        ret.update((k.replace(' ', '_'), v['cur']*1e-3) for k, v in jetson.power[1].items())
        return ret
    read_power_watts()
    logging.info('power api initialized')
except Exception as e:
    logging.error(f'failed to initialize power api: {e}')
    exit()

time.sleep(2)
logging.info('started')

prev_watts = None
energy = defaultdict(lambda: 0)
while True:
    curr_watts = read_power_watts()
    curr_ts = time.monotonic()
    if prev_watts is not None:
        for k in curr_watts.keys():
            energy[k] += (curr_watts[k]+prev_watts[k])/2*(curr_ts - prev_ts)
        msg = ''
        for k in energy:
            msg += f'{k} {energy[k]:.3f} '
        logging.info(f'energy (J): {msg}')
    prev_watts = curr_watts
    prev_ts = curr_ts
    time.sleep(monitor_interval)

