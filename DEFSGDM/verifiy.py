import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DEFSGDM.DEFSGDM import DEFSGDM_worker, DEFSGDM_server, compress_cost, decompress_cost, serialize_cost, deserialize_cost
from ConvMLP import ConvMLP
from timm.utils import *
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, MultiStepLR
import time
import random
import logging
from datetime import datetime


total_statistical_efficiency = AverageMeter()
last_accuracy = 65.56
last_step = 0
cudnn.enabled = True
cudnn.benchmark = True

num_workers = 4
threshold = 3
if len(sys.argv) >  1:
    threshold = int(sys.argv[1])

lr = 1e-6
if len(sys.argv) >  2:
    lr = float(sys.argv[2])
print(f'Threshold {threshold} lr {lr}')

logger = logging.getLogger(f'threshold_{threshold}_lr_{lr:.2E}')
logger.setLevel(logging.INFO)

fh = logging.FileHandler(f'{datetime.now().strftime("%m-%d_%H:%M:%S")}_lr_{lr:.2E}_thres_{threshold}.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

batch_size = [2,24,24,8]


def freeze_bn(m):
    if isinstance(m, torch.nn.LayerNorm) or isinstance(m, torch.nn.BatchNorm2d):
        m.training = False
        m.track_running_stats = True

def init_workers():
    '''
    init
    '''
    workers = []
    for i in range(num_workers):
        device = torch.device(f"cuda:{i}" if torch.cuda.is_available() else "cpu")
        idx_num = 125
        convmlp = ConvMLP(batch_size=batch_size[i], num_shards=4, worker_idx=i, device=device)
        model, _, _, loader_train, loader_eval, train_loss_fn, mixup_fn = convmlp.init()
        model.apply(freeze_bn)
        optimizer = DEFSGDM_worker(model.parameters(), lr=lr, device=device, momentum=0.9, dampening=0.)
        # lr_scheduler = StepLR(optimizer, 500, gamma=0.1)
        lr_scheduler = MultiStepLR(optimizer, [200, 400], gamma=0.5)
        optimizer.zero_grad()
        temp = {
            'model': model,
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'train_dl': loader_train,
            'train_iter': iter(loader_train),
            'test_dl': loader_eval,
            'criterion': train_loss_fn,
            'mixup_fn': mixup_fn,
            'device': device,
            'versions': [0] * num_workers
        }
        workers.append(temp)
        workers[i]['model'].train()
        workers[i]['model'].to(workers[i]['device'])
    return workers


def init_server():
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    convmlp = ConvMLP(
        batch_size=128,num_shards=1, num_client=1, worker_idx=0, device=device)
    model, _, _, loader_train, loader_eval, train_loss_fn, mixup_fn = convmlp.init()
    model.to(device)
    optimizer = DEFSGDM_server(model.parameters(), worker_num=num_workers, device=device, local_copy=True)
    server = {
        'model': model,
        'optimizer': optimizer,
        'device': device,
        'test_dl': loader_eval,
        'versions': [0] * num_workers
    }
    return server

def check_divergence(workers, prefix='worker'):
    max_std = None
    for groups in zip(*[worker['optimizer'].param_groups for worker in workers]):
        for ps in zip(*[group['params'] for group in groups]):
            p = torch.cat([p.data.cpu().unsqueeze(0) for p in ps])
            _max_std = p.std(dim=0).max()
            if max_std is None:
                max_std = _max_std
            elif _max_std > max_std:
                max_std = _max_std
    print(f'Current max std among models on {prefix} is {max_std}')


logger.info('step,versions,accuracy,statistical_efficiency,average_statistical_efficiency')
def eval_acc(workers, prefix='worker', supplement_loader=None):
    for i, worker in enumerate(workers):
        top1_m = AverageMeter()
        speed = AverageMeter()
        start = time.time()
        model = worker['model']
        loader_eval = worker['test_dl']
        if supplement_loader is not None:
            loader_eval = supplement_loader
        device = worker['device']
        model.eval()
        with torch.no_grad():
            for input, target in tqdm(loader_eval, desc='test', leave=False):
                input, target = input.to(device), target.to(device)
                output = model(input)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1_m.update(acc1.item(), output.size(0))
                end = time.time()
                speed.update(end - start)
                start = end
        acc = top1_m.avg
        step = max(server['versions'])
        global last_step
        step_diff = step - last_step
        last_step = step
        statistical_efficiency = 0
        global last_accuracy
        if last_accuracy is not None and step_diff != 0:
            statistical_efficiency = (acc - last_accuracy)/step_diff
            total_statistical_efficiency.update(statistical_efficiency, n=step_diff)
        last_accuracy = acc
        
        logger.info(f"{step},'{'-'.join([str(v) for v in server['versions']])}',{acc},{statistical_efficiency:.2E},{total_statistical_efficiency.avg:.2E}")

server = None
def main():
    workers = init_workers()
    global server
    training = 0
    push = 1
    pull = 2
    states = [training] * 4
    random_stop = [-1] * 4
    straggler = [0] * 4
    server = init_server()
    supplement_loader = server['test_dl']
    timestep = 0
    eval_step = 10
    last_step = -1
    # eval_acc([server], 'server')
    print(workers[0]['optimizer'].defaults)
    pbar = tqdm(range(eval_step), leave=False)
    while True:
        if timestep > 6000:
            break
        timestep += 1
        report = f'{timestep}: '
        if max(straggler) <= 0:
            num_straggler = len(workers) // 2
            for i in range(num_straggler):
                idx = random.randint(0, len(workers) - 1)
                while straggler[idx] > 0:
                    idx = random.randint(0, len(workers) - 1)
                straggler[idx] = random.randint(int(threshold/2), threshold * 2)
        for i, worker in enumerate(workers):
            state = states[i]
            report += f'{i} '
            # if now_move == 0:
            #     report += f'{i}-sleep; '
            #     continue
            if state == training:
                worker['model'].train()
                worker['model'].apply(freeze_bn)
                if worker['versions'][i] - min(worker['versions']) < threshold:
                    try:
                        images, target = next(worker['train_iter'])
                    except StopIteration:
                        worker['train_iter'] = iter(worker['train_dl'])
                        images, target = next(worker['train_iter'])
                    images, target = images.to(worker['device']), target.to(worker['device'])
                    images, target = worker['mixup_fn'](images, target)
                    output = worker['model'](images)
                    acc = accuracy(output, target.max(dim=-1).indices)[0]
                    loss = worker['criterion'](output, target)
                    loss.backward()
                    worker['versions'][i] += 1
                    report += f'compute loss {loss:.2f} acc {acc:.2f}'
                else:
                    report += f'threshold '
                states[i] = push
            elif state == push:
                if straggler[i] > 0 and random_stop[i] < 0:
                    random_stop[i] = random.randint(2, threshold)
                if 0 >= random_stop[i] or max(server['versions']) - worker['versions'][i] >= threshold:
                    sign_model, param_group = worker['optimizer'].compress()
                    worker['optimizer'].zero_grad()
                    server['optimizer'].recv(sign_model, param_group, i)
                    worker['optimizer'].update_error()
                    if worker['versions'][i] > max(server['versions']):
                        pbar.update(worker['versions'][i] - max(server['versions']))
                    server['versions'][i] = worker['versions'][i]
                    report += 'send '
                    states[i] = pull
                    random_stop[i] = -1
                random_stop[i] -= 1
            elif state == pull:
                if server['versions'][i] - min(server['versions']) >= threshold:
                    report += 'wait '
                else:
                    if straggler[i] > 0 and random_stop[i] < 0:
                        random_stop[i] = random.randint(int(threshold/2), threshold*2)
                    if 0 >= random_stop[i] or max(server['versions']) - worker['versions'][i] >= threshold:
                        sign_model, param_group = server['optimizer'].send(i)
                        server['optimizer'].update_error(i)
                        # worker['optimizer'].decompress_store(sign_model, param_group)
                        worker['optimizer'].decompress_optimize(sign_model, param_group)
                        worker['optimizer'].step()
                        for j in range(num_workers):
                            worker['versions'][j] = server['versions'][j]
                        worker['lr_scheduler'].step()
                        report += 'recv-optim lr-sched'
                        states[i] = training
                        random_stop[i] = -1
                        straggler[i] -= 1
                    random_stop[i] -= 1
            report += '; '
        # report += f'compress {compress_cost.val:.2E}; serialize {serialize_cost.val:.2E}; decompress {decompress_cost.val:.2E}; deserialize {deserialize_cost.val:.2E}'
        # print(report)
        if max(server['versions']) % eval_step == 0 and max(server['versions']) != last_step:
            server['optimizer'].step()
            pbar.close()
            # check_divergence(workers, 'workers')
            # logger.info(f'training steps {server["versions"]}')
            # check_divergence(workers + [server], 'workers and server')
            # eval_acc(workers, prefix=f'worker {server["versions"]}', supplement_loader=supplement_loader)
            eval_acc([server], f'server {server["versions"]}', supplement_loader)

            pbar = tqdm(range(eval_step), leave=False)
            last_step = max(server['versions'])

if __name__ == '__main__':
    main()
