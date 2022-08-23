from ConvMLP import ConvMLP
from timm.utils import *
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import time

cudnn.enabled = True
cudnn.benchmark = True

true_batch_size = 2
mini_batch_size = 2

convmlp = ConvMLP(batch_size=mini_batch_size, idx_per_worker_train=200, total_worker=1, idx_per_worker_test=50)
model, optimizer, lr_scheduler, loader_train, loader_eval, train_loss_fn, mixup_fn = convmlp.init()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
def eval_orig():
    top1_m = AverageMeter()
    speed = AverageMeter()
    start = time.time()
    model.eval()
    for input, target in tqdm(loader_eval, desc='test'):
        input, target = input.to(device), target.to(device)
        output = model(input)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1_m.update(acc1.item(), output.size(0))
        end = time.time()
        speed.update(end - start)
        start = end
        del input, target, output
    print(f'eval: {speed.avg} s per batch; noised avg: {top1_m.avg};')

def train():
    model.train()
    top1_m = AverageMeter()
    speed = AverageMeter()
    start = time.time()
    import math
    pbar = tqdm(total=math.ceil(len(loader_train) / true_batch_size * mini_batch_size), desc='train')
    data = iter(loader_train)
    local_update = int(true_batch_size / mini_batch_size)
    while True:
        finish = False
        processed = False
        optimizer.zero_grad()
        for _ in range(local_update):
            try:
                input, _target = next(data)
                processed = True
            except:
                finish = True
                break
            input, _target = input.to(device), _target.to(device)
            input, target = mixup_fn(input, _target)
            output = model(input)
            loss = train_loss_fn(output, target)
            loss.backward()
            acc1, acc5 = accuracy(output, _target, topk=(1, 5))
            top1_m.update(acc1.item(), output.size(0))
            # del input, target, _target, output
        if processed:
            optimizer.step()
            end = time.time()
            speed.update(end - start)
            start = end
            pbar.update(1)
            pbar.set_description_str(f'train acc1: {top1_m.avg:.2f}')
        if finish:
            break
    print(f'train: {speed.avg} s per batch; training avg: {top1_m.avg}')

def eval_adapted():
    top1_m = AverageMeter()
    speed = AverageMeter()
    start = time.time()
    model.eval()
    for input, target in tqdm(loader_eval, desc='test'):
        input, target = input.to(device), target.to(device)
        output = model(input)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1_m.update(acc1.item(), output.size(0))
        # del input, target, output
        end = time.time()
        speed.update(end - start)
        start = end
        del input, target, output
    print(f'eval: {speed.avg} s per batch; adapted avg: {top1_m.avg}')

if __name__ == "__main__":
    eval_orig()
    train()
    eval_adapted()