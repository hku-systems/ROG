from ConvMLP import ConvMLP
from timm.utils import AverageMeter, accuracy
from torch.utils.tensorboard import SummaryWriter
import os
import torch
from tqdm import tqdm
import sys
import torch.backends.cudnn as cudnn
import csv

cudnn.enabled = True
cudnn.benchmark = True

dir_path = 'result/01-27-17-40'
if len(sys.argv) > 1:
    dir_path = sys.argv[1]
if os.path.exists(os.path.join(dir_path, 'chkpt')):
    dir_path = [dir_path]
else:
    dir_path = [os.path.join(dir_path, p)  for p in os.listdir(dir_path)]
for path in dir_path:
    print(f"Examining on {path}.")
    writer = SummaryWriter(path, filename_suffix='_eval_chkpt')
    convmlp = ConvMLP(worker_idx=-1, idx_per_worker_test=100, batch_size=128 * 5)
    model, _, _, _, test_dl, _, _ = convmlp.init()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_chkpt = os.listdir(os.path.join(path, 'chkpt'))
    temp = []
    for chkpt in all_chkpt:
        model_time, step = os.path.splitext(chkpt)[0].split('-')
        model_time = float(model_time)
        step = int(step)
        temp.append([model_time, step, chkpt])
    temp.sort(key=lambda x: x[1])

    f = open(os.path.join(path, 'accuracy.csv'), 'w')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['time', 'step', f'accuracy/{path}'])
    with torch.no_grad():
        for model_time, step, chkpt in temp:
            chkpt_path = os.path.join(path, 'chkpt', chkpt)
            model.load_state_dict(torch.load(chkpt_path))
            model.to(device)
            top1_m = AverageMeter()
            model.eval()
            for image, target in tqdm(test_dl, desc='test', leave=False):
                image, target = image.to(device), target.to(device)
                output = model(image)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1_m.update(acc1.item(), output.size(0))
            csv_writer.writerow([model_time, step, top1_m.avg])
            print(f"validate: step-{step} top1-{top1_m.avg}")
            writer.add_scalar('global_model/top1', top1_m.avg, step, float(model_time))
    f.close()