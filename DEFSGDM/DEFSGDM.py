'''
Implementation code (SSP compatible version) of Distributed Blockwise Momentum
SGD with Error-Feedback (dist-EF-blockSGDM) in NIPS 2019 paper:
'Communication-Efﬁcient Distributed Blockwise Momentum SGD with Error-Feedback'
'''
import torch
from torch.optim import Optimizer
import numpy as np
from timm.utils import AverageMeter
import time
from threading import Lock

numba_avail = False
try:
    from numba import njit, prange
    numba_avail = True
except:
    pass

cupy_avail = False
try:
    import cupy
    cupy_avail = True
except:
    pass

# delete
# cupy_avail = False
# numba_avail = False
print(f'numba_avail: {numba_avail}; cupy_avail: {cupy_avail}')

if not numba_avail and not cupy_avail:
    def packbits(param: torch.Tensor):
        # param = np.where(param >= 0, 1, 0)
        return np.packbits(param.cpu().numpy())

    def unpackbits(param: np.ndarray):
        param = np.unpackbits(param)
        param = np.where(param > 0, 1., -1.)
        return param

elif cupy_avail:
    # cupy is faster than numpy in unpacking, but slower in packing,
    # but I expect cupy is always faster when there isn't a powerful cpu
    def packbits(param: torch.Tensor):
        # param = torch.where(param >= 0, 1, 0)
        return cupy.asnumpy(cupy.packbits(cupy.asarray(param)))

    def unpackbits(param: np.ndarray):
        param = cupy.unpackbits(cupy.asarray(param))
        param = cupy.where(param > 0, 1., -1.)
        return param

elif numba_avail:
    # numpy seems to be faster than numba in packing bits, but slower in unpacking bits
    # but I expect numba is always faster when there isn't a powerful cpu
    # @njit('void(uint8[::1], uint8[::1], int_)', inline='never')
    # def _numba_pack_x64(arr, su, pos):
    #     for i in range(64):
    #         j = i * 8
    #         su[i] = (arr[j]<<7)|(arr[j+1]<<6)|(arr[j+2]<<5)|(arr[j+3]<<4)|(arr[j+4]<<3)|(arr[j+5]<<2)|(arr[j+6]<<1)|arr[j+7]

    # @njit('void(uint8[::1], int_, uint8[::1])', parallel=True)
    # def _packbits(arr, div, su):
    #     # arr = np.where(arr >= 0, True, False)
    #     for i in prange(div//64):
    #         _numba_pack_x64(arr[i*8:(i+64)*8], su[i:i+64], i)
    #     for i in range(div//64*64, div):
    #         j = i * 8
    #         su[i] = (arr[j]<<7)|(arr[j+1]<<6)|(arr[j+2]<<5)|(arr[j+3]<<4)|(arr[j+4]<<3)|(arr[j+5]<<2)|(arr[j+6]<<1)|arr[j+7]

    # def packbits(param: torch.Tensor):
    #     length = param.numel()
    #     param = param.cpu().numpy()
    #     div, mod = np.divmod(length, 8)
    #     su = np.zeros(div + (mod > 0), dtype=np.uint8)
    #     _packbits(param[:div*8], div, su)
    #     if mod > 0:
    #         su[-1] = sum(x*y for x, y in zip(param[div*8:], (128, 64, 32, 16, 8, 4, 2, 1)))
    #     return su

    def packbits(param: torch.Tensor):
        # param = np.where(param >= 0, 1, 0)
        return np.packbits(param.cpu().numpy())

    # numba version of unpackbits seems to be a bit faster than numpy.unpackbits
    mask = 2**np.arange(7, -1, -1, dtype=np.uint8)

    @njit(parallel=True)
    def unpackbits(x, Nbits=8):
        out_NbitAr = np.zeros(len(x) * Nbits)
        for idx, n in enumerate(x):
            for _idx, m in enumerate(mask):
                if m & n > 0:
                    out_NbitAr[idx*8 + _idx] = 1.
                else:
                    out_NbitAr[idx*8 + _idx] = -1.
        return out_NbitAr

else:
    raise RuntimeError

compress_cost = AverageMeter()
serialize_cost = AverageMeter()
decompress_cost = AverageMeter()
deserialize_cost = AverageMeter()


def serialize(param: torch.Tensor):
    '''
    compress 1d torch tensor float32/64 to packed one bit array (uint8)
    '''
    stime = time.time()
    assert isinstance(param, torch.Tensor)
    length = param.numel()
    # one bit filling uint8; zeros can be padded
    param = packbits(param)
    data = {
        'length': length,
        'param': param
    }
    serialize_cost.update(time.time() - stime)
    return data


def deserialize(data: np.ndarray, dtype=torch.float32,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    '''
    decompress packed one bit array to torch 1d tensor float32
    '''
    stime = time.time()
    length = data['length']
    param = data['param']
    param = unpackbits(param)
    # erase the padded 0s
    param = param[:length]
    param = torch.as_tensor(param, device=device,  dtype=dtype)
    deserialize_cost.update(time.time() - stime)
    return param


class DEFSGDM_worker(Optimizer):
    '''
    DEFSGDM compressor on the worker side.
    Can be applied to either a whole model or certain param group of the model.
    '''
    def __init__(self, params, lr=0.000001, momentum=0.9, dampening=0.0,
                 weight_decay=0.0005, nesterov=True, device=None):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                        nesterov=nesterov)
        super(DEFSGDM_worker, self).__init__(params, defaults)
        self.whole_model = None
        self.device = device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_state()
        self.send_scale_monitor = AverageMeter()
        self.recv_scale_monitor = AverageMeter()

    def init_state(self):
        self.whole_model = None
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # State initialization
                if self.whole_model is None:
                    state['pos'] = 0
                    self.whole_model = torch.zeros_like(p.data.view(-1), dtype=torch.uint8, device=self.device)
                else:
                    state['pos'] = self.whole_model.shape[0]
                    self.whole_model = torch.cat([self.whole_model,
                        torch.zeros_like(p.data.view(-1), dtype=torch.uint8, device=self.device)])
                p.error = torch.zeros_like(p.data, device=self.device)
                # p.temp_error = torch.zeros_like(p.data, device=self.device)
                p.momentum = torch.zeros_like(p.data, device=self.device)
                p.remain_d_p = torch.zeros_like(p.data, device=self.device)
                state['last_lr'] = group['lr']
                state['accumulated_grad'] = torch.zeros_like(p.data, device=self.device)
                state['has_grad'] = False

    @torch.no_grad()
    def update_error(self):
        # wholely update error for SSP and FLOWN
        # for ROG, you need to manually update error of those sent rows
        for group in self.param_groups:
            for p in group['params']:
                p.remain_d_p.zero_()


    @torch.no_grad()
    def compress(self, compress_here=True):
        stime = time.time()
        param_groups = []
        temp_scale_monitor = AverageMeter()
        for group in self.param_groups:
            temp_scale_lr = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p in group['params']:
                state = self.state[p]
                if p.grad is None:
                    temp_scale_lr.append([None, None, None])
                    continue
                grad = p.grad.data

                # SGD
                p.momentum.mul_(momentum).add_(grad, alpha=1 - dampening)
                if nesterov:
                    d_p = grad + momentum * p.momentum
                else:
                    d_p = p.momentum

                # Add d_p from previous iteration
                d_p.add_(p.remain_d_p, alpha=state['last_lr'] / group['lr'])
                # error feedback
                d_p.add_(p.error, alpha=state['last_lr'] / group['lr'])

                # Weight decay
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=group['weight_decay'] / (1-momentum))

                # compress gradient
                scale = d_p.abs().mean()

                temp_scale_monitor.update(scale, n=p.numel())

                sign = torch.sign(d_p)
                sign = torch.where(sign.view(-1) >= 0, 1, 0)
                self.whole_model[state['pos']: state['pos'] + p.data.numel()] = sign
                temp_scale_lr.append([scale.cpu(), state['last_lr'], group['lr']])
                sign = torch.where(sign > 0, 1., -1.).view(p.data.shape)
                compressed = scale * sign
                p.error.copy_(d_p - compressed)
                # Store the compressed model; error feedback in the following iterations will cancel the error
                p.remain_d_p.copy_(compressed)

            param_groups.append(temp_scale_lr)
        self.send_scale_monitor.update(temp_scale_monitor.avg)
        if compress_here:
            whole_model = serialize(self.whole_model)
        else:
            # If not compress here, you need to serialize self.whole_model outside
            whole_model = self.whole_model
        compress_cost.update(time.time() - stime)
        return whole_model, param_groups

    @torch.no_grad()
    def step(self):
        pass

    @torch.no_grad()
    def optimize_stored(self):
        # avoid warning
        self.step()
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if not state['has_grad']:
                    continue
                d_p = state['accumulated_grad']
                # Step: learning rate has been multiplied
                # in the scale at the server side
                p.data.add_(d_p, alpha=-1.)
                state['last_lr'] = group['lr']
                state['has_grad'] = False
                state['accumulated_grad'] = torch.zeros_like(d_p)

    @torch.no_grad()
    def decompress_store(self, sign, param_groups):
        stime = time.time()
        sign = deserialize(sign, device=self.device)
        temp_scale_monitor = AverageMeter()
        for group, scales in zip(self.param_groups, param_groups):
            for p, scale in zip(group['params'], scales):
                if isinstance(scale, list):
                    scale = scale[0]
                if scale is None or scale == 0.:
                    continue
                temp_scale_monitor.update(scale, p.data.numel())
                length = p.data.numel()
                state = self.state[p]
                state['accumulated_grad'].add_(
                    sign[state['pos']: state['pos'] + length].view(p.data.shape), alpha=scale)
                state['has_grad'] = True
        self.recv_scale_monitor.update(temp_scale_monitor.avg)
        decompress_cost.update(time.time() - stime)

    @torch.no_grad()
    def decompress_optimize(self, sign, param_groups, decompress_here=True):
        # avoid warning
        self.step()
        stime = time.time()
        if decompress_here:
            sign = deserialize(sign, device=self.device)
        temp_scale_monitor = AverageMeter()
        for group, scales in zip(self.param_groups, param_groups):
            for p, scale in zip(group['params'], scales):
                length = p.data.numel()
                if scale is None or scale == 0.:
                    continue
                state = self.state[p]
                temp_scale_monitor.update(scale, p.data.numel())
                # Step: learning rate has been multiplied in scale at the server side
                p.data.add_(sign[state['pos']: state['pos'] + length].view(p.data.shape), alpha=-scale)
                state['last_lr'] = group['lr']
        self.recv_scale_monitor.update(temp_scale_monitor.avg)
        decompress_cost.update(time.time() - stime)


class DEFSGDM_server(Optimizer):
    '''
    DEFSGDM compressor on the server side.
    Can be applied to either a whole model or certain param group of the model.
    '''
    def __init__(self, params, worker_num=4, local_copy=True, device=None):
        self.worker_num = worker_num
        self.local_copy = local_copy
        self.device = device
        self.whole_model_size = 0
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        defaults = dict(worker_num=worker_num)
        super(DEFSGDM_server, self).__init__(params, defaults)
        self.init_state()
        self.locks = [Lock() for _ in range(worker_num)]

    def init_state(self):
        self.whole_model_size = 0
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['pos'] = self.whole_model_size
                self.whole_model_size += p.data.numel()
                if self.local_copy:
                    state['accumulated_grad'] = torch.zeros_like(p.data, device=self.device)
                    state['has_grad'] = False
                p.grad_per_worker = [torch.zeros_like(p.data.view(-1), device=self.device) for _ in range(self.worker_num)]
                p.temp_grad_per_worker = [torch.zeros_like(p.data.view(-1), device=self.device) for _ in range(self.worker_num)]
                p.error_per_worker = [torch.zeros_like(p.data.view(-1), device=self.device) for _ in range(self.worker_num)]
                p.temp_error_per_worker = [torch.zeros_like(p.data.view(-1), device=self.device) for _ in range(self.worker_num)]
                # state['error_per_worker'] = [torch.zeros_like(p.data.view(-1), device=self.device) for _ in range(self.worker_num)]
                # state['temp_error_per_worker'] = [torch.zeros_like(p.data.view(-1), device=self.device) for _ in range(self.worker_num)]
                state['lr_per_worker'] = [[0, 0] for _ in range(self.worker_num)]

    @torch.no_grad()
    def recv(self, sign, param_groups, worker_id, decompress_here=True):
        stime = time.time()
        if decompress_here:
            sign = deserialize(sign, device=self.device)
        [lock.acquire() for lock in self.locks]
        for group, Param in zip(self.param_groups, param_groups):
            for p, param in zip(group['params'], Param):
                state = self.state[p]
                scale, lr_last, lr_now = param
                length = p.data.numel()
                end = state['pos'] + length

                if scale is None or scale == 0.:
                    continue
                else:
                    sign[state['pos']: end].mul_(
                        scale * lr_now / self.worker_num)
                for i in range(self.worker_num):
                    p.grad_per_worker[i].add_(sign[state['pos']: end])
                    state["lr_per_worker"][i][0] = lr_last
                    state["lr_per_worker"][i][1] = lr_now
                    
                # if self.local_copy:
                #     state['accumulated_grad'].add_(
                #         sign[state['pos']: end].view(p.data.shape))
                #     state['has_grad'] = True
        [lock.release() for lock in self.locks]
        decompress_cost.update(time.time() - stime)

    @torch.no_grad()
    def update_error(self, worker_id):
        self.locks[worker_id].acquire()
        for group in self.param_groups:
            for p in group['params']:
                p.error_per_worker[worker_id].copy_(
                    p.temp_error_per_worker[worker_id])
                p.grad_per_worker[worker_id].add_(p.temp_grad_per_worker[worker_id], alpha=-1.)
        self.locks[worker_id].release()

    @torch.no_grad()
    def send(self, worker_id, compress_here=True):
        stime = time.time()
        param_groups = []
        whole_model = torch.zeros(self.whole_model_size, device=self.device, dtype=torch.uint8)
        self.locks[worker_id].acquire()
        for group in self.param_groups:
            temp_scale = []
            for p in group['params']:
                state = self.state[p]
                grad = p.grad_per_worker[worker_id].data
                p.temp_grad_per_worker[worker_id].copy_(p.grad_per_worker[worker_id])
                error = p.error_per_worker[worker_id]
                lr_last, lr_now = state['lr_per_worker'][worker_id]
                # error feedback
                p_t = grad + lr_last / lr_now * error
                # p_t.add_(error, alpha=lr_last/lr_now)
                # p_t = grad + lr_last / lr_now * error
                # compress scale & sign
                scale = p_t.abs().mean()
                sign = torch.sign(p_t)
                sign = torch.where(sign.view(-1) >= 0., 1, 0)
                whole_model[state['pos']: state['pos'] + p.data.numel()] = sign
                # update error
                sign = torch.where(sign > 0, 1., -1.)
                p.temp_error_per_worker[worker_id].copy_(p_t - scale * sign)
                temp_scale.append(scale)
                state['accumulated_grad'].add_(sign.view(p.data.shape), alpha = scale/self.worker_num)
                state['has_grad'] = True
            param_groups.append(temp_scale)
        self.locks[worker_id].release()
        if compress_here:
            whole_model = serialize(whole_model)
        compress_cost.update(time.time() - stime)
        # If not compress here, you need to serialize whole_model outside
        return whole_model, param_groups

    @torch.no_grad()
    def step(self):
        assert self.local_copy
        [l.acquire() for l in self.locks]
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if not state['has_grad']:
                    continue
                d_p = state['accumulated_grad']
                # Step: learning rate has been multiplied
                # in the scale at the server side
                p.data.add_(d_p, alpha=-1.)
                state['has_grad'] = False
                state['accumulated_grad'] = torch.zeros_like(d_p)
        [l.release() for l in self.locks]
