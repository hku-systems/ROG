import torch
from SSP_utils import Local_Worker, Parameter_Server
from DEFSGDM.DEFSGDM import DEFSGDM_server, DEFSGDM_worker
import logging
server = 0
worker = 1

class SSP_Optimizer(torch.optim.Optimizer):
    def __init__(self, model,args,communication_library,local_copy=False):
        if args.rank == 0:
            self.role = server
        else:
            self.role = worker
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.role == server:
            self.optimizer = DEFSGDM_server(params=model.parameters(), worker_num=args.world_size-1, device=self.device, local_copy=local_copy)
            self.server = Parameter_Server(args, model=model, communication_library = communication_library, device=self.device,optimizer=self.optimizer)

        else:
            self.optimizer = DEFSGDM_worker(params=model.parameters(), device=self.device,lr =1e-6)
            self.worker = Local_Worker(args, model=model, communication_library = communication_library,device=self.device, optimizer=self.optimizer )
            logging.info("worker start\n")

    def push_and_pull(self):
        assert self.role == worker
        self.worker.push_and_pull()
        
        
    def zero_grad(self):
        assert self.role == worker
        self.optimizer.zero_grad()

    def set_worker_adapt_noise(self,train_dl,criterion, mixup_fn):
        assert self.role == worker
        self.worker.set_adapt_noise(train_dl,criterion, mixup_fn)
    
        
    def train(self,local_update):
        assert self.role == worker
        self.worker.train(local_update)
        
    def terminate(self):
        assert self.role == worker
        self.worker.terminate()