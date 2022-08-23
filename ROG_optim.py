import torch
from ROG_utils import ROG_Local_Worker, ROG_Parameter_Server, layer_unit
from DEFSGDM.DEFSGDM import DEFSGDM_server, DEFSGDM_worker
import logging
server = 0
worker = 1

class ROG_Optimizer(torch.optim.Optimizer):
    def __init__(self, model,args,communication_library,local_copy=False):
        if args.rank == 0:
            self.role = server
        else:
            self.role = worker
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.layer_info=[]
        start_idx=0
        start_pos=0
        self.temp_optimizer = None
        for p in model.parameters():
            each_layer_rows,start_idx,start_pos = layer_unit(p, start_idx, start_pos)
            self.layer_info.append(each_layer_rows)

        if self.role == server:
            self.optimizer = DEFSGDM_server(params=model.parameters(), worker_num=args.world_size-1, device=self.device, local_copy=local_copy)
            self.server = ROG_Parameter_Server(args, model=model, layer_info=self.layer_info, communication_library = communication_library, device=self.device,optimizer=self.optimizer)

        else:
            self.optimizer = DEFSGDM_worker(params=model.parameters(), device=self.device,lr =1e-6)
            self.worker = ROG_Local_Worker(args, model=model, layer_info=self.layer_info, communication_library = communication_library,device=self.device, optimizer=self.optimizer )
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