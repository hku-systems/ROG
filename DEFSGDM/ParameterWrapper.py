import torch


class Grad:
    def __init__(self, param, position: tuple):
        self.p = param
        self.position = position

    def __getattr__(self, name):
        if name == 'data':
            return self.p.grad.data[self.position]
        else:
            return getattr(self.p.grad, name)

a = torch.Tensor([1,2,3])
class Parameter(torch.Tensor):
    '''
    Wrapper that enables retriving part (determined by position) of the data and gradient of a parameter,
        while other attributes of the parameter remains.
    Parameter.data => Parameter.data[position]
    Parameter.grad.data => Parameter.grad.data[position]
    '''
    def __init__(self, param: torch.Tensor, position: tuple):
        self._p = param
        self._data = param.data
        self.position = position
        self.grad = Grad(param, position)

    @property
    def __class__(self):
        return torch.Tensor

    def __getattr__(self, name):
        if name == 'data':
            return self._p.data[self.position]
        elif name == 'grad':
            if self._p.grad is None:
                return None
            else:
                return self.grad
        elif hasattr(self._p, name):
            return getattr(self._p, name)
        else:
            raise AttributeError