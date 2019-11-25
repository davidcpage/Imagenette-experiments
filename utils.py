import math
from itertools import chain
import torch
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator

class _Pipe(Pipeline):
    def __init__(self, graph, batch_size, num_threads, device_id, seed):
        super().__init__(batch_size, num_threads, device_id, seed=seed)
        self.define_graph = graph
        self.build()

class DALIDataLoader():
    def __init__(self, graph, batch_size, drop_last=False, num_threads=4, device=None, seed=-1):
        self.device = device
        self.pipe = _Pipe(graph, batch_size, num_threads=num_threads, device_id=self.device.index, seed=seed)
        n = self.pipe.epoch_size('Reader')
        if drop_last:
            self.length = n // batch_size
            n = self.length * batch_size 
        else:
            self.length = math.ceil(n/batch_size)
        self.dali_iter = DALIGenericIterator([self.pipe], ['data', 'label'], n, auto_reset=True, fill_last_batch=False)

    def __iter__(self): return ((batch[0]['data'], 
                                 batch[0]['label'].squeeze().to(dtype=torch.int64, device=self.device)
                                ) for batch in self.dali_iter)
    
    def __len__(self): return self.length

class Chain():
    def __init__(self, *dls):
        self.dls = dls
        self.device = self.dls[0].device
    def __iter__(self): return chain(*self.dls)
    def __len__(self): return sum(len(dl) for dl in self.dls)

class MockV1DataBunch(): #adaptor to fastai v1 databunch api
    def __init__(self, train_dl, valid_dl, path='dummy', empty_val=False):
        self.train_dl = train_dl
        if not hasattr(train_dl, 'dataset'): train_dl.dataset = 'dummy'
        self.valid_dl = valid_dl
        if not hasattr(valid_dl, 'dataset'): valid_dl.dataset = 'dummy'
        self.path = path
        self.device = train_dl.device
        self.empty_val = empty_val
    def add_tfm(self, tfm): pass
    def remove_tfm(self, tfm): pass