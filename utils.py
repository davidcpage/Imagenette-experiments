##########################
## DALI DataLoaders
##########################
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

class Map():
    def __init__(self, fn, dl):
        self.fn, self.dl, self.device = fn, dl, dl.device
    def __iter__(self): return map(self.fn, self.dl)
    def __len__(self): return len(self.dl)

class MockV1DataBunch(): 
    #adaptor to fastai v1 databunch api
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

##########################
## DALI Imagenet Pipeline
##########################
import nvidia.dali.ops as ops
import nvidia.dali.types as types

imagenet_stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

def imagenet_train_graph(data_dir, size, random_aspect_ratio, random_area, 
                interp_type=types.INTERP_TRIANGULAR,
                stats=imagenet_stats):
    inputs = ops.FileReader(file_root=data_dir, random_shuffle=True)
    decode = ops.ImageDecoderRandomCrop(device='mixed',
            random_aspect_ratio=random_aspect_ratio, random_area=random_area)
    resize = ops.Resize(device='gpu', resize_x=size, resize_y=size, 
                        interp_type=interp_type)
    mean, std = [[x*255 for x in stat] for stat in stats]
    crop_mirror_norm = ops.CropMirrorNormalize(
                        device='gpu', output_dtype=types.FLOAT16, 
                        crop=(size, size), mean=mean, std=std)
    coin = ops.CoinFlip(probability=0.5)

    def define_graph():    
        jpegs, labels = inputs(name='Reader')
        output = crop_mirror_norm(resize(decode(jpegs)), mirror=coin())
        return [output, labels]
    return define_graph

def imagenet_valid_graph(data_dir, size, val_xtra_size, mirror=0,
                interp_type=types.INTERP_TRIANGULAR, 
                stats=imagenet_stats):
    inputs = ops.FileReader(file_root=data_dir, random_shuffle=False)
    decode = ops.ImageDecoder(device='mixed', output_type=types.RGB)
    resize = ops.Resize(device='gpu', resize_shorter=size+val_xtra_size, 
                        interp_type=interp_type)
    mean, std = [[x*255 for x in stat] for stat in stats]
    crop_mirror_norm = ops.CropMirrorNormalize(
                        device='gpu', output_dtype=types.FLOAT16,
                        crop=(size, size), mean=mean, std=std, mirror=mirror)
    
    def define_graph():
        jpegs, labels = inputs(name='Reader')
        output = crop_mirror_norm(resize(decode(jpegs)))
        return [output, labels]
    return define_graph

##########################
## Models
##########################
import fastai2.vision.models
import fastai2.layers
import torch.nn as nn
from functools import partial

class XResNet(nn.Sequential):
    def __init__(self, expansion, layers, c_in=3, c_out=1000, 
                 sa=False, sym=False, act_cls=fastai2.basics.defaults.activation,
                 ):
        stem = []
        sizes = [c_in, 16,32,64] if c_in < 3 else [c_in, 32, 64, 64] 
        for i in range(3):
            stem.append(fastai2.layers.ConvLayer(sizes[i], sizes[i+1], stride=2 if i==0 else 1, act_cls=act_cls))

        block_szs = [64//expansion,64,128,256,512] +[256]*(len(layers)-4)
        blocks = [self._make_layer(expansion, ni=block_szs[i], nf=block_szs[i+1], blocks=l, stride=1 if i==0 else 2,
                                  sa=sa if i==len(layers)-4 else False, sym=sym, act_cls=act_cls)
                  for i,l in enumerate(layers)]
        super().__init__(
            *stem,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *blocks,
            nn.AdaptiveAvgPool2d(1), fastai2.layers.Flatten(),
            nn.Linear(block_szs[-1]*expansion, c_out),
        )
        fastai2.vision.models.xresnet.init_cnn(self)

    def _make_layer(self, expansion, ni, nf, blocks, stride, sa, sym, act_cls):
        return nn.Sequential(
            *[fastai2.layers.ResBlock(expansion, ni if i==0 else nf, nf, stride if i==0 else 1,
                      sa if i==(blocks-1) else False, sym=sym, act_cls=act_cls)
              for i in range(blocks)])
        
xresnet18 = partial(XResNet, expansion=1, layers=[2,2,2,2])
xresnet50 = partial(XResNet, expansion=4, layers=[3,4,6,3])

#faster Mish activation
@torch.jit.script
def mish_fwd(x):
    a = torch.exp(x)
    return x*(1. - 2./(2. + 2*a + a*a))

@torch.jit.script
def mish_bwd(x):
    a = torch.exp(x)
    t = (1. - 2./(2. + 2*a + a*a))
    return (t + x*(1.-t*t)*(a/(1.+a)))

class MishJitFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return mish_fwd(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return mish_bwd(x)*grad_output

class MishJit(nn.Module):
    def forward(self, x):
        return MishJitFunc.apply(x)

##########################################
## Flat_then_cos schedule for fastai v1
##########################################
import fastai.callback, fastai.callbacks

def flat_then_cosine_sched(learn, n_batch, lr, pct_start):
    return fastai.callbacks.GeneralScheduler(learn, phases=[
        fastai.callbacks.TrainingPhase(pct_start*n_batch).schedule_hp('lr', lr),
        fastai.callbacks.TrainingPhase((1-pct_start)*n_batch).schedule_hp('lr', lr, anneal=fastai.callback.annealing_cos)     
    ])

def fit_flat_cos(learn, n_epoch, lr, pct_start):
    learn.fit(n_epoch, callbacks=[
        flat_then_cosine_sched(learn, len(learn.data.train_dl) * n_epoch, lr=lr, pct_start=pct_start)])
    return learn

##########################################
## General utils
##########################################

import time
from collections import defaultdict 

class Timer():
    def __init__(self, synch=None):
        self.synch = synch or (lambda: None)
        self.synch()
        self.times = [time.perf_counter()]
        self.total_time = 0.0

    def __call__(self, include_in_total=True):
        self.synch()
        self.times.append(time.perf_counter())
        delta_t = self.times[-1] - self.times[-2]
        if include_in_total:
            self.total_time += delta_t
        return delta_t

union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}

def group_by_key(items, func=(lambda v: v)):
    res = defaultdict(list)
    for k, v in items: 
        res[k].append(v) 
    return {k: func(v) for k, v in res.items()}

##########################################
## Pytorch utils
##########################################

def params_with_parents(module):
    for m in module.children():
        yield from params_with_parents(m)
    for name, param in module.named_parameters(recurse=False):
        yield(module, name, param)

def split_params(func, module):
    return group_by_key((func(mod, name), param) for (mod, name, param) in params_with_parents(module))


