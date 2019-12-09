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
            *[fastai2.layers.ResBlock(expansion, ni if i==0 else nf, nf, stride=stride if i==0 else 1,
                      sa=sa if i==(blocks-1) else False, sym=sym, act_cls=act_cls)
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

def smoothed_acc(logits, targets, beta=3.): #replace argmax with soft(arg)max
    return torch.mean(nn.functional.softmax(logits*beta, dim=-1)[torch.arange(0, targets.size(0), device=device), targets])


##########################################
## Fastai v1 adaptors
##########################################

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

import fastai.train
class FuncScheduler(fastai.train.LearnerCallback):
    def __init__(self, func, learn, n_epoch):
        super().__init__(learn)
        self.learn, self.func, self.n_epoch = learn, func, n_epoch
        
    def on_train_begin(self, **kwargs):
        self.step()

    def on_batch_end(self, train, **kwargs):
        if train: self.step()

    def step(self):
        if not hasattr(self, 'iter_vals'):
            n_batch = len(self.learn.data.train_dl)*self.n_epoch 
            self.iter_vals = iter(self.func(x/n_batch) for x in range(0, n_batch+1))
        self.learn.opt.set_stat('lr', next(self.iter_vals))

#####################
## network visualisation (requires pydot)
#####################
class ColorMap(dict):
    palette = ['#'+x for x in (
        'bebada,ffffb3,fb8072,8dd3c7,80b1d3,fdb462,b3de69,fccde5,bc80bd,ccebc5,ffed6f,1f78b4,33a02c,e31a1c,ff7f00,'
        '4dddf8,e66493,b07b87,4e90e3,dea05e,d0c281,f0e189,e9e8b1,e0eb71,bbd2a4,6ed641,57eb9c,3ca4d4,92d5e7,b15928'
    ).split(',')]

    def __missing__(self, key):
        self[key] = self.palette[len(self) % len(self.palette)]
        return self[key]

    def _repr_html_(self):
        css = (
        '.pill {'
            'margin:2px; border-width:1px; border-radius:9px; border-style:solid;'
            'display:inline-block; width:100px; height:15px; line-height:15px;'
        '}'
        '.pill_text {'
            'width:90%; margin:auto; font-size:9px; text-align:center; overflow:hidden;'
        '}'
        )
        s = '<div class=pill style="background-color:{}"><div class=pill_text>{}</div></div>'
        return '<style>'+css+'</style>'+''.join((s.format(color, text) for text, color in self.items()))

sep = '/'

def split(path):
    i = path.rfind(sep) + 1
    return path[:i].rstrip(sep), path[i:]

def make_dot_graph(nodes, edges, direction='LR', **kwargs):
    from pydot import Dot, Cluster, Node, Edge
    class Subgraphs(dict):
        def __missing__(self, path):
            parent, label = split(path)
            subgraph = Cluster(path, label=label, style='rounded, filled', fillcolor='#77777744')
            self[parent].add_subgraph(subgraph)
            return subgraph
    g = Dot(rankdir=direction, directed=True, **kwargs)
    g.set_node_defaults(
        shape='box', style='rounded, filled', fillcolor='#ffffff')
    subgraphs = Subgraphs({'': g})
    for path, attr in nodes:
        parent, label = split(path)
        subgraphs[parent].add_node(
            Node(name=path, label=label, **attr))
    for src, dst, attr in edges:
        g.add_edge(Edge(src, dst, **attr))
    return g

def to_dict(inputs):
    return dict(enumerate(inputs)) if isinstance(inputs, list) else inputs

class DotGraph():
    def __init__(self, graph, size=15, direction='LR'):
        self.nodes = [(k, v) for k, (v,_) in graph.items()]
        self.edges = [(src, dst, {'tooltip': name}) for dst, (_, inputs) in graph.items() for name, src in to_dict(inputs).items()]
        self.size, self.direction = size, direction

    def dot_graph(self, **kwargs):
        return make_dot_graph(self.nodes, self.edges, size=self.size, direction=self.direction,  **kwargs)

    def svg(self, **kwargs):
        return self.dot_graph(**kwargs).create(format='svg').decode('utf-8')
    try:
        import pydot
        _repr_svg_ = svg
    except ImportError:
        def __repr__(self): return 'pydot is needed for network visualisation'

def iter_nodes(graph):
 #   graph = {name: node for (name, node) in graph.items() if node is not None}
    keys = list(graph.keys())
    if not all(isinstance(k, str) for k in keys):
        raise Exception('Node names must be strings.')
    return ((name, (node if isinstance(node, tuple) else (node, [0 if j is 0 else keys[j-1]]))) for (j, (name, node)) in enumerate(graph.items()))

map_ = lambda func, vals: [func(x) for x in vals] if isinstance(vals, list) else {k: func(v) for k,v in vals.items()}
pfx = lambda prefix, name: f'{prefix}/{name}'
external_inputs = lambda graph: set(i for name, (value, inputs) in iter_nodes(graph) for i in inputs if i not in graph)

def bindings(graph, inputs):
    if isinstance(inputs, list): inputs = dict(enumerate(inputs))
    required_inputs = external_inputs(graph)
    missing = [k for k in required_inputs if k not in inputs]
    if len(missing): 
        raise Exception(f'Required inputs {missing} are missing from inputs {inputs} for graph {graph}')
    return inputs

walk = lambda dct, key: walk(dct, dct[key]) if key in dct else key

from functools import singledispatch
@singledispatch
def to_graph(value): 
    return value

def explode(graph, max_levels=-1):
    graph = to_graph(graph)
    if max_levels==0 or not isinstance(graph, dict): return graph
    redirects = {}
    def iter_(graph):
        for name, (value, inputs) in iter_nodes(graph):
            value = explode(value, max_levels-1)
            if isinstance(value, dict):
                #special case empty dict
                if not len(value): 
                    if len(inputs) != 1: raise Exception('Empty graphs (pass-thrus) should have exactly one input')
                    redirects[name] = inputs[0] #redirect to input
                else:
                    bindings_dict = bindings(value, inputs)
                    for n, (val, ins) in iter_nodes(value):
                        yield (pfx(name, n), (val, map_((lambda i: bindings_dict.get(i, pfx(name, i))), ins)))
                    redirects[name] = pfx(name, n) #redirect to previous node
            else:
                yield (name, (value, inputs))
    return {name: (value, map_((lambda i: walk(redirects, i)), inputs)) for name, (value, inputs) in iter_(graph)}

class Network(nn.Module):
    colors = ColorMap()
    def __init__(self, graph, cache_activations=False):
        super().__init__()
        self.cache_activations = cache_activations
        self.graph = {k: (Network(v) if isinstance(v, dict) else v,i) for (k, (v,i)) in iter_nodes(to_graph(graph))}
        
        for path, (val, _) in iter_nodes(self.graph): 
            setattr(self, path.replace('/', '_'), val)
    
    def forward(self, *args):
        prev = args[0]
        outputs = dict(enumerate(args))
        for k, (node, inputs) in iter_nodes(self.graph):
            if k not in outputs: 
                prev = outputs[k] = node(*[outputs[x] for x in inputs])
        if self.cache_activations: self.cache = outputs
        return prev

    def draw(self, **kwargs):
        return DotGraph({p: ({'fillcolor': self.colors[type(v).__name__], 'tooltip': str(v)}, inputs) for p, (v, inputs) in iter_nodes(to_graph(self))}, **kwargs)

    def explode(self, max_levels=-1):
        return Network(explode(self, max_levels))

@to_graph.register(Network)
def f(x): 
    return x.graph    

short_names_ = {
    nn.Conv2d: 'Conv',
    nn.BatchNorm2d: 'Norm',
    nn.ReLU: 'Actn',
    nn.AdaptiveAvgPool2d: 'Avgpool',
    nn.AdaptiveMaxPool2d: 'Maxpool',
    nn.AvgPool2d: 'Avgpool',
    nn.MaxPool2d: 'Maxpool',
    nn.Identity: 'Id',
}

def short_name(typ):
    return short_names_.get(typ, typ.__name__)

@to_graph.register(nn.Sequential)
def f(x):
    if all([(str(i) == k) for i,k in enumerate(x._modules.keys())]):
        mods = {f'{short_name(type(v))}_{k}': v for k,v in x._modules.items()}
    else:
        mods = x._modules
    return dict(iter_nodes(mods))

class Mul(nn.Module):
    def forward(self, x, y): return x * y

class Add(nn.Module):
    def forward(self, x, y): return x + y

class SplitMerge(Network):
    def __init__(self, branches, merge=Add, **post):
        if isinstance(branches, list):
            branches = {f'branch{i}': branch for i, branch in enumerate(branches)}
        graph = union({'in': nn.Identity()}, {k: (v, ['in']) for k,v in branches.items()})
        graph[short_name(merge)] = (merge(), list(branches.keys()))
        if post: graph = union(graph, post)
        super().__init__(graph)