from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce

class LambdaBase(nn.Sequential):
    def __init__(self, *args):
        super(LambdaBase, self).__init__(*args)

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def __init__(self, *args):
        super(Lambda, self).__init__(*args)
        self.lambda_func = identity

    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def __init__(self, *args):
        super(LambdaMap, self).__init__(*args)
        self.lambda_func = identity

    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def __init__(self, *args):
        super(LambdaReduce, self).__init__(*args)
        self.lambda_func = add

    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))

def identity(x): return x

def add(x, y): return x + y

resnext101_32x4d_features = nn.Sequential( # Sequential,
    nn.Conv2d(3,64,(7, 7),(2, 2),(3, 3),1,1,bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d((3, 3),(2, 2),(1, 1)),
    nn.Sequential( # Sequential,
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(64,128,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(128,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(256),
                ),
                nn.Sequential( # Sequential,
                    nn.Conv2d(64,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(256),
                ),
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(256,128,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(128,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(256),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(256,128,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(128,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(256),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential( # Sequential,
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(256,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256,256,(3, 3),(2, 2),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(512),
                ),
                nn.Sequential( # Sequential,
                    nn.Conv2d(256,512,(1, 1),(2, 2),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(512),
                ),
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(512,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(512),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(512,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(512),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(512,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(512),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential( # Sequential,
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(512,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(2, 2),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                nn.Sequential( # Sequential,
                    nn.Conv2d(512,1024,(1, 1),(2, 2),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential( # Sequential,
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024,1024,(3, 3),(2, 2),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(2048),
                ),
                nn.Sequential( # Sequential,
                    nn.Conv2d(1024,2048,(1, 1),(2, 2),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(2048),
                ),
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(2048,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(2048),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential( # Sequential,
            LambdaMap( # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(2048,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,32,bias=False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                    nn.BatchNorm2d(2048),
                ),
                Lambda(), # Identity,
            ),
            LambdaReduce(), # CAddTable,
            nn.ReLU(),
        ),
    )
)