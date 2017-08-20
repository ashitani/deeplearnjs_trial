import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from matplotlib import pyplot as plt
import time

def get_batch(n):
    x = np.random.random(n)
    y = np.exp(x)
    return x,y

class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
             l1=L.Linear(1, 16),
             l2=L.Linear(16, 32),
             l3=L.Linear(32, 1),
             # l1=L.Linear(1, 256),
             # l2=L.Linear(256, 1024),
             # l3=L.Linear(1024, 1),
        )
        print(self.l1.W.data)
        print(self.l1.b.data)

    def __call__(self,x,t):
        return F.mean_squared_error(self.predict(x),t)

    def  predict(self,x):
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        h3 = F.leaky_relu(self.l3(h2))
        return h3

    def get(self,x):
        return self.predict(Variable(np.array([x]).astype(np.float32).reshape(1,1))).data[0][0]

model = MyChain()
optimizer = optimizers.SGD(lr=0.01)
optimizer.setup(model)

start_time = time.time()

xs,ys = get_batch(100)

for i in range(100):
    for j in range(100):
        x=xs[j]
        y=ys[j]
        x_ = Variable(x.astype(np.float32).reshape(1,1))
        t_ = Variable(y.astype(np.float32).reshape(1,1))

        model.zerograds()
        loss=model(x_,t_)
        loss.backward()
        optimizer.update()
        if j==0:
            print(loss)

print(model.get(0.2))
print(np.exp(0.2))

elapsed_time=time.time()-start_time
print("elasped time for training: %f [sec]" % elapsed_time)

start_time = time.time()

import random
for i in range(1000):
    x=random.random()
    y=model.get(x)

elapsed_time=time.time()-start_time
print("elasped time for prediction: %f [msec/cycle]" % (elapsed_time))



