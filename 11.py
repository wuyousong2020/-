import numpy as np

def _softmax(x):
    if x.ndim == 2:
        c = np.max(x, axis=1)
        x = x.T - c  # 溢出对策
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    c = np.max(x)
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x)

def cross_entropy_error(p, y):
    delta = 1e-7
    batch_size = p.shape[0]
    return -np.sum(y * np.log(p + delta)) / batch_size

class simpleNet:
    def __init__(self):
        np.random.seed(0)
        self.W = np.random.randn(2,3)

    def forward(self,x):
        return np.dot(x,self.W)

    def loss(self,x,y):
        z = self.forward(x)
        p = _softmax(z)
        loss = cross_entropy_error(p, y)
        return loss

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val #还原值
        it.iternext()
    return grad

def gradient_descent(f,init_x,lr=0.01,step_num=1000):
    x =init_x
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr*grad
        return x


net = simpleNet()
print(net.W)
x= np.array([[0.6,0.9]])
p = net.forward(x)

print('预测值为：',p)
print('预测的类别为: ',np.argmax(p))

y = np.array([0,0,1]) #输入正确类别
print(net.loss(x,y))

f =  lambda w: net.loss(x,y)
for i in range(500):
    dw = gradient_descent(f,net.W) #需要更新的主要是W
    print('损失值变为： ',cross_entropy_error(_softmax(np.dot(x,dw)),y))
print(dw)

print('损失值变为： ',cross_entropy_error(_softmax(np.dot(x,dw)),y))
print('预测类别为: ',np.argmax(np.dot(x,dw)))
