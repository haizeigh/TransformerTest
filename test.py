print("test")
# import torch.nn as nn
import torch
# from attention.attention import Attention
import torch.nn as nn

a = torch.arange(6).reshape(3, 2)
print(a)
index= torch.LongTensor([[2,1], [1,1]]) # 这里index和input必须有一样的维度...
print(index)
b = torch.gather(a,0,index)
print(b)


i = index[0]

print(i)
print(a[i.numpy()])


def foo(num):
    print("starting...")
    while num<10:
        num=num+2
        yield num
for n in foo(0):
    print(n)



label = torch.Tensor([1, 1, 0])
pred = torch.Tensor([3, 2, 1])
pred_sig = torch.sigmoid(pred)
# BCELoss must be used together with sigmoid
loss = nn.BCELoss()
print(loss(pred_sig, label))
# BCEWithLogitsLoss
loss = nn.BCEWithLogitsLoss()
print(loss(pred, label))