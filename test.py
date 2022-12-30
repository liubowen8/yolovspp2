from statistics import mode
from turtle import forward, right
import torch, cv2
import numpy as np
import torch.nn as nn
x=cv2.imread("/home/ifpp/liubowen/code/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/ratest_result.jpg")
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cov1=nn.Conv2d(3,10,3)
        self.l1=nn.Linear(20676040,1)
    
    def forward(self, x):
        x=self.cov1(x)
        x=x.view(-1)
        x=self.l1(x)
        return x


x = np.transpose(x, (2, 0, 1)) / 255.
x[0]-=np.mean(x[0])
x[1]-=np.mean(x[1])
x[2]-=np.mean(x[2])
x=torch.from_numpy(x).float()
x=x.unsqueeze(0)
m=model()


cri=nn.MSELoss(size_average=False)
opti=torch.optim.SGD(m.parameters(), lr=0.001)

print(list(m.children()))
print(list(m.children())[0])
for p in list(m.children())[0].parameters():
    print(p)

for epoch in range(2):
    y=m(x).to(torch.float32)
    opti.zero_grad()
    print(y)
    ll=torch.from_numpy(np.array([0.0])).to(torch.float32)
    print(ll)
    loss=cri(y, ll)
    loss.backward()
    opti.step()
    print(m.state_dict()[list(m.state_dict().keys())[0]][-1],11111111)





