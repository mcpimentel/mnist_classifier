import torch
import torch.nn as nn
import torch.nn.functional as F

N_SAMPLES = 5
MEAN = 0.1307
STANDARD_DEVIATION = 0.3081

class LeNet_standard(nn.Module):
    def __init__(self):
        super(LeNet_standard, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # format input according to canvas!
        x = x.reshape(280, 280, 4)
        x = torch.narrow(x, dim=2, start=3, length=1)
        x = x.reshape(1, 1, 280, 280)
        x = F.avg_pool2d(x, 10, stride=10)
        x = x / 255
        x = (x - MEAN) / STANDARD_DEVIATION
        
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.softmax(self.fc2(x), dim=1)
        return torch.unsqueeze(x, 0)


class LeNet_dropout(nn.Module):
    def __init__(self):
        super(LeNet_dropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)

    def _sample(self, x):
        x = F.relu(F.max_pool2d(F.dropout(self.conv1(x), training=True), 2))
        x = F.relu(F.max_pool2d(F.dropout(self.conv2(x), training=True), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=True)
        x = self.fc2(x)
        return x 

    def forward(self, x):
        # format input according to canvas!
        x = x.reshape(280, 280, 4)
        x = torch.narrow(x, dim=2, start=3, length=1)
        x = x.reshape(1, 1, 280, 280)
        x = F.avg_pool2d(x, 10, stride=10)
        x = x / 255
        x = (x - MEAN) / STANDARD_DEVIATION

        outputs = []
        for i in range(N_SAMPLES):
            outputs.append(torch.unsqueeze(F.softmax(self._sample(x), dim=1), 0))
        outputs = torch.cat(outputs, 0)
        output_mean = outputs.mean(0)
        output_var = torch.sum((outputs-output_mean)**2, 0)/(N_SAMPLES-1)
        return torch.cat([output_mean, torch.tensor([output_var.max()]).reshape([-1,1])], 1)


class LeNet_manualdropout(nn.Module):
    def __init__(self):
        super(LeNet_manualdropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)
        self.myrand = myrandom()

    def _randm(self, L):
        # for the manual dropout (p=.5)
        a = torch.tensor([self.myrand.getuniform() for x in range(L)]) < 0.5    
        return a

    def _sample(self, x):
        x = self.conv1(x)
        x = x - self._randm(18432).reshape([1,32,24,24])*x      # manual dropout
        x = F.relu(F.max_pool2d(x*2, 2))                        # don't forget to multiply by 1/(1-p), i.e. 2
        x = self.conv2(x)
        x = x - self._randm(4096).reshape([1,64,8,8])*x         # manual dropout
        x = F.relu(F.max_pool2d(x*2, 2))                        # don't forget to multiply by 1/(1-p), i.e. 2
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = x - self._randm(128).reshape([1,128])*x             # manual dropout
        x = self.fc2(x*2)                                       # don't forget to multiply by 1/(1-p), i.e. 2
        return x

    def forward(self, x):
        # format input according to canvas!
        x = x.reshape(280, 280, 4)
        x = torch.narrow(x, dim=2, start=3, length=1)
        x = x.reshape(1, 1, 280, 280)
        x = F.avg_pool2d(x, 10, stride=10)
        x = x / 255
        x = (x - MEAN) / STANDARD_DEVIATION

        outputs = []
        for i in range(N_SAMPLES):
            outputs.append(torch.unsqueeze(F.softmax(self._sample(x), dim=1), 0))
        outputs = torch.cat(outputs, 0)
        output_mean = outputs.mean(0)
        output_var = torch.sum((outputs-output_mean)**2, 0)/(N_SAMPLES-1)
        return torch.cat([output_mean, output_var], 1)
    

class myrandom:
    kz=36969
    kw=18000
    k3=65535
    maxz=kz*k3+(2<<16)
    maxw=kw*k3+(2<<16)
    max=(maxz<<16 )+maxw
    # Optionally initiate with different seed. Two numbers below 2<<16
    def __init__(self,z=123456789,w=98764321):
        self.m_w = w
        self.m_z = z
    def step(self):
        self.m_z = self.kz * (self.m_z & self.k3) + (self.m_z >> 16)  
        self.m_w = self.kw * (self.m_w & self.k3) + (self.m_w >> 16)       
    def get(self):
        self.step()
        return (self.m_z << 16) + self.m_w
    def time_reseed(self):
        # yes, sure, move out import if you like to
        import time
        t=int(time.time())
        # completely made up way to got two new numbers below 2<<16
        self.m_z = (self.m_z+(t*34567891011)) & ((2<<16)-1)
        self.m_w = (self.m_w+(t*10987654321)) & ((2<<16)-1)
        self.step()
    def getuniform(self):
        return self.get()*1.0/self.max

