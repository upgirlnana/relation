import torch
from torch import nn

class Ctritic_Value(nn.Module):
    def __init__(self,data,a,down_ratio=8):
        #因为没有加super所以才报出cannt asssign before __init__
        super(Ctritic_Value, self).__init__()
        self.x=data
        self.down_ratio=down_ratio
        b, c, h, w= self.x.size()
        # self.x_conv = nn.Conv2d(c, 1, 1, 1, 0, False)
        self.x_conv=nn.Conv2d(in_channels=c, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.x_fc=nn.Linear(h*w,(h*w)//self.down_ratio)
        self.a_fc = nn.Linear(h * w, (h * w)//self.down_ratio)
        self.v_fc=nn.Linear(2*(h * w) // self.down_ratio,1)
        # self.x_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=c, out_channels=1,
        #               kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(self.inter_spatial),
        #     nn.ReLU()
        # )
        # self.a_conv=nn.Sequential()
        # torch.flatten()
    def forward(self, data,a):

        x=self.x_conv(data)
        b, c, _, _ = x.size()
        x=x.view(b,-1)
        # x=torch.flatten(x,start_dim=1)
        x=self.x_fc(x)

        # a=torch.flatten(a,start_dim=1)
        bb, bc, _, _ = a.size()
        a=a.view(bb,-1)
        a=self.a_fc(a)
        xa=torch.cat((x,a),dim=1)
        v=self.v_fc(xa)
        return v

height=256
width=128
down_ratio=8
# data=torch.randn(4,256, 64,32)
# a_map=torch.randn(4,1,64,32)
# data1=torch.randn(4,256, 32,16)
# a_map1=torch.randn(4,1,32,16)
data2=torch.randn(4,256, 16,8)
a_map2=torch.randn(4,1,16,8)
# # 1, 256, 64, 32
# # 1, 512, 32, 16
# # 1, 1024, 16, 8
# # 1, 2048, 16, 8
# a_map=torch.randn(4,1,64,32)
# net=Ctritic_Value(data,a_map,down_ratio)
# net=Ctritic_Value(data1,a_map1,down_ratio)
net=Ctritic_Value(data2,a_map2,down_ratio)
out=net(data2,a_map2)
print(net)


