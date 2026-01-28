import torch
import torch.nn as nn
from torch import sigmoid

# Normalization helper

def make_norm(act_norm: str, ch: int) -> nn.Module:
    """ Return an activation-normalization layer (or Identity for 'none') """
    
    act_norm = act_norm.lower()

    if act_norm == "none":
        return nn.Identity()

    if act_norm == "batch":
        return nn.BatchNorm2d(ch, momentum=0.005)

    if act_norm == "group":
        groups = 8 if ch % 8 == 0 else 1
        return nn.GroupNorm(groups, ch)

    if act_norm == "instance":
        return nn.InstanceNorm2d(ch, affine=True)

    raise ValueError("act_norm must be one of ['none','batch','group','instance']")


# Core blocks

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, act_norm: str):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            make_norm(act_norm, out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            make_norm(act_norm, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, act_norm):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, act_norm)

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, act_norm):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, act_norm),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, act_norm):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, act_norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# Attention blocks

class AttentionBlock(nn.Module):
    def __init__(self, act_norm: str, out_ch: int, n_coeff: int):
        super().__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(out_ch, n_coeff, 1),
            make_norm(act_norm, n_coeff),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(out_ch, n_coeff, 1),
            make_norm(act_norm, n_coeff),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coeff, 1, 1),
            make_norm(act_norm, 1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        a = self.relu(self.W_gate(g) + self.W_x(x))
        a = self.psi(a)
        return x * a


class Up_att(nn.Module):
    def __init__(self, in_ch, out_ch, act_norm):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.att = AttentionBlock(act_norm, out_ch, max(1, out_ch // 2))
        self.conv = DoubleConv(in_ch, out_ch, act_norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.att(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# Output

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


# UNet

class UNet(nn.Module):
    """
    U-Net for CR detection.

    act_norm: 'none' | 'batch' | 'group' | 'instance'    
    """
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        hidden: int = 32,
        act_norm: str = "batch",
        att: bool = False,
        deeper: bool = False,
    ):
        super().__init__()
        self.deeper = deeper
        self.att = att

        self.inc = InConv(n_channels, hidden, act_norm)
        self.down1 = Down(hidden, hidden * 2, act_norm)

        if deeper:
            self.down2 = Down(hidden * 2, hidden * 4, act_norm)
            self.up7 = (Up_att if att else Up)(hidden * 4, hidden * 2, act_norm)

        self.up8 = (Up_att if att else Up)(hidden * 2, hidden, act_norm)
        self.outc = OutConv(hidden, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)

        if self.deeper:
            x3 = self.down2(x2)
            x = self.up7(x3, x2)
            x = self.up8(x, x1)
        else:
            x = self.up8(x2, x1)

        return sigmoid(self.outc(x))

class WrappedModel(nn.Module):
    def __init__(self, network):
        super(type(self), self).__init__()
        self.module = network

    def forward(self, *x):
        return self.module(*x)
    