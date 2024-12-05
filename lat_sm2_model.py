import torch
import torch.nn as nn

class LSM(nn.Module):

    def __init__(self):
        super().__init__()
  
        self.layers = nn.Sequential(
            nn.Linear(20, 32),
            nn.Softplus(),
            nn.Linear(32, 64),
            nn.Softplus(),
            nn.Linear(64, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64),
            nn.Softplus(),
            nn.Linear(64, 1),)

    def forward(self, x):
        return self.layers(x)

class LSM2(nn.Module):

    def __init__(self):
        super().__init__()
  
        self.layers = nn.Sequential(
            nn.Linear(40, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64),
            nn.Softplus(),
            nn.Linear(64, 1),)

    def forward(self, x):
        return self.layers(x)

class LSM3(nn.Module):

    def __init__(self):
        super().__init__()
  
        self.layers = nn.Sequential(
            nn.Linear(40, 128),
            nn.Softplus(),
            nn.Linear(128, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, 128),
            nn.Softplus(),
            nn.Linear(128, 64),
            nn.Softplus(),
            nn.Linear(64, 1),)

    def forward(self, x):
        return self.layers(x)

class LSM4(nn.Module):

    def __init__(self):
        super().__init__()
  
        self.layers = nn.Sequential(
            nn.Linear(60, 128),
            nn.Softplus(),
            nn.Linear(128, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, 128),
            nn.Softplus(),
            nn.Linear(128, 64),
            nn.Softplus(),
            nn.Linear(64, 1),)

    def forward(self, x):
        return self.layers(x)

class LSM5(nn.Module):

    def __init__(self):
        super().__init__()
  
        self.layers = nn.Sequential(
            nn.Linear(40, 128),
            nn.Softplus(),
            nn.Linear(128, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, 128),
            nn.Softplus(),
            nn.Linear(128, 64),
            nn.Softplus(),
            nn.Linear(64, 1),)

    def forward(self, x):
        return self.layers(x)

class LSM_SV_MN(nn.Module):

    def __init__(self):
        super().__init__()

        self.l1_down = nn.Sequential(
            nn.Linear(40, 128),
            nn.Softplus(),)

        self.l2_down = nn.Sequential(
            nn.Linear(128, 64),
            nn.Softplus(),)

        self.l3_down = nn.Sequential(
            nn.Linear(64, 32),
            nn.Softplus(),)

        self.l3_up = nn.Sequential(
            nn.Linear(32, 64),
            nn.Softplus(),)

        self.l2_up = nn.Sequential(
            nn.Linear(64, 128),
            nn.Softplus(),)
        
        self.l1_up = nn.Sequential(
            nn.Linear(128, 40),
            nn.Softplus(),)

    def forward(self, x):
        l1_down_out = self.l1_down(x)
        l2_down_out = self.l2_down(l1_down_out)
        l3_down_out = self.l3_down(l2_down_out)
        l3_up_out = self.l3_up(l3_down_out)
        l2_up_out = self.l2_up(l3_up_out + l2_down_out)
        l1_up_out = self.l1_up(l2_up_out + l1_down_out)
        return l1_up_out

class LSMPoly(nn.Module):

    def __init__(self):
        super().__init__()
  
        self.layers = nn.Sequential(
            nn.Linear(512*5, 512*4),
            nn.Softplus(),
            nn.Linear(512*4, 512*4),
            nn.Softplus(),
            nn.Linear(512*4, 512*4),
            nn.Softplus(),
            nn.Linear(512*4, 512*2),
            nn.Softplus(),
            nn.Linear(512*2, 512*2),
            nn.Softplus(),
            nn.Linear(512*2, 512*4),
            nn.Softplus(),
            nn.Linear(512*4, 512*5),)

    def forward(self, x):
        return self.layers(x)

class LSMPoly2(nn.Module):

    def __init__(self):
        super().__init__()
  
        self.layers = nn.Sequential(
            nn.Linear(512*5, 512*4),
            nn.Softplus(),
            nn.Linear(512*4, 512*4),
            nn.Softplus(),
            nn.Linear(512*4, 512*4),
            nn.Softplus(),
            nn.Linear(512*4, 512*2),
            nn.Softplus(),
            nn.Linear(512*2, 512),
            nn.Softplus(),
            nn.Linear(512, 64),
            nn.Softplus(),
            nn.Linear(64, 1),)

    def forward(self, x):
        return self.layers(x)

class LSMPoly64(nn.Module):

    def __init__(self):
        super().__init__()
  
        self.layers = nn.Sequential(
            nn.Linear(64*5, 64*4),
            nn.Softplus(),
            nn.Linear(64*4, 64*4),
            nn.Softplus(),
            nn.Linear(64*4, 64*4),
            nn.Softplus(),
            nn.Linear(64*4, 64*2),
            nn.Softplus(),
            nn.Linear(64*2, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 1),)

    def forward(self, x):
        return self.layers(x)

class LSMPoly64Upd(nn.Module):

    def __init__(self):
        super().__init__()
  
        self.layers = nn.Sequential(
            nn.Linear(64*5, 64*3),
            nn.Softplus(),
            nn.Linear(64*3, 64*3),
            nn.Softplus(),
            nn.Linear(64*3, 64*3),
            nn.Softplus(),
            nn.Linear(64*3, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 1),
            )

    def forward(self, x):
        return self.layers(x)

class Poly_sm(nn.Module):

    def __init__(self, n_mod=5, size_z=128):
        super().__init__()
        self.size_z = size_z
        self.n_mod = n_mod
  
        self.layers = nn.Sequential(
            nn.Linear(self.size_z*self.n_mod, self.size_z*4),
            nn.Softplus(),
            nn.Linear(self.size_z*4, self.size_z*2),
            nn.Softplus(),
            # nn.Linear(self.size_z*4, self.size_z*4),
            # nn.Softplus(),
            # nn.Linear(self.size_z*4, self.size_z*2),
            # nn.Softplus(),
            nn.Linear(self.size_z*2, self.size_z*2),
            nn.Softplus(),
            nn.Linear(self.size_z*2, self.size_z*4),
            nn.Softplus(),
            nn.Linear(self.size_z*4, self.size_z*self.n_mod),)

    def forward(self, x):
        return self.layers(x)

class Poly_sm2(nn.Module):

    def __init__(self, n_mod=5, size_z=128):
        super().__init__()
        self.size_z = size_z
        self.n_mod = n_mod
  
        self.layers = nn.Sequential(
            nn.Linear(self.size_z*self.n_mod, self.size_z*4),
            nn.Softplus(),
            nn.Linear(self.size_z*4, self.size_z*2),
            nn.Softplus(),
            nn.Linear(self.size_z*2, self.size_z*2),
            nn.Softplus(),
            nn.Linear(self.size_z*2, self.size_z*2),
            nn.Softplus(),
            nn.Linear(self.size_z*2, self.size_z*2),
            nn.Softplus(),
            nn.Linear(self.size_z*2, self.size_z*4),
            nn.Softplus(),
            nn.Linear(self.size_z*4, self.size_z*self.n_mod),)

    def forward(self, x):
        return self.layers(x)

class Poly_sm3(nn.Module):

    def __init__(self, n_mod=5, size_z=128):
        super().__init__()
        self.size_z = size_z
        self.n_mod = n_mod
  
        self.layers = nn.Sequential(
            nn.Linear(self.size_z*self.n_mod, self.size_z*self.n_mod*4),
            nn.Softplus(),
            nn.Linear(self.size_z*n_mod*4, self.size_z*self.n_mod*2),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*2, self.size_z*self.n_mod*2),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*2, self.size_z*self.n_mod*2),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*2, self.size_z*self.n_mod*2),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*2, self.size_z*self.n_mod*4),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*4, self.size_z*self.n_mod),)

    def forward(self, x):
        return self.layers(x)

# class Poly_sm(nn.Module):

#     def __init__(self, n_mod=5, size_z=128):
#         super().__init__()
#         self.size_z = size_z
#         self.n_mod = n_mod
  
#         self.layers = nn.Sequential(
#             nn.Linear(self.size_z*self.n_mod, self.size_z*self.n_mod),
#             nn.Softplus(),
#             nn.Linear(self.size_z*self.n_mod, self.size_z*self.n_mod),
#             nn.Softplus(),
#             nn.Linear(self.size_z*self.n_mod, self.size_z*self.n_mod),
#             nn.Softplus(),
#             nn.Linear(self.size_z*self.n_mod, self.size_z*self.n_mod),
#             nn.Softplus(),
#             nn.Linear(self.size_z*self.n_mod, self.size_z*self.n_mod),
#             nn.Softplus(),
#             nn.Linear(self.size_z*self.n_mod, self.size_z*self.n_mod),
#             nn.Softplus(),
#             nn.Linear(self.size_z*self.n_mod, self.size_z*self.n_mod),)

#     def forward(self, x):
#         return self.layers(x)

class LSMPoly64_sm(nn.Module):

    def __init__(self, n_mod=5, size_z=128):
        super().__init__()
        self.size_z = size_z
        self.n_mod = n_mod
  
        self.layers = nn.Sequential(
            nn.Linear(self.size_z*self.n_mod, self.size_z*self.n_mod*2),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*2, self.size_z*self.n_mod*6),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*6, self.size_z*self.n_mod*6),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*6, self.size_z*self.n_mod*4),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*4, self.size_z*self.n_mod*2),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*2, self.size_z*self.n_mod),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod, self.size_z*self.n_mod),)

    def forward(self, x):
        return self.layers(x)

class LSMPoly64_deep(nn.Module):

    def __init__(self, n_mod=5, size_z=128):
        super().__init__()
        self.size_z = size_z
        self.n_mod = n_mod
  
        self.layers = nn.Sequential(
            nn.Linear(self.size_z*self.n_mod, self.size_z*self.n_mod*2),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*2, self.size_z*self.n_mod*6),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*6, self.size_z*self.n_mod*6),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*6, self.size_z*self.n_mod*5),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*5, self.size_z*self.n_mod*5),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*5, self.size_z*self.n_mod*5),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*5, self.size_z*self.n_mod*5),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*5, self.size_z*self.n_mod*4),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*4, self.size_z*self.n_mod*2),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*2, self.size_z*self.n_mod),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod, self.size_z*self.n_mod),)

    def forward(self, x):
        return self.layers(x)

class LSMPoly64_smOLD(nn.Module):

    def __init__(self, n_mod=5, size_z=128):
        super().__init__()
        self.size_z = size_z
        self.n_mod = n_mod
  
        self.layers = nn.Sequential(
            nn.Linear(self.size_z*self.n_mod, self.size_z*self.n_mod),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod, self.size_z*self.n_mod*3),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*3, self.size_z*self.n_mod*3),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*3, self.size_z*self.n_mod*2),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*2, self.size_z*self.n_mod*1),
            nn.Softplus(),
            nn.Linear(self.size_z*self.n_mod*1, self.size_z*5),
            nn.Softplus(),
            nn.Linear(self.size_z*5, self.size_z*self.n_mod),)

    def forward(self, x):
        return self.layers(x)


class LSMPoly64_em(nn.Module):

    def __init__(self, n_mod=5, size_z=128):
        super().__init__()
        self.size_z = size_z
        self.n_mod = n_mod
  
        self.layers = nn.Sequential(
            nn.Linear(self.size_z*self.n_mod, self.size_z*4),
            nn.Softplus(),
            nn.Linear(self.size_z*4, self.size_z*4),
            nn.Softplus(),
            nn.Linear(self.size_z*4, self.size_z*4),
            nn.Softplus(),
            nn.Linear(self.size_z*4, self.size_z*2),
            nn.Softplus(),
            nn.Linear(self.size_z*2, self.size_z*2),
            nn.Softplus(),
            nn.Linear(self.size_z*2, self.size_z),
            nn.Softplus(),
            nn.Linear(self.size_z, 1),)

    def forward(self, x):
        return self.layers(x)

class LSMPoly64_dsm2(nn.Module):

    def __init__(self):
        super().__init__()

        self.l1 = nn.Linear(64*5, 64*4)
        self.t1 = nn.Linear(1,64*4)

        self.l2 = nn.Linear(64*4, 64*4)
        self.t2 = nn.Linear(1,64*4)

        self.l3 = nn.Linear(64*4, 64*3)
        self.t3 = nn.Linear(1,64*3)

        self.l4 = nn.Linear(64*3, 64*2)
        self.t4 = nn.Linear(1,64*2)

        self.l5 = nn.Linear(64*2, 64*2)
        self.t5 = nn.Linear(1,64*2)

        self.l6 = nn.Linear(64*2, 64*3)
        self.t6 = nn.Linear(1,64*3)

        self.l7 = nn.Linear(64*3, 64*5)
        self.t7 = nn.Linear(1,64*5)

        self.act = nn.ReLU()
  
    def forward(self, x, t):
        out1 = self.act(self.l1(x) + self.t1(t))
        out2 = self.act(self.l2(out1) + self.t2(t))
        out3 = self.act(self.l3(out2) + self.t3(t))
        out4 = self.act(self.l4(out3) + self.t4(t))
        out5 = self.act(self.l5(out4) + self.t5(t))
        out6 = self.act(self.l6(out5) + self.t6(t))
        out7 = self.act(self.l7(out6) + self.t7(t))
        return out7

class LSMPoly4_64(nn.Module):

    def __init__(self):
        super().__init__()
  
        self.layers = nn.Sequential(
            nn.Linear(64*4, 64*3),
            nn.Softplus(),
            nn.Linear(64*3, 64*2),
            nn.Softplus(),
            nn.Linear(64*2, 64*2),
            nn.Softplus(),
            nn.Linear(64*2, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 1),)

    def forward(self, x):
        return self.layers(x)

class LSMPoly3_64(nn.Module):

    def __init__(self, size_z=64):
        super().__init__()
        self.size_z = size_z
  
        self.layers = nn.Sequential(
            nn.Linear(self.size_z*3, self.size_z*2),
            nn.Softplus(),
            nn.Linear(self.size_z*2, self.size_z*2),
            nn.Softplus(),
            nn.Linear(self.size_z*2, self.size_z*2),
            nn.Softplus(),
            nn.Linear(self.size_z*2, self.size_z),
            nn.Softplus(),
            nn.Linear(self.size_z, self.size_z),
            nn.Softplus(),
            nn.Linear(self.size_z, 1),)

    def forward(self, x):
        return self.layers(x)

class LSMPoly2_64(nn.Module):

    def __init__(self, size_z=64):
        super().__init__()
        self.size_z = size_z

        self.layers = nn.Sequential(
            nn.Linear(self.size_z*2, self.size_z*2),
            nn.Softplus(),
            nn.Linear(self.size_z*2, self.size_z*2),
            nn.Softplus(),
            nn.Linear(self.size_z*2, self.size_z*2),
            nn.Softplus(),
            nn.Linear(self.size_z*2, self.size_z),
            nn.Softplus(),
            nn.Linear(self.size_z, self.size_z),
            nn.Softplus(),
            nn.Linear(self.size_z, 1),)

    def forward(self, x):
        return self.layers(x)

class EM3(nn.Module):
    def __init__(self, pair_es):
        super().__init__()
        self.pair_es = pair_es

    def forward(self, z1, z2, z3):
        e_out = self.pair_es['12'](torch.cat([z1,z2], dim=1)) + \
            self.pair_es['13'](torch.cat([z1,z3], dim=1)) + \
            self.pair_es['23'](torch.cat([z2,z3],dim=1))
        return e_out

class EM3_train(nn.Module):
    def __init__(self, size_z):
        super().__init__()
        self.size_z = size_z
        self.e12 = LSMPoly2_64(self.size_z)
        self.e13 = LSMPoly2_64(self.size_z)
        self.e23 = LSMPoly2_64(self.size_z)

    def forward(self, z):
        z1 = z[:,:self.size_z]
        z2 = z[:,self.size_z:2*self.size_z]
        z3 = z[:,self.size_z*2:]
        e_out = self.e12(torch.cat([z1,z2], dim=1)) + \
            self.e13(torch.cat([z1,z3], dim=1)) + \
            self.e23(torch.cat([z2,z3],dim=1))
        return e_out

class EM5(nn.Module):
    def __init__(self, pair_es):
        super().__init__()
        self.pair_es = pair_es

    def forward(self, z0, z1, z2, z3, z4):
        e_out = self.pair_es['01'](torch.cat([z0,z1], dim=1)) + \
                self.pair_es['02'](torch.cat([z0,z2], dim=1)) + \
                self.pair_es['03'](torch.cat([z0,z3], dim=1)) + \
                self.pair_es['04'](torch.cat([z0,z4], dim=1)) + \
                self.pair_es['12'](torch.cat([z1,z2], dim=1)) + \
                self.pair_es['13'](torch.cat([z1,z3], dim=1)) + \
                self.pair_es['14'](torch.cat([z1,z4], dim=1)) + \
                self.pair_es['23'](torch.cat([z2,z3], dim=1)) + \
                self.pair_es['24'](torch.cat([z2,z4],dim=1)) + \
                self.pair_es['34'](torch.cat([z3,z4], dim=1))
        return e_out

class LSMPolyConv(nn.Module):

    def __init__(self, n_mod):
        super().__init__()
        self.n_mod = n_mod
  
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=self.n_mod, out_channels=32, kernel_size=4), 
            nn.Softplus(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.Softplus(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.Softplus(),
            nn.Flatten(),
            nn.Linear(128,1),
            )

    def forward(self, x):
        return self.layers(x)

class Quad2Poly(nn.Module):

    def __init__(self,size_z=64):
        super().__init__()

        self.size_z = size_z
        self.w12 = torch.nn.Parameter(torch.randn(1,size_z), requires_grad=True)
        self.w21 = torch.nn.Parameter(torch.randn(1,size_z), requires_grad=True)

        # self.w12 = nn.Linear(size_z,1) # Wz1z2 10x10 uz1
        # self.w21 = nn.Linear(size_z*size_z,1)
        self.u1 = nn.Linear(size_z, 1)
        self.u2 = nn.Linear(size_z,1)

    def forward(self, z):
        z1, z2 = z[:,:self.size_z], z[:,self.size_z:]
        z1z2 = z1 * self.w12 * z2
        z2z1 = z2 * self.w21 * z1
        u1_out = self.u1(z1)
        u2_out = self.u2(z2)
        return z1z2.sum() + z2z1.sum() + u1_out + u2_out

class LSM_FMK10(nn.Module):

    def __init__(self, size_z=10, n=3):
        super().__init__()
        self.size_z = size_z
        self.n = n
  
        self.layers = nn.Sequential(
            nn.Linear(self.size_z*n, 64*2),
            nn.Softplus(),
            nn.Linear(64*2, 64*2),
            nn.Softplus(),
            nn.Linear(64*2, 64*2),
            nn.Softplus(),
            nn.Linear(64*2, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 1),)

    def forward(self, x):
        return self.layers(x)

class ULSM(nn.Module):

    def __init__(self, size_z=10):
        super().__init__()
        self.size_z = size_z
  
        self.layers = nn.Sequential(
            nn.Linear(self.size_z, self.size_z*3),
            nn.Softplus(),
            nn.Linear(self.size_z*3, self.size_z*3),
            nn.Softplus(),
            nn.Linear(self.size_z*3, self.size_z*3),
            nn.Softplus(),
            nn.Linear(self.size_z*3, self.size_z),
            nn.Softplus(),
            nn.Linear(self.size_z, 1),)

    def forward(self, x):
        return self.layers(x)

class USM(nn.Module):

    def __init__(self, size_z=10):
        super().__init__()
        self.size_z = size_z
  
        self.layers = nn.Sequential(
            nn.Linear(self.size_z, self.size_z*3),
            nn.Softplus(),
            nn.Linear(self.size_z*3, self.size_z*3),
            nn.Softplus(),
            nn.Linear(self.size_z*3, self.size_z*3),
            nn.Softplus(),
            nn.Linear(self.size_z*3, self.size_z),
            nn.Softplus(),
            nn.Linear(self.size_z, self.size_z),)

    def forward(self, x):
        return self.layers(x)

class USM2(nn.Module):

    def __init__(self, size_z=10):
        super().__init__()
        self.size_z = size_z
        self.label_lin_size = 10

        self.label_linear1 = nn.Linear(1,self.label_lin_size)
        self.linear1 = nn.Linear(self.size_z+self.label_lin_size, self.size_z*3)
        self.label_linear2 = nn.Linear(1,self.label_lin_size)
        self.linear2 = nn.Linear(self.size_z*3 + self.label_lin_size, self.size_z*3)
        self.label_linear3 = nn.Linear(1,self.label_lin_size)
        self.linear3 = nn.Linear(self.size_z*3 + self.label_lin_size, self.size_z*3)
        self.linear4 = nn.Linear(self.size_z*3, self.size_z)
        self.act = nn.Softplus()

    def forward(self, x, label):
        label_linear_out1 = self.act(self.label_linear1(label))
        linear1_out = self.act(self.linear1(torch.cat([x, label_linear_out1], dim=1)))

        label_linear_out2 = self.act(self.label_linear2(label))
        linear2_out = self.act(self.linear2(torch.cat([linear1_out, label_linear_out2], dim=1)))

        label_linear_out3 = self.act(self.label_linear3(label))
        linear3_out = self.act(self.linear3(torch.cat([linear2_out, label_linear_out3], dim=1)))

        return self.linear4(linear3_out)


class CelebAEnergy(nn.Module):

    def __init__(self, size_z1=256, size_z2=30):
        super().__init__()
        self.size_z1 = size_z1
        self.size_z2 = size_z2
        self.size_z = self.size_z1 + self.size_z2

        self.layers = nn.Sequential(
            nn.Linear(self.size_z, self.size_z),
            nn.Softplus(),
            nn.Linear(self.size_z, self.size_z*2),
            nn.Softplus(),
            nn.Linear(self.size_z*2, self.size_z*2),
            nn.Softplus(),
            nn.Linear(self.size_z*2, self.size_z),
            nn.Softplus(),
            nn.Linear(self.size_z, self.size_z),
            nn.Softplus(),
            nn.Linear(self.size_z, 1),)

    def forward(self, x):
        return self.layers(x)