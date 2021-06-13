import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from utils import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F


# Weight_init_D
def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class PatchEmbed(nn.Module):

    def __init__(self, patch_size=1, in_chans=3, embed_dim=64):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            # nn.LeakyReLU(0.1,inplace=True)
        )

    def forward(self, x):
        x = self.embed(x).flatten(2).transpose(1, 2)
        return x


class UnPatchEmbed(nn.Module):

    def __init__(self, patch_size=1, out_chans=3, embed_dim=64):
        super().__init__()
        self.unbed = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, out_chans, kernel_size=patch_size, stride=patch_size),
            nn.Tanh()
        )

    def forward(self, x):
        H = int(x.shape[1] ** 0.5)
        B = x.shape[0]
        C = x.shape[2]
        x = self.unbed(x.transpose(1, 2).view(B, C, H, H))
        return x  # B*C*H*W


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Mlp_T(nn.Module):
    def __init__(self, dim, mlp_factor, bias_linear = False,dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim*mlp_factor, 1, bias=bias_linear),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim*mlp_factor,dim, 1, bias=bias_linear),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
        
class TokenMix(nn.Module):
    def __init__(self, in_channels, in_width, in_height, token_facter, dropout = 0.):
        super().__init__()
        self.height = in_height
        self.width = in_width
        self.channels = in_channels
        self.norm = nn.LayerNorm(in_channels)
        self.mlp1 = Mlp_T(in_height, token_facter, bias_linear = False,dropout = 0.)
        self.mlp2 = Mlp_T(in_width, token_facter, bias_linear=False, dropout=0.)
        self.mlp3 = Mlp_T(in_height, token_facter, bias_linear=False, dropout=0.)
        self.mlp4 = Mlp_T(in_width, token_facter, bias_linear=False, dropout=0.)

    def forward(self,x):
        b = x.shape[0]
        x = x.permute(0,3,2,1) #b c h w -> b w h c
        x = self.norm(x)
        x = x.view(b*self.width,self.height,self.channels)#b w h c -> (b w) h c
        x = self.mlp1(x)
        x = x.view(b,self.width,self.height,self.channels).transpose(1,2).flatten(0,1)#(b w) h c -> (b h) w c
        x = self.mlp2(x)
        x = x.view(b, self.height, self.width, self.channels).transpose(1, 2).flatten(0,1)#(b h) w c -> (b w) h c
        x = self.mlp3(x)
        x = x.view(b, self.width, self.height, self.channels).transpose(1, 2).flatten(0,1)#(b w) h c -> (b h) w c
        x = self.mlp4(x)
        x = x.view(b, self.height, self.width, self.channels).permute(0,3,1,2)#(b h) w c -> b c h w
        return x

def para_partition(x, len_para):
    B,nh, N, C_ = x.shape
    x = x.view(B,nh, N//len_para, len_para,  C_)
    return x


def para_reverse(paras):
    B, nh, np, len_para, C_ = paras.shape
    x = paras.permute(0,2,3,1,4).contiguous().view(B,np*len_para,nh*C_)
    return x


class ParaAttention(nn.Module):

    def __init__(self, dim, len_para, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,res=None,mix=True):

        super().__init__()
        self.dim = dim
        self.mix = mix
        self.len_para = len_para  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.res = res
        self.qkv = nn.Conv2d(dim, dim * 3, 3,padding=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim , 3,padding=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.token_mix = TokenMix(dim, res[0], res[1], token_facter=2, dropout = 0.)

    def attention(self, x, transpose=False):
        B, C, H,W = x.shape
        N = H*W
        if transpose:
            x = x.transpose(-2,-1)
        qkv = self.qkv(x).permute(0,2,3,1).view(B,N,-1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale #B,nh,N,C//nh
        q_paras = para_partition(q,self.len_para) #B,nh,np,len_para,C//nh
        k_paras = para_partition(k,self.len_para) #B,nh,np,len_para,C//nh
        v_paras = para_partition(v,self.len_para) #B,nh,np,len_para,C//nh
        attn = (q_paras @ k_paras.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x_paras = (attn@v_paras)
        x = para_reverse(x_paras).view(B,H,W,C).permute(0,3,1,2)
        return x

    def forward(self,x):
        B,N,C = x.shape
        H,W=self.res
        x = x.contiguous().view(B,H,W,C).permute(0,3,1,2)#b n c -> b h w c -> b c h w
        if self.mix:
          x = self.token_mix(x)
        x = self.attention(x,transpose=True)
        x = self.attention(x,transpose=True)
        x = self.proj(x).contiguous().view(B,C,N).permute(0,2,1)
        return x


class ParaTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, len_para=7, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU,norm_layer=nn.LayerNorm,attn_mix=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.len_para = len_para
        self.mlp_ratio = mlp_ratio
        self.attn = ParaAttention(dim, len_para=self.len_para, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,res=input_resolution,mix=attn_mix)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm1=norm_layer(dim)
        self.norm2=norm_layer(dim)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class Generator(nn.Module):
    def __init__(self, img_size=256, patch_size=1, in_chans=3, embed_dim=64, num_heads=8, depth=6,
                 length_para=2, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0,
                 norm_layer=nn.LayerNorm,glob_layers=3,loc_layers=7):
        super().__init__()


        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.input_resolution = (img_size, img_size)
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(patch_size=1, in_chans=3, embed_dim=64)
        self.patch_unbed = UnPatchEmbed(patch_size=1, out_chans=3, embed_dim=64)

        # keep
        self.glob = nn.ModuleList([
                    ParaTransformerBlock(embed_dim, self.input_resolution, num_heads, len_para=length_para, mlp_ratio=4.,
                                         qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                                         act_layer=nn.GELU,norm_layer=nn.LayerNorm,attn_mix=True)
                    for i in range(glob_layers)
        ])
        self.loc = nn.ModuleList([
                    ParaTransformerBlock(embed_dim, self.input_resolution, num_heads, len_para=length_para, mlp_ratio=4.,
                                         qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                                         act_layer=nn.GELU,norm_layer=nn.LayerNorm,attn_mix=False)
                    for i in range(loc_layers)
        ])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        x = self.patch_embed(x)
        for blk in self.glob:
          x = blk(x)
        for blk_ in self.loc:
          x = blk_(x)
        x = self.patch_unbed(x)
        return x


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=True),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)  # 1*1*16*16 tensor
        return self.model(img_input)




