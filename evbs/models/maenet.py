"""MAENet (Sun et al., "Motion Aware Event Representation-driven Image Deblurring", ECCV 2024).

Network definition from https://github.com/ZhijingS/DA_event_deblur (MIT License, Copyright (c) 2024
ZhijingS), reduced to the modules used by Baseline_Biattention_CrossRecurrent_FC.
"""
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Mutual_AttentionwithFC(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Mutual_AttentionwithFC, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.scale = nn.Conv2d(dim,dim,kernel_size=1,bias=bias)
        self.shift = nn.Conv2d(dim,dim,kernel_size=1,bias=bias)


    def forward(self, x, y):

        assert x.shape == y.shape, 'The shape of feature maps from image and event branch are not equal!'

        b,c,h,w = x.shape

        scale = F.leaky_relu(self.scale(x))
        shift = F.leaky_relu(self.shift(x))
        y_ = y * (scale + 1) + shift

        q = self.q(x) # image
        k = self.k(y_) # event
        v = self.v(y_) # event
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out


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


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class Sparsemask(nn.Module):
    def __init__(self, k=20.0):
        super(Sparsemask, self).__init__()
        self.k_tensor = nn.Parameter(torch.tensor(k))

    def forward(self, x):
        topk_values, topk_indices = torch.topk(x, k=int(self.k_tensor), dim=-1)
        bottomk_values, bottomk_indices = torch.topk(x, k=int(self.k_tensor), dim=-1, largest=False)

        mask = torch.ones_like(x)
        mask.scatter_(-1, topk_indices, 0)
        mask.scatter_(-1, bottomk_indices, 0)
        x = x.masked_fill(mask == 1,0)
        return x


class EventImage_BiAttentionTransformerBlockwithFC(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(EventImage_BiAttentionTransformerBlockwithFC, self).__init__()

        self.norm1_image = LayerNorm(dim, LayerNorm_type)
        self.norm1_event = LayerNorm(dim, LayerNorm_type)
        self.attn = Mutual_AttentionwithFC(dim, num_heads, bias)
        # mlp
        self.norm2 = nn.LayerNorm(dim*2)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim*2, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        self.conv = nn.Conv2d(in_channels=2*dim,out_channels=dim,kernel_size=1,stride=1)

    def forward(self, image, event):
        # image: b, c, h, w
        # event: b, c, h, w
        # return: b, 2c, h, w
        assert image.shape == event.shape, 'the shape of image doesnt equal to event'
        b, c, h, w = image.shape

        # out_i,im_ = self.attn(self.norm1_image(image), self.norm1_event(event))
        # fused_image = image + out_i
        # out_e,e_ = self.attn(self.norm1_image(event), self.norm1_event(image))
        # fused_event = event + out_e
        fused_image = image + self.attn(self.norm1_image(image), self.norm1_event(event)) # b, c, h, w b 768 h/16 w/16
        fused_event = event + self.attn(self.norm1_image(event), self.norm1_event(image))
        # mlp
        fused = torch.cat([fused_image,fused_event],dim=1)
        fused = to_3d(fused) # b, h*w, 2c
        fused = fused + self.ffn(self.norm2(fused))
        fused = to_4d(fused, h, w) # b,2c,h,w
        fused = self.conv(fused)

        return fused


class BaselineBlock(nn.Module):
    def __init__(self, c, DW_Expand=1, FFN_Expand=2, drop_out_rate=0., num_heads=None):
        super().__init__()
        dw_channel = c * DW_Expand
        self.num_heads = num_heads
        
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Channel Attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )

        # GELU
        self.gelu = nn.GELU()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        
        if self.num_heads is not None:
            self.image_event_transformer = EventImage_BiAttentionTransformerBlockwithFC(c, num_heads=self.num_heads,
                                                                                       ffn_expansion_factor=4, bias=False,
                                                                                       LayerNorm_type='WithBias')

    def forward(self, inp, event_filter=None):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = x * self.se(x)
        x = self.conv3(x)
        if self.num_heads:
            assert event_filter != None
            x = self.image_event_transformer(x, event_filter) # x=b,c,h,w

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.gelu(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class BaselineBlock_seq(nn.Module):
    def __init__(self, c, DW_Expand=1, FFN_Expand=2, drop_out_rate=0., num_heads=None,n_block=2):
        super().__init__()
        self.fusion = BaselineBlock(c, DW_Expand=DW_Expand, FFN_Expand=2, drop_out_rate=drop_out_rate, num_heads=num_heads)
        self.seq_block = nn.Sequential(
            *[BaselineBlock(c, DW_Expand=DW_Expand, FFN_Expand=2, drop_out_rate=drop_out_rate, num_heads=None) for _ in range(n_block-1)]
        )
    
    def forward(self, inp, event_filter=None):
        x = inp
        
        x = self.fusion(x, event_filter)
        x = self.seq_block(x)
        
        return x


class RecurrentBaseBlock(nn.Module):
    def __init__(self,chan,num,dw_expand,ffn_expand):
        super(RecurrentBaseBlock,self).__init__()
        self.nowencoder = nn.Sequential(
                    *[BaselineBlock(chan,dw_expand,ffn_expand) for _ in range(num)]
        )
        self.lastencoder = BaselineBlock(chan,dw_expand,ffn_expand)
        self.alpha = nn.Parameter(torch.zeros((1, chan, 1, 1)), requires_grad=True)
    def forward(self,eventlist):
        statels = []
        for i in range(len(eventlist)):
            now = self.nowencoder(eventlist[i])
            if i == 0:
                # temp = torch.zeros_like(now)
                # last = self.lastencoder(temp)
                new = now 
                statels.append(new)
                continue
            last = self.lastencoder(statels[i-1])
            new = now * (1-self.alpha)  + last * self.alpha
            statels.append(new)
        return statels


class MakeRecurrentList(nn.Module):
    def __init__(self):
        super(MakeRecurrentList,self).__init__()
    def forward(self,event):
        B ,C ,H, W = event.shape
        chunks = event.chunk(C,dim=1)
        listlen = int(C/2)
        eventlist = []
        for i in range(listlen):
            temp_y = torch.cat([chunks[i],chunks[-1-i]],dim=1)
            eventlist.append(temp_y)
        return eventlist


class Baseline_Biattention_CrossRecurrent_FC(nn.Module):
    def __init__(self, in_chn=3, ev_chn=7, width=16,
                middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],
                dw_expand=1, ffn_expand=2,
                num_heads=[1,2,4]):
        super().__init__()
        
        self.num_heads = num_heads
        
        self.intro = nn.Conv2d(in_channels=in_chn, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.makelist = MakeRecurrentList()
        self.ev_intro = nn.Conv2d(in_channels=2, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=in_chn, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.cross = nn.Conv2d(in_channels=in_chn+2,out_channels=in_chn,kernel_size=3, padding=1, stride=1, groups=1,bias=True)
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        self.ev_encoders = nn.ModuleList()
        self.ev_downs = nn.ModuleList()
        
        
        chan = width
        for i, num in enumerate(enc_blk_nums):
            self.encoders.append(
                BaselineBlock_seq(chan, dw_expand, ffn_expand,num_heads=self.num_heads[i],n_block=num)
            )
            self.ev_encoders.append(
                RecurrentBaseBlock(chan,num,dw_expand,ffn_expand)
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            self.ev_downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(middle_blk_num)]
            )
            
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(num)]
                )
            )
    
    def forward(self, x, event=None, mask=None, gt=None):
        x_ = self.intro(x)
        eventlist = self.makelist(event)
        e1 = []

        for i in range(len(eventlist)):
            ev_x = self.ev_intro(eventlist[i])
            e1.append(ev_x)
        
        
        ev = []
        #event encoder
        for encoder, down in zip(self.ev_encoders, self.ev_downs):
            e1 = encoder(e1)
            ev.append(e1[-1])
            for i in range(len(e1)):
                e1[i] = down(e1[i])
        
        encs = []
        #image encoder
        for encoder, down, ev_feature in zip(self.encoders, self.downs, ev):
            x_ = encoder(x_, ev_feature)
            encs.append(x_)
            x_ = down(x_)
        
        x_ = self.middle_blks(x_)
        
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x_ = up(x_)
            x_ = x_ + enc_skip
            x_ = decoder(x_)
        
        x_ = self.ending(x_)
        x_ = torch.cat([x_,eventlist[-1]],dim=1)
        x_ = self.cross(x_)

        x_ = x_ + x
        
        return x_
