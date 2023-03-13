
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)


def func_attention(query, context, gamma1):

    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    attn = torch.bmm(contextT, query) 
    attn = attn.view(batch_size*sourceL, queryL)
    attn = nn.Softmax()(attn)  

    attn = attn.view(batch_size, sourceL, queryL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size*queryL, sourceL)
    attn = attn * gamma1
    attn = nn.Softmax()(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)


class SpatialAttention(nn.Module):
    def __init__(self, idf, cdf):
        super(SpatialAttention, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.tanh = nn.Tanh()
        self.conv_img = conv1x1(idf, idf)
        self.conv_text = conv1x1(cdf, idf)

    def forward(self, input, context):

        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        sourceT = context.unsqueeze(3)
        text = F.tanh(self.conv_text(sourceT).squeeze(3))
        sourceT = self.conv_context(sourceT).squeeze(3)

        
        img = F.tanh(self.conv_img(input))
        img = img.view(batch_size, -1, queryL)
        img = torch.transpose(img, 1, 2).contiguous()

        attn = torch.bmm(targetT, sourceT)
        combine = torch.bmm(img, text)
        combine = self.tanh(combine)
        combine = torch.transpose(combine, 1, 2).contiguous()

        attn = attn.view(batch_size*queryL, sourceL)
        attn = F.softmax(attn, dim = 1)

        attn = attn.view(batch_size, queryL, sourceL)
        attn = torch.transpose(attn, 1, 2).contiguous()
        attn = attn.view(batch_size*sourceL, queryL)
        attn = F.softmax(attn, dim = 1)
        attn = attn.view(batch_size, sourceL, queryL)

        attn = attn * combine
        weightedContext = torch.bmm(sourceT, attn)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn



