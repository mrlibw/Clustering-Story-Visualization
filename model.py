import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.nn.utils import spectral_norm
from torchvision.models.video.resnet import r2plus1d_18
from miscc.config import cfg
from torch.autograd import Variable
import numpy as np
import pdb
if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from Attention import SpatialAttention as SPATIAL_NET
from torchvision import models
import torch.utils.model_zoo as model_zoo

def conv3x3(in_planes, out_planes, stride=1, use_spectral_norm=False):
    "3x3 convolution with padding"
    if use_spectral_norm:
        return spectral_norm(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False))
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)

def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block


class CA_NET(nn.Module):

    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.DIMENSION * cfg.VIDEO_LEN
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.relu = nn.ReLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar



class CNN_ENCODER(nn.Module):
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        if cfg.TRAIN.FLAG:
            self.nef = nef
        else:
            self.nef = 256 

        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)

        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        features = x

        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.avg_pool2d(x, kernel_size=8)
        x = x.view(x.size(0), -1)

        cnn_code = self.emb_cnn_code(x)
        if features is not None:
            features = self.emb_features(features)
        return features, cnn_code

# LSTM text encoder
class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = cfg.TEXT.WORDS_NUM
        self.ntoken = ntoken  
        self.ninput = ninput  
        self.drop_prob = drop_prob 
        self.nlayers = nlayers 
        self.bidirectional = bidirectional
        self.rnn_type = cfg.RNN_TYPE
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                       bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden):
        emb = self.drop(self.encoder(captions))
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        output, hidden = self.rnn(emb, hidden)
        output = pad_packed_sequence(output, batch_first=True)[0]
        words_emb = output.transpose(1, 2)
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb

class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=True):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if bcondition:
            self.outlogits = nn.Sequential(
                conv3x3(ndf * 8 + nef, ndf * 8, use_spectral_norm=True),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4)),
                nn.Sigmoid())
        else:
            self.outlogits = nn.Sequential(
                spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4)),
                nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            h_c_code = torch.cat((h_code, c_code), 1)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)

class R2Plus1dStem(nn.Sequential):
    def __init__(self):
        super(R2Plus1dStem, self).__init__(
            spectral_norm(nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False)),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv3d(45, 64, kernel_size=(1, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

class BasicBlock(nn.Module):

    __constants__ = ['downsample']
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class VideoEncoder(nn.Module):
    def __init__(self):
        super(VideoEncoder, self).__init__()
        video_resnet = r2plus1d_18(pretrained=False, progress=True)
        padding= 1
        block = [
            R2Plus1dStem(),
            spectral_norm(nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, padding, padding)
                ,bias=False)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(128, 128, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(padding, 0, 0),
                            bias=False)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(128, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, padding, padding), 
                bias=False)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(128, 256, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(padding, 0, 0),
                bias=False)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(256, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, padding, padding), 
                bias=False)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(256, 512, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(padding, 0, 0),
                bias=False)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(512, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, padding, padding), 
                bias=False)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(512, 512, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(padding, 0, 0), 
                bias=False)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2),
        ]
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.story_encoder = nn.Sequential(*block)
        self.detector = nn.Sequential(
            spectral_norm(nn.Linear(512, 128)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            spectral_norm(nn.Linear(128, 1)),
        )

    def forward(self, story):

        B = story.shape[0]
        latents = self.story_encoder(story)
        latents = self.pool(latents)
        latents = latents.view(B, -1)
        return self.detector(latents)
class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out

def Block3x3(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU())
    return block

class ATTN_STAGE(nn.Module):
    def __init__(self, ngf, nef):
        super(ATTN_STAGE, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.num_residual = cfg.GAN.R_NUM
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        self.att = SPATIAL_NET(self.gf_dim, self.ef_dim)

        self.residual = self._make_layer(ResBlock, self.gf_dim * 2)
        self.sample = Block3x3(self.gf_dim * 2, self.gf_dim)

    def forward(self, h_code, word_embs):

        c_code, att = self.att(h_code, word_embs)        
        c_code = c_code.view(word_embs.size(0), -1, h_code.size(2), h_code.size(3))
        h_c_code = torch.cat((h_code, c_code), 1)
        out_code = self.residual(h_c_code)
        out_code = self.sample(out_code)

        return out_code
 
class StoryGAN(nn.Module):
    def __init__(self, video_len):
        super(StoryGAN, self).__init__()
        self.batch_size = cfg.TRAIN.IM_BATCH_SIZE
        self.gf_dim = cfg.GAN.GF_DIM * 8 
        self.motion_dim = cfg.TEXT.DIMENSION + cfg.LABEL_NUM
        self.content_dim = cfg.GAN.CONDITION_DIM 
        self.noise_dim = cfg.GAN.Z_DIM  
        self.recurrent = nn.GRUCell(self.noise_dim + self.motion_dim, self.motion_dim) 
        self.mocornn = nn.GRUCell(self.motion_dim, self.content_dim) 
        self.video_len = video_len
        self.n_channels = 3
        self.filter_num = 3
        self.filter_size = 21
        self.image_size = 124
        self.out_num = 1

        self.aux_size = 5
        self.fix_input = 0.1*torch.tensor(range(self.aux_size)).float().cuda()
        self.define_module()

    def define_module(self):
        from layers import DynamicFilterLayer1D as DynamicFilterLayer
        ninput = self.motion_dim + self.content_dim + self.image_size 
        ngf = self.gf_dim 
        
        self.ca_net = CA_NET()
        
        self.filter_net = nn.Sequential(
            nn.Linear(self.content_dim, self.filter_size * self.filter_num * self.out_num),
            nn.BatchNorm1d(self.filter_size * self.filter_num * self.out_num))

        self.image_net = nn.Sequential(
            nn.Linear(self.motion_dim, self.image_size * self.filter_num),
            nn.BatchNorm1d(self.image_size * self.filter_num),
            nn.Tanh())
        
        self.fc = nn.Sequential(
            nn.Linear(ninput, ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4),
            nn.ReLU(True))
        self.upsample1 = upBlock(ngf, ngf//2)
        self.upsample2 = upBlock(ngf//2, ngf//4) 
        self.upsample3 = upBlock(ngf//4, ngf//8)
        self.upsample4 = upBlock(ngf//8, ngf//16)
        self.img = nn.Sequential(
            conv3x3(ngf // 16, 3),
            nn.Tanh())
    
        self.m_net = nn.Sequential(
            nn.Linear(self.motion_dim, self.motion_dim),
            nn.BatchNorm1d(self.motion_dim))

        self.c_net = nn.Sequential(
            nn.Linear(self.content_dim, self.content_dim),
            nn.BatchNorm1d(self.content_dim))

        self.dfn_layer = DynamicFilterLayer(self.filter_size, 
            pad = self.filter_size//2)

        self.attn_1 = ATTN_STAGE(ngf//8, cfg.TEXT.DIMENSION)
        self.attn_2 = ATTN_STAGE(ngf//16, cfg.TEXT.DIMENSION)

    def get_iteration_input(self, motion_input):
        num_samples = motion_input.shape[0]
        noise = T.FloatTensor(num_samples, self.noise_dim).normal_(0,1)
        return torch.cat((noise, motion_input), dim = 1)

    def get_gru_initial_state(self, num_samples):
        return Variable(T.FloatTensor(num_samples, self.motion_dim).normal_(0, 1))

    def sample_z_motion(self, motion_input, video_len=None):
        video_len = video_len if video_len is not None else self.video_len
        num_samples = motion_input.shape[0]
        h_t = [self.m_net(self.get_gru_initial_state(num_samples))]
        
        for frame_num in range(video_len):
            if len(motion_input.shape) == 2:
                e_t = self.get_iteration_input(motion_input)
            else:
                e_t = self.get_iteration_input(motion_input[:,frame_num,:])
            h_t.append(self.recurrent(e_t, h_t[-1]))
        z_m_t = [h_k.view(-1, 1, self.motion_dim) for h_k in h_t]
        z_motion = torch.cat(z_m_t[1:], dim=1).view(-1, self.motion_dim)
        return z_motion

    def motion_content_rnn(self, motion_input, content_input):
        video_len = 1 if len(motion_input.shape) == 2 else self.video_len
        h_t = [self.c_net(content_input)]
        if len(motion_input.shape) == 2:
            motion_input = motion_input.unsqueeze(1)
        for frame_num in range(video_len):
            h_t.append(self.mocornn(motion_input[:,frame_num, :], h_t[-1]))
        
        c_m_t = [h_k.view(-1, 1, self.content_dim) for h_k in h_t]
        mocornn_co = torch.cat(c_m_t[1:], dim=1).view(-1, self.content_dim)
        return mocornn_co

    def sample_videos(self, motion_input, content_input, words_embs):  

        bs, video_len  = motion_input.shape[0], motion_input.shape[1]
        num_img = bs * video_len
        content_input = content_input.view(-1, cfg.VIDEO_LEN * content_input.shape[2])
        if content_input.shape[0] > 1:
            content_input = torch.squeeze(content_input)
        r_code, r_mu, r_logvar = self.ca_net(content_input)
        c_mu = r_mu.repeat(self.video_len, 1).view(-1, r_mu.shape[1])

        crnn_code = self.motion_content_rnn(motion_input, r_code) 
        temp = motion_input.view(-1, motion_input.shape[2])
        m_code, m_mu, m_logvar = temp, temp, temp 
        m_code = m_code.view(motion_input.shape[0], self.video_len, self.motion_dim)
        zm_code = self.sample_z_motion(m_code, self.video_len) 

        zmc_code = torch.cat((zm_code, c_mu), dim = 1)
        m_image = self.image_net(m_code.view(-1, m_code.shape[2])) 
        m_image = m_image.view(-1, self.filter_num, self.image_size)
        c_filter = self.filter_net(crnn_code) 
        c_filter = c_filter.view(-1, self.out_num, self.filter_num, self.filter_size)
        mc_image = self.dfn_layer([m_image, c_filter]) 
        zmc_all_ = torch.cat((zmc_code, mc_image.squeeze(1)), dim = 1)
        zmc_img = self.fc(zmc_all_).view(-1, self.gf_dim, 4, 4)

        h_code = self.upsample1(zmc_img) 
        h_code = self.upsample2(h_code)  
        h_code = self.upsample3(h_code)  
        h_code = self.attn_1(h_code, words_embs)
        h_code = self.upsample4(h_code) 
        h_code = self.attn_2(h_code, words_embs)

        h = self.img(h_code)
        fake_video = h.view( int(h.size(0)/self.video_len), self.video_len, self.n_channels, h.size(3), h.size(3)) 
        fake_video = fake_video.permute(0, 2, 1, 3, 4)
        return fake_video, m_mu, m_logvar, r_mu, r_logvar 

class DIS_STY(nn.Module):
    def __init__(self):
        super(DIS_STY, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.text_dim = cfg.TEXT.DIMENSION
        self.label_num = cfg.LABEL_NUM
        self.define_module()
    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            spectral_norm(nn.Conv2d(40, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.seq_consisten_model = None
        
        self.get_uncond_logits = D_GET_LOGITS(ndf, nef + self.text_dim + self.label_num, bcondition=False)
        self.cate_classify = None
        self.convert = conv3x3(3, 356)
    def forward(self, story, new_word_emb):
        N, C, video_len, W, H = story.shape

        story = story.permute(0,2,1,3,4)
        story = story.contiguous().view(-1, C,W,H)

        new_story = self.convert(story)
        new_story = new_story.view(new_story.size(0), new_story.size(1), -1)
        new_word_emb = torch.transpose(new_word_emb, 1, 2)
        combine = torch.bmm(new_word_emb, new_story)
        combine = combine.view(combine.size(0), combine.size(1), 64, -1)
    
        
        story_embedding = torch.squeeze(self.encode_img(combine))
        return story_embedding

if __name__ == "__main__":
    img = torch.randn(3,3,5,64, 64)
    m = VideoEncoder()
    m(img)
