from __future__ import print_function
from six.moves import range
from PIL import Image

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time
import pdb
import numpy as np
import torchfile

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import weights_init, count_param
from miscc.utils import save_story_results, save_model
from miscc.utils import KL_loss
from miscc.utils import compute_discriminator_loss, compute_generator_loss
from shutil import copyfile

from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
class GANTrainer(object):
    def __init__(self, output_dir, args, ratio=1.0):
        if cfg.TRAIN.FLAG:
            output_dir = "{}/".format(output_dir)
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'log')
            self.test_dir = os.path.join(output_dir, 'Test')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            mkdir_p(self.test_dir)
            if not os.path.exists(os.path.join(self.model_dir, 'model.py')):
                copyfile(args.cfg_file, output_dir + 'setting.yml')
                copyfile('./model.py', output_dir + 'model.py')
                copyfile('./trainer.py', output_dir + 'trainer.py')

        self.video_len = cfg.VIDEO_LEN
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.imbatch_size = cfg.TRAIN.IM_BATCH_SIZE * self.num_gpus
        self.stbatch_size = cfg.TRAIN.ST_BATCH_SIZE * self.num_gpus
        self.ratio = ratio
        self.con_ckpt = args.continue_ckpt
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

    # ############# For training stageI GAN #############
    def load_network_stageI(self, n_words):
        from model import StoryGAN, DIS_STY, RNN_ENCODER, CNN_ENCODER
        netG = StoryGAN(self.video_len)
        netG.apply(weights_init)

        netD_st = DIS_STY()
        netD_st.apply(weights_init)

        netG_param_cnt, netD_st_param = count_param(netG), count_param(netD_st)
        total_params = netG_param_cnt + netD_st_param
        
        print('The total parameter is : {}M, netG:{}M, netD_st:{}M'.format(total_params//1e6, netG_param_cnt//1e6,
            netD_st_param//1e6))

        if cfg.NET_G != '':
            state_dict = \
                torch.load(cfg.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_G)
        if cfg.NET_D != '':
            state_dict = \
                torch.load(cfg.NET_D,
                           map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_D)
        if self.con_ckpt:
            print('Continue training from epoch {}'.format(self.con_ckpt))
            path = '{}/netG_epoch_{}.pth'.format(self.model_dir, self.con_ckpt)
            netG.load_state_dict(torch.load(path))
            path = '{}/netD_st_epoch_last.pth'.format(self.model_dir)
            netD_st.load_state_dict(torch.load(path))
            

        # load the pretrained text encoder
        text_encoder = RNN_ENCODER(n_words, nhidden=cfg.TEXT.DIMENSION)
        if cfg.TRAIN.TEXT_ENCODER != '':
            path = cfg.TRAIN.TEXT_ENCODER 
            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from: ', path)
        text_encoder.eval()


        # load the pretrained image encoder
        image_encoder = CNN_ENCODER(cfg.TEXT.DIMENSION)
        if cfg.TRAIN.TEXT_ENCODER != '':
            img_encoder_path = cfg.TRAIN.TEXT_ENCODER.replace('text_encoder', 'image_encoder')
            state_dict = \
                    torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
            image_encoder.load_state_dict(state_dict)
            print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()


        if cfg.CUDA:
            netG.cuda()
            netD_st.cuda()
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            

        return netG, netD_st, text_encoder, image_encoder


    def sample_real_image_batch(self):
        if self.imagedataset is None:
            self.imagedataset = enumerate(self.imageloader)
        batch_idx, batch = next(self.imagedataset)
        b = batch
        if cfg.CUDA:
            for k, v in batch.items():
                if k == 'text' or k == 'full_text':
                    continue
                else:
                    b[k] = v.cuda()

        if batch_idx == len(self.imageloader) - 1:
            self.imagedataset = enumerate(self.imageloader)
        return b

    def train(self, imageloader, storyloader, testloader, storydataset, stage=1):
        c_time = time.time()
        self.imageloader = imageloader
        self.imagedataset = None


        captions, ixtoword, wordtoix, n_words = storydataset.return_info()

        netG, netD_st, text_encoder, image_encoder = self.load_network_stageI(n_words)

        non_word_emb = nn.Embedding(1, cfg.TEXT.DIMENSION)
        idx = torch.LongTensor([0])
        non_word_emb = non_word_emb(idx).unsqueeze(2).cuda()
        
        start = time.time()
        st_real_labels = Variable(torch.FloatTensor(self.imbatch_size).fill_(1))
        st_fake_labels = Variable(torch.FloatTensor(self.imbatch_size).fill_(0))
        match_labels = F.one_hot(torch.arange(self.imbatch_size), self.imbatch_size)
        if cfg.CUDA:
            st_real_labels, st_fake_labels = st_real_labels.cuda(), st_fake_labels.cuda()
            match_labels = match_labels.cuda()


        image_weight = cfg.IMAGE_RATIO

        generator_lr = cfg.TRAIN.GENERATOR_LR
        discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
        lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH

        st_optimizerD = optim.Adam(netD_st.parameters(), lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.Adam(netG_para, lr=cfg.TRAIN.GENERATOR_LR, betas=(0.5, 0.999))

        mse_loss = nn.MSELoss()

        scheduler_stD = ReduceLROnPlateau(st_optimizerD, 'min', verbose=True, factor=0.5, min_lr=1e-7, patience=0)
        scheduler_G = ReduceLROnPlateau(optimizerG, 'min', verbose=True, factor=0.5, min_lr=1e-7, patience=0)
        count = 0

        if not self.con_ckpt:
            start_epoch = 0
        else:
            start_epoch = int(self.con_ckpt)

        print('LR DECAY EPOCH: {}'.format(lr_decay_step))
        batch_cnt = 1
        batch_container_img = []
        batch_container_fake_img = []
        batch_container_text = []
        batch_texts = None
        batch_imgs = None
        batch_fake_imgs = None

        for epoch in range(start_epoch, self.max_epoch):
            l = self.ratio * (2. / (1. + np.exp(-10. * epoch)) - 1)
            start_t = time.time()

            num_step = len(storyloader)
            stats = {}

            with tqdm(total=len(storyloader), dynamic_ncols=True) as pbar:
                for i, data in enumerate(storyloader):
                    ######################################################
                    # (1) Prepare training data
                    ######################################################
                    st_batch = data
                    st_real_cpu = st_batch['images']
                    st_motion_input = st_batch['description'][:, :, :cfg.TEXT.DIMENSION] 
                    st_content_input = st_batch['description'][:, :, :cfg.TEXT.DIMENSION]


                    ##############################
                    # convert text descriptions into sentence embedding and word embeddings for story generation part
                    st_texts = None
                    if 'text' in st_batch:
                        st_texts = st_batch['text']

                    st_list = []
                    for idx in range(cfg.TRAIN.ST_BATCH_SIZE):
                        for j in range(cfg.VIDEO_LEN):
                            cur = st_texts[j]
                            st_list.append(cur[idx])

                    captions = []
                    cap_lens = []


                    for cur_text in st_list:
                        if len(cur_text) == 0:
                            continue
                        cur_text = cur_text.replace("\ufffd\ufffd", " ")
                        tokenizer = RegexpTokenizer(r'\w+')
                        tokens = tokenizer.tokenize(cur_text.lower())
                        if len(tokens) == 0:
                            print('cur_text', cur_text)
                            continue

                        rev = []
                        for t in tokens:
                            t = t.encode('ascii', 'ignore').decode('ascii')
                            if len(t) > 0 and t in wordtoix:
                                rev.append(wordtoix[t])
                        captions.append(rev)
                        cap_lens.append(len(rev))

                    max_len = np.max(cap_lens)

                    sorted_indices = np.argsort(cap_lens)[::-1]
                    cap_lens = np.asarray(cap_lens)
                    
                    org_st_cap_lens = cap_lens


                    cap_lens = cap_lens[sorted_indices]
                    cap_array = np.zeros((len(captions), max_len), dtype='int64')
                    for pointer in range(len(captions)):
                        idx_cap = sorted_indices[pointer]
                        cap = captions[idx_cap]
                        c_len = len(cap)
                        cap_array[pointer, :c_len] = cap
  

                    st_captions = cap_array
                    batch_size = st_captions.shape[0]
                    st_captions = Variable(torch.from_numpy(st_captions), volatile=True)
                    st_cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)
                    org_st_cap_lens = Variable(torch.from_numpy(org_st_cap_lens), volatile=True)
                    

                    st_captions = st_captions.cuda()
                    st_cap_lens = st_cap_lens.cuda()
                    org_st_cap_lens = org_st_cap_lens.cuda()

                    batch_size = st_captions.size(0)
                    hidden = text_encoder.init_hidden(batch_size)
                    st_words_embs, st_sent_emb = text_encoder(st_captions, st_cap_lens, hidden)
                    st_words_embs = st_words_embs.detach()
                    st_sent_emb =st_sent_emb.detach()
                    invert_st_sent_emb = Variable(torch.FloatTensor(st_sent_emb.size(0), st_sent_emb.size(1)).fill_(0)).cuda()
                    invert_st_words_embs = Variable(torch.FloatTensor(st_words_embs.size(0), st_words_embs.size(1), st_words_embs.size(2)).fill_(0)).cuda()

                    for pointer in range(len(st_sent_emb)):
                        idx_cap = sorted_indices[pointer]
                        cur_sent = st_sent_emb[pointer, :]
                        cur_word = st_words_embs[pointer, :, :]
                        invert_st_sent_emb[idx_cap, :] = cur_sent
                        invert_st_words_embs[idx_cap, :, :] = cur_word
                    new_invert_st_sent_emb = invert_st_sent_emb.view(-1, cfg.VIDEO_LEN, invert_st_sent_emb.size(1))

                    st_real_imgs = Variable(st_real_cpu)
                    st_motion_input = Variable(st_motion_input)
                    st_content_input = Variable(st_content_input)
                    st_labels = Variable(st_batch['labels']) 

                    
                    if cfg.CUDA:
                        st_real_imgs = st_real_imgs.cuda() 
                        st_motion_input = st_motion_input.cuda()
                        st_content_input = st_content_input.cuda()
                        st_labels = st_labels.cuda()

                    st_motion_input = torch.cat((new_invert_st_sent_emb, st_labels), 2) 
                    st_content_input = new_invert_st_sent_emb


                    with torch.no_grad():
                        st_fake, m_mu, m_logvar, c_mu, c_logvar = \
                            netG.sample_videos(st_motion_input, st_content_input, invert_st_words_embs) 

                    characters_mu = (st_labels.mean(1)>0).type(torch.FloatTensor).cuda() 
                    st_mu = torch.cat((c_mu, st_motion_input[:,:, :cfg.TEXT.DIMENSION].mean(1).squeeze(), characters_mu), 1)
                    
                    netD_st.zero_grad()
                    
                    new_word_emb = torch.ones(invert_st_words_embs.size(0), invert_st_words_embs.size(1), 40).cuda()
                    new_word_emb = new_word_emb * non_word_emb

                    if invert_st_words_embs.size(2) <= 40:
                        new_word_emb[:,:,:invert_st_words_embs.size(2)] = invert_st_words_embs
                    else:
                        ix = list(np.arange(invert_st_words_embs.size(2)))  
                        np.random.shuffle(ix)
                        ix = ix[:40]
                        ix = np.sort(ix)
                        new_word_emb = invert_st_words_embs[:,:,ix]

                    

                    cluster_flag = 0
                    if batch_cnt % 10 == 0:
                        batch_cnt = 1
                        cluster_flag = 1
                        batch_texts = torch.cat(batch_container_text, 0)
                        batch_imgs = torch.cat(batch_container_img, 0)
                        batch_fake_imgs = torch.cat(batch_container_fake_img, 0)
                    else:
                        batch_container_text.append(new_word_emb)
                        batch_container_img.append(st_real_imgs.view(-1, st_real_imgs.size(1),st_real_imgs.size(3), st_real_imgs.size(4)))
                        batch_container_fake_img.append(st_fake.reshape(-1, st_fake.size(1), st_fake.size(3), st_fake.size(4)))
                        batch_cnt += 1

                    st_errD, st_errD_real, st_errD_fake, loss_real, loss_real_st, loss_cluster  = \
                        compute_discriminator_loss(netD_st, st_real_imgs, st_fake,
                                               st_real_labels, st_fake_labels, st_labels,
                                               st_mu, self.gpus, invert_st_words_embs, new_word_emb,
                                               invert_st_sent_emb, image_encoder, match_labels, org_st_cap_lens, batch_texts, batch_imgs, cluster_flag)

      
                    st_errD.backward(retain_graph=True)
                    st_optimizerD.step()

                    step = i+num_step*epoch

                    ############################
                    # (2) Update G network
                    ###########################
                    netG.zero_grad()

                    st_fake, m_mu, m_logvar, c_mu, c_logvar = netG.sample_videos(st_motion_input, st_content_input, invert_st_words_embs)
                    encoder_decoder_loss = 0
            
                    characters_mu = (st_labels.mean(1)>0).type(torch.FloatTensor).cuda()
                    st_mu = torch.cat((c_mu, st_motion_input[:,:, :cfg.TEXT.DIMENSION].mean(1).squeeze(), characters_mu), 1)
                    
                    st_errG, st_w_loss, st_s_loss, loss_fake, loss_fake_st, loss_cluster_gene  = compute_generator_loss(netD_st, st_fake, st_real_imgs,
                                                st_real_labels, st_labels, st_mu, self.gpus, image_encoder, 
                                                invert_st_words_embs, invert_st_sent_emb, org_st_cap_lens, match_labels, new_word_emb, batch_texts, batch_fake_imgs, cluster_flag)

                    if cluster_flag == 1:
                        cluster_flag = 0
                        batch_container_text = []
                        batch_container_img = []
                        batch_container_fake_img = []
                        batch_texts = None
                        batch_imgs = None
                        batch_fake_imgs = None


                    st_kl_loss = KL_loss(c_mu, c_logvar)
                    kl_loss = self.ratio * st_kl_loss 

                    errG_total = self.ratio * (st_errG*image_weight + st_kl_loss * cfg.TRAIN.COEFF.KL)
                    errG_total.backward(retain_graph=True)

                    optimizerG.step()
                    count = count + 1
                    pbar.update(1)


            print('''[%d/%d]
                    st_errD: %.2f st_errG: %.2f \\
                    st_w_loss: %.2f st_s_loss: %.2f fake_loss: %.2f real_loss: %.2f \\
                    loss_real_st: %.2f loss_fake_st: %.2f loss_cluster_dis: %.2f loss_cluster_gene: %.2f'''
                    % (epoch, self.max_epoch, st_errD, st_errG,
                    st_w_loss, st_s_loss, loss_fake, loss_real, loss_real_st, loss_fake_st, loss_cluster, loss_cluster_gene))
            with torch.no_grad():
                fake,_,_,_,_ = netG.sample_videos(st_motion_input, st_content_input, invert_st_words_embs)
                st_result = save_story_results(st_real_cpu, fake, st_texts, epoch, self.image_dir, i)
                
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.5
                for param_group in st_optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr
                lr_decay_step *= 2

            g_lr, st_lr = 0, 0
            for param_group in optimizerG.param_groups:
                g_lr = param_group['lr']
            for param_group in st_optimizerD.param_groups:
                st_lr = param_group['lr']

            time_mins = int((time.time() - c_time)/60)
            time_hours = int(time_mins / 60)
            epoch_mins = int((time.time()-start_t)/60)
            epoch_hours = int(epoch_mins / 60)

            print("----[{}/{}]Epoch time:{} hours {} mins, Total time:{} hours----".format(epoch, self.max_epoch, epoch_hours, epoch_mins, time_hours))

            if epoch % self.snapshot_interval == 0:
                save_model(netG, netD_st, epoch, self.model_dir)
        save_model(netG, netD_st, self.max_epoch, self.model_dir)
