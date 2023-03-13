import os
import errno
import numpy as np
import PIL
from copy import deepcopy
from miscc.config import cfg
import pdb
from torch.nn import init
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.autograd import Variable
import random
from tqdm import tqdm
import torch.nn.functional as F

from Attention import func_attention
from nltk.tokenize import RegexpTokenizer
from kmeans_pytorch import kmeans, kmeans_predict

def Focal_Loss(bce_loss, targets, gamma = 2.0, alpha = 1):
    p_t = torch.exp(-bce_loss)
    f_loss = alpha * (1 - p_t) ** gamma * bce_loss
    return f_loss.mean()

# cosine similarity
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def sent_loss(cnn_code, rnn_code, labels,
              batch_size, eps=1e-8):
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3

    scores0 = scores0.squeeze()
    scores1 = scores0.transpose(0, 1)

    loss_0 = tf_cross_entropy_loss_with_logits(
                labels=labels, logits=scores0)
    loss_1 = tf_cross_entropy_loss_with_logits(
                labels=labels, logits=scores1)

    loss_0 = torch.mean(loss_0)
    loss_1 = torch.mean(loss_1)

    return loss_0, loss_1


def words_loss(img_features, words_emb, labels,
               cap_lens, batch_size):

    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        words_num = cap_lens[i]
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        word = word.repeat(batch_size, 1, 1)
        context = img_features

        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)

        row_sim = cosine_similarity(word, weiContext)
        row_sim = row_sim.view(batch_size, words_num)

        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)

    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3
    similarities1 = similarities.transpose(0, 1)


    loss_0 = tf_cross_entropy_loss_with_logits(
                labels=labels, logits=similarities)
    loss_1 = tf_cross_entropy_loss_with_logits(
                labels=labels, logits=similarities1)

    loss_0 = torch.mean(loss_0)
    loss_1 = torch.mean(loss_1)

    return loss_0, loss_1



def words_loss_st(img_features, words_emb, labels,
               cap_lens, batch_size):

    res_loss0 = 0
    res_loss1 = 0

    cap_lens = cap_lens.data.tolist()
    for jj in range(words_emb.size(1)):
        similarities = []
        cur_st_words = words_emb[:,jj,:,:]

        for i in range(batch_size):
            words_num = cap_lens[batch_size*(jj)+i]
            word = cur_st_words[i, :, :words_num].unsqueeze(0).contiguous()
            word = word.repeat(batch_size, 1, 1)
            context = img_features

            weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)
            word = word.transpose(1, 2).contiguous()
            weiContext = weiContext.transpose(1, 2).contiguous()
            word = word.view(batch_size * words_num, -1)
            weiContext = weiContext.view(batch_size * words_num, -1)

            row_sim = cosine_similarity(word, weiContext)
            row_sim = row_sim.view(batch_size, words_num)

            row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)

            similarities.append(row_sim)

        similarities = torch.cat(similarities, 1)

        similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3
        similarities1 = similarities.transpose(0, 1)


        loss_0 = tf_cross_entropy_loss_with_logits(
                labels=labels, logits=similarities)
        loss_1 = tf_cross_entropy_loss_with_logits(
                labels=labels, logits=similarities1)

        loss_0 = torch.mean(loss_0)
        loss_1 = torch.mean(loss_1)

        res_loss0 += loss_0
        res_loss1 += loss_1


    return res_loss0, res_loss1

#############################
def check_is_order(sequence):
    return (np.diff(sequence)>=0).all()

def create_random_shuffle(stories,  random_rate=0.5):
    o3n_data, labels = [], []
    device = stories.device
    stories = stories.cpu()
    story_size = len(stories)
    for idx, result in enumerate(stories):
        video_len = result.shape[1]
        label = 1 if random_rate > np.random.random() else 0
        if label == 0:
            o3n_data.append(result.clone())
        else:
            random_sequence = random.sample(range(video_len), video_len)
            while (check_is_order(random_sequence)): # make sure not sorted
                np.random.shuffle(random_sequence)
            shuffled_story = result[:, list(random_sequence), :, :].clone()
            story_size_idx = random.randint(0, story_size-1)
            if story_size_idx != idx:
                story_mix = random.sample(range(video_len), 1)
                shuffled_story[:, story_mix, :, : ] = stories[story_size_idx, :, story_mix, :, :].clone()
            o3n_data.append(shuffled_story)
        labels.append(label)

    order_labels = Variable(torch.from_numpy(np.array(labels)).float(), requires_grad=True).detach()
    shuffle_imgs = Variable(torch.stack(o3n_data, 0), requires_grad=True)
    return shuffle_imgs.to(device), order_labels.to(device)


def l2_normalize(x, axis=None, epsilon=1e-12):
    epsilon = torch.tensor(epsilon)
    square_sum = torch.sum(torch.square(x), axis=axis, keepdims=True)
    x_inv_norm = torch.rsqrt(torch.maximum(square_sum, epsilon))
    return torch.multiply(x, x_inv_norm)


def tf_cross_entropy_loss_with_logits(labels, logits):
    logp = F.log_softmax(logits)
    loss = - torch.sum(torch.multiply(labels, logp), axis=-1)
    return loss



def compute_discriminator_loss(netD, real_imgs, fake_imgs,
                               real_labels, fake_labels,real_catelabels,
                               conditions, gpus, word_embs, new_word_emb,
                               sent_emb, image_encoder, match_labels, cap_lens, batch_texts, batch_imgs, cluster_flag):

    dis_loss = 0
    batch_size = cfg.TRAIN.IM_BATCH_SIZE

    st_real = torch.mean(real_imgs, dim = 2)
    st_sent = sent_emb.view(real_imgs.size(0),-1, sent_emb.size(1))

    st_sent = torch.mean(st_sent, dim = 1)

    region_features, cnn_code = image_encoder(st_real)

    local_batch_size = region_features.size(0)
    labels = F.one_hot(torch.arange(local_batch_size), local_batch_size).cuda()

    
    cnn_code = l2_normalize(cnn_code, -1)
    st_sent = l2_normalize(st_sent, -1)

    temperature = torch.tensor(0.1)

    logits_img2cond = torch.matmul(cnn_code,
                                 torch.transpose(st_sent,0,1).contiguous()) / temperature
    logits_cond2img = torch.matmul(st_sent,
                                 torch.transpose(cnn_code,0,1).contiguous()) / temperature

    loss_img2cond = tf_cross_entropy_loss_with_logits(
        labels=labels, logits=logits_img2cond)
    loss_cond2img = tf_cross_entropy_loss_with_logits(
        labels=labels, logits=logits_cond2img)
    loss_img2cond = torch.mean(loss_img2cond)
    loss_cond2img = torch.mean(loss_cond2img)
    loss_real_st = loss_img2cond + loss_cond2img

    dis_loss = dis_loss + loss_real_st

    st_word_embs = word_embs.view(real_imgs.size(0), -1, word_embs.size(1), word_embs.size(2))
    st_new_word_emb = new_word_emb.view(real_imgs.size(0), -1, new_word_emb.size(1), new_word_emb.size(2))

    st_word_real = torch.mean(real_imgs, dim = 2)
    
    region_features, cnn_code = image_encoder(st_word_real)
    st_cap_lens = cap_lens.view(region_features.size(0), -1)
    st_cap_lens = torch.sum(st_cap_lens, dim=1)
    lables = F.one_hot(torch.arange(region_features.size(0)), region_features.size(0))
    lables = lables.cuda()
    w_loss0, w_loss1 = words_loss_st(region_features, st_word_embs,
                                        lables, cap_lens,
                                        region_features.size(0))

    w_loss = (w_loss0 + w_loss1) * \
                cfg.TRAIN.SMOOTH.LAMBDA

    dis_loss = dis_loss + w_loss

    st_real_imgs = real_imgs.view(-1, real_imgs.size(1), real_imgs.size(3), real_imgs.size(4))

    if cluster_flag:
        temperature = torch.tensor(0.1).cuda()
        words_features, sent_code = image_encoder(batch_imgs)
        nef, att_sze = words_features.size(1), words_features.size(2)
        
        word_embs_trans = torch.transpose(batch_texts, 1, 2).contiguous()
        region_features = words_features.view(words_features.size(0), words_features.size(1), -1)
        confusion_features = torch.bmm(word_embs_trans, region_features)

        confusion_features = confusion_features.view(confusion_features.size(0), -1)
        list_centroid = []
        cluster_ids_x, cluster_centers = kmeans(
            X=confusion_features, num_clusters=8, distance='euclidean', device=torch.device(int(cfg.GPU_ID))
            )


        for idx in cluster_ids_x:
            list_centroid.append(cluster_centers[idx].unsqueeze(0))

        list_centroid = torch.cat(list_centroid, 0)
        list_centroid = list_centroid.cuda()

        local_batch_size = confusion_features.size(0)
        labels = F.one_hot(torch.arange(local_batch_size), local_batch_size).cuda()

        normal_confusion_features = l2_normalize(confusion_features, -1)
        normal_cluster_centers = l2_normalize(list_centroid, -1) 

        logits_conf2center = torch.matmul(normal_confusion_features,
                                 torch.transpose(normal_cluster_centers,0,1).contiguous()) / temperature
        logits_center2conf = torch.matmul(normal_cluster_centers,
                                 torch.transpose(normal_confusion_features,0,1).contiguous()) / temperature


        loss_conf2center = tf_cross_entropy_loss_with_logits(
            labels=labels, logits=logits_conf2center)
        loss_center2conf = tf_cross_entropy_loss_with_logits(
            labels=labels, logits=logits_center2conf)

        loss_conf2center = torch.mean(loss_conf2center)
        loss_center2conf = torch.mean(loss_center2conf)
        loss_cluster = loss_conf2center + loss_center2conf

        dis_loss = dis_loss + loss_cluster

    st_real_imgs = real_imgs.view(-1, real_imgs.size(1), real_imgs.size(3), real_imgs.size(4))
    
    region_features, cnn_code = image_encoder(st_real_imgs)

    local_batch_size = region_features.size(0)
    labels = F.one_hot(torch.arange(local_batch_size), local_batch_size).cuda()

    cnn_code = l2_normalize(cnn_code, -1)
    sent_emb = l2_normalize(sent_emb, -1)

    temperature = torch.tensor(0.1)

    logits_img2cond = torch.matmul(cnn_code,
                                 torch.transpose(sent_emb,0,1).contiguous()) / temperature
    logits_cond2img = torch.matmul(sent_emb,
                                 torch.transpose(cnn_code,0,1).contiguous()) / temperature

    loss_img2cond = tf_cross_entropy_loss_with_logits(
        labels=labels, logits=logits_img2cond)
    loss_cond2img = tf_cross_entropy_loss_with_logits(
        labels=labels, logits=logits_cond2img)
    loss_img2cond = torch.mean(loss_img2cond)
    loss_cond2img = torch.mean(loss_cond2img)
    loss_real = loss_img2cond + loss_cond2img

    dis_loss = dis_loss + loss_real

    if real_imgs.size(0) < cfg.TRAIN.IM_BATCH_SIZE:
        N, C, video_len, W, H = real_imgs.shape
        real_imgs_w = real_imgs.permute(0,2,1,3,4)
        real_imgs_w = real_imgs_w.contiguous().view(-1, C,W,H)
    region_features, cnn_code = image_encoder(real_imgs_w)

    
    w_loss0, w_loss1 = words_loss(region_features, word_embs,
                                        match_labels, cap_lens,
                                        batch_size)
    w_loss = (w_loss0 + w_loss1) * \
                cfg.TRAIN.SMOOTH.LAMBDA

    s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb,
                                match_labels, batch_size)
    s_loss = (s_loss0 + s_loss1) * \
                cfg.TRAIN.SMOOTH.LAMBDA

    dis_loss = dis_loss + w_loss + s_loss

    criterion = nn.BCELoss()
    cate_criterion = nn.MultiLabelSoftMarginLoss()
    batch_size = real_imgs.size(0)
    fake = fake_imgs.detach()



    cond = conditions.detach() 
    real_features = nn.parallel.data_parallel(netD, (real_imgs, new_word_emb), gpus)
    fake_features = nn.parallel.data_parallel(netD, (fake, new_word_emb), gpus)


    if netD.get_uncond_logits is not None:
        real_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (real_features), gpus)
        fake_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (fake_features), gpus)
        uncond_errD_real = criterion(real_logits, real_labels)
        uncond_errD_fake = criterion(fake_logits, fake_labels)
        errD = uncond_errD_real + uncond_errD_fake 
        errD_real = uncond_errD_real 
        errD_fake = uncond_errD_fake 
    else:
        errD = errD_real + (errD_fake + errD_wrong) * 0.5

    errD = errD + dis_loss


    if cluster_flag:
        return errD, errD_real.data, errD_fake.data, loss_real, loss_real_st, loss_cluster
    else:
        return errD, errD_real.data, errD_fake.data, loss_real, loss_real_st, 0



##### Generating Loss #####
def compute_generator_loss(netD, fake_imgs, real_imgs, real_labels, fake_catelabels, 
                        conditions, gpus, image_encoder, word_embs, sent_emb, 
                        cap_lens, match_labels, new_word_emb, batch_texts, batch_fake_imgs, cluster_flag):
    criterion = nn.BCELoss()
    cate_criterion =nn.MultiLabelSoftMarginLoss()
    batch_size = cfg.TRAIN.IM_BATCH_SIZE
    
    cond = conditions.detach()
    fake_features = nn.parallel.data_parallel(netD, (fake_imgs, new_word_emb), gpus)
     
    if netD.get_uncond_logits is not None:
        fake_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (fake_features), gpus)
        uncond_errD_fake = criterion(fake_logits, real_labels)
        errD_fake = uncond_errD_fake
    
    st_fake_imgs = fake_imgs.reshape(-1, fake_imgs.size(1), fake_imgs.size(3), real_imgs.size(4))
    region_features, cnn_code = image_encoder(st_fake_imgs)

    local_batch_size = region_features.size(0)
    labels = F.one_hot(torch.arange(local_batch_size), local_batch_size).cuda()

    cnn_code = l2_normalize(cnn_code, -1)
    sent_emb = l2_normalize(sent_emb, -1)

    temperature = torch.tensor(0.1)

    logits_img2cond = torch.matmul(cnn_code,
                                 torch.transpose(sent_emb,0,1).contiguous()) / temperature
    logits_cond2img = torch.matmul(sent_emb,
                                 torch.transpose(cnn_code,0,1).contiguous()) / temperature

    loss_img2cond = tf_cross_entropy_loss_with_logits(
        labels=labels, logits=logits_img2cond)
    loss_cond2img = tf_cross_entropy_loss_with_logits(
        labels=labels, logits=logits_cond2img)
    loss_img2cond = torch.mean(loss_img2cond)
    loss_cond2img = torch.mean(loss_cond2img)
    loss_fake = loss_img2cond + loss_cond2img

    st_fake = torch.mean(fake_imgs, dim = 2)
    st_sent = sent_emb.view(fake_imgs.size(0),-1, sent_emb.size(1))
    st_sent = torch.mean(st_sent, dim = 1)
    region_features, cnn_code = image_encoder(st_fake)

    local_batch_size = region_features.size(0)
    labels = F.one_hot(torch.arange(local_batch_size), local_batch_size).cuda()


    cnn_code = l2_normalize(cnn_code, -1)
    st_sent = l2_normalize(st_sent, -1)

    temperature = torch.tensor(0.1)

    logits_img2cond = torch.matmul(cnn_code,
                                 torch.transpose(st_sent,0,1).contiguous()) / temperature
    logits_cond2img = torch.matmul(st_sent,
                                 torch.transpose(cnn_code,0,1).contiguous()) / temperature

    loss_img2cond = tf_cross_entropy_loss_with_logits(
        labels=labels, logits=logits_img2cond)
    loss_cond2img = tf_cross_entropy_loss_with_logits(
        labels=labels, logits=logits_cond2img)
    loss_img2cond = torch.mean(loss_img2cond)
    loss_cond2img = torch.mean(loss_cond2img)
    loss_fake_st = loss_img2cond + loss_cond2img

    errD_fake += loss_fake_st

    # clustering losses
    if cluster_flag:
        temperature = torch.tensor(0.1).cuda()
        words_features, sent_code = image_encoder(batch_fake_imgs)
        nef, att_sze = words_features.size(1), words_features.size(2)
        word_embs_trans = torch.transpose(batch_texts, 1, 2).contiguous()
        region_features = words_features.view(words_features.size(0), words_features.size(1), -1)
        confusion_features = torch.bmm(word_embs_trans, region_features)

        confusion_features = confusion_features.view(confusion_features.size(0), -1)
        list_centroid = []
        cluster_ids_x, cluster_centers = kmeans(
            X=confusion_features, num_clusters=8, distance='euclidean', device=torch.device(int(cfg.GPU_ID))
            )


        for idx in cluster_ids_x:
            list_centroid.append(cluster_centers[idx].unsqueeze(0))

        list_centroid = torch.cat(list_centroid, 0)
        list_centroid = list_centroid.cuda()

        local_batch_size = confusion_features.size(0)
        labels = F.one_hot(torch.arange(local_batch_size), local_batch_size).cuda()

        normal_confusion_features = l2_normalize(confusion_features, -1)
        normal_cluster_centers = l2_normalize(list_centroid, -1)       

        logits_conf2center = torch.matmul(normal_confusion_features,
                                 torch.transpose(normal_cluster_centers,0,1).contiguous()) / temperature
        logits_center2conf = torch.matmul(normal_cluster_centers,
                                 torch.transpose(normal_confusion_features,0,1).contiguous()) / temperature

        loss_conf2center = tf_cross_entropy_loss_with_logits(
            labels=labels, logits=logits_conf2center)
        loss_center2conf = tf_cross_entropy_loss_with_logits(
            labels=labels, logits=logits_center2conf)
        loss_conf2center = torch.mean(loss_conf2center)
        loss_center2conf = torch.mean(loss_center2conf)
        loss_cluster = loss_conf2center + loss_center2conf

        errD_fake = errD_fake + loss_cluster



    if fake_imgs.size(0) < cfg.TRAIN.IM_BATCH_SIZE:
        N, C, video_len, W, H = fake_imgs.shape
        fake_imgs = fake_imgs.permute(0,2,1,3,4)
        fake_imgs = fake_imgs.contiguous().view(-1, C,W,H)
    region_features, cnn_code = image_encoder(fake_imgs)
    w_loss0, w_loss1 = words_loss(region_features, word_embs,
                                        match_labels, cap_lens,
                                        batch_size)
    w_loss = (w_loss0 + w_loss1) * \
                cfg.TRAIN.SMOOTH.LAMBDA

    s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb,
                                match_labels, batch_size)
    s_loss = (s_loss0 + s_loss1) * \
                cfg.TRAIN.SMOOTH.LAMBDA

    errD_fake += w_loss + s_loss

    errD_fake += loss_fake



    if cluster_flag:
        return errD_fake, w_loss, s_loss, loss_fake, loss_fake_st, loss_cluster
    else:
        return errD_fake, w_loss, s_loss, loss_fake, loss_fake_st, 0



def compute_cyc_loss_img(loss_fn, st_cyc_imgs, st_real_imgs):
    loss = loss_fn(st_cyc_imgs, st_real_imgs) 
    return loss


def compute_cyc_loss_txt(loss_fn, st_motion_cyc, st_motion_input):
    loss = loss_fn(st_motion_cyc, st_motion_input).mean()
    return loss

def KL_loss(mu, logvar):
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

#############################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


#############################
def save_img_results(data_img, fake, texts, epoch, image_dir):
    num = cfg.VIS_COUNT
    fake = fake[0:num]
    if data_img is not None:
        data_img = data_img[0:num]
        vutils.save_image(
            data_img, '%s/real_samples_epoch_%03d.png' % 
            (image_dir, epoch), normalize=True)
        vutils.save_image(
            fake.data, '%s/fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)
    else:
        vutils.save_image(
            fake.data, '%s/lr_fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)

    if texts is not None:
        fid = open('%s/lr_fake_samples_epoch_%03d.txt' % (image_dir, epoch), 'wb')
        for i in range(num):
            fid.write(str(i) + ':' + texts[i] + '\n')
        fid.close()

##########################\
def images_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(1,2,0)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')

def save_story_results(ground_truth, images, texts, name, image_dir, step=0, lr = False):
    video_len = cfg.VIDEO_LEN
    all_images = []
    for i in range(images.shape[0]):
        all_images.append(vutils.make_grid(torch.transpose(images[i], 0,1), video_len))
    all_images= vutils.make_grid(all_images, 1)
    all_images = images_to_numpy(all_images)
    
    if ground_truth is not None:
        gts = []
        for i in range(ground_truth.shape[0]):
            gts.append(vutils.make_grid(torch.transpose(ground_truth[i], 0,1), video_len))
        gts = vutils.make_grid(gts, 1)
        gts = images_to_numpy(gts)
        all_images = np.concatenate([all_images, gts], axis = 1)


    output = PIL.Image.fromarray(all_images)
    if lr:
        output.save('{}/lr_samples_{}_{}.png'.format(image_dir, name, step))
    else:
        output.save('{}/fake_samples_{}_{}.png'.format(image_dir, name, step))

    if texts is not None:
        fid = open('{}/fake_samples_{}.txt'.format(image_dir, name), 'w')
        for idx in range(images.shape[0]):
            fid.write(str(idx) + '--------------------------------------------------------\n')
            for i in range(len(texts)):
                fid.write(texts[i][idx] +'\n' )
            fid.write('\n\n')
        fid.close()
    return all_images

def save_image_results(ground_truth, images, size=cfg.IMSIZE):
    video_len = cfg.VIDEO_LEN
    st_bs = cfg.TRAIN.ST_BATCH_SIZE
    images = images.reshape(st_bs, video_len, -1, size, size)
    all_images = []
    for i in range(images.shape[0]):
        all_images.append(vutils.make_grid(images[i], video_len))
    all_images= vutils.make_grid(all_images, 1)
    all_images = images_to_numpy(all_images)
    
    if ground_truth is not None:
        ground_truth = ground_truth.reshape(st_bs, video_len, -1, size, size)
        gts = []
        for i in range(ground_truth.shape[0]):
            gts.append(vutils.make_grid(ground_truth[i], video_len))
        gts = vutils.make_grid(gts, 1)
        gts = images_to_numpy(gts)
        all_images = np.concatenate([all_images, gts], axis = 1)
    return all_images

def save_all_img(images, count, image_dir):
    bs, size_c, v_len, size_w, size_h = images.shape
    for b in range(bs):
        imgs = images[b].transpose(0,1)
        for i in range(v_len):
            count += 1
            png_name = os.path.join(image_dir, "{}.png".format(count))
            vutils.save_image(imgs[i], png_name)
    return count

def get_multi_acc(predict, real):
    predict = 1/(1+np.exp(-predict))
    correct = 0
    for i in range(predict.shape[0]):
        for j in range(predict.shape[1]):
            if real[i][j] == 1 and predict[i][j]>=0.5 :
                correct += 1
    acc = correct / float(np.sum(real))
    return acc

def save_model(netG, netD_st, epoch, model_dir, whole=False):
    if whole == True:
        torch.save(netG, '%s/netG.pkl' % (model_dir))
        torch.save(netD_st, '%s/netD_st.pkl' % (model_dir))
        print('Save G/D model')
        return
    torch.save(netG.state_dict(),'%s/netG_epoch_%d.pth' % (model_dir, epoch))
    torch.save(netD_st.state_dict(),'%s/netD_st_epoch_last.pth' % (model_dir))
    print('Save G/D models ')

def mkdir_p(path):
    os.makedirs(path, exist_ok=True)

def save_test_samples(netG, dataloader, save_path):
    print('Generating Test Samples...')
    save_images = []
    save_labels = []
    for i, batch in enumerate(dataloader, 0):
        #print('Processing at ' + str(i))
        real_cpu = batch['images']
        motion_input = batch['description'][:, :, :cfg.TEXT.DIMENSION]
        content_input = batch['description'][:, :, :cfg.TEXT.DIMENSION]
        catelabel = batch['labels']
        real_imgs = Variable(real_cpu)
        motion_input = Variable(motion_input)
        content_input = Variable(content_input)
        if cfg.CUDA:
            real_imgs = real_imgs.cuda()            
            motion_input = motion_input.cuda()
            content_input = content_input.cuda()
            catelabel = catelabel.cuda()
        motion_input = torch.cat((motion_input, catelabel), 2)
        _, fake, _,_,_,_,_ = netG.sample_videos(motion_input, content_input)
        save_story_results(real_cpu, fake, batch['text'], '{:03d}'.format(i), save_path)
        save_images.append(fake.cpu().data.numpy())
        save_labels.append(catelabel.cpu().data.numpy())
    save_images = np.concatenate(save_images, 0)
    save_labels = np.concatenate(save_labels, 0)
    np.save(save_path + '/images.npy', save_images)
    np.save(save_path + '/labels.npy', save_labels)

def save_train_samples(netG, dataloader, save_path):
    print('Generating Train Samples...')
    save_images = []
    save_labels = []
    for i, batch in enumerate(dataloader, 0):
        real_cpu = batch['images']
        motion_input = batch['description'][:, :, :cfg.TEXT.DIMENSION]
        content_input = batch['description'][:, :, :cfg.TEXT.DIMENSION]
        catelabel = batch['labels']
        real_imgs = Variable(real_cpu)
        motion_input = Variable(motion_input)
        content_input = Variable(content_input)
        if cfg.CUDA:
            real_imgs = real_imgs.cuda()
            motion_input = motion_input.cuda()
            content_input = content_input.cuda()
            catelabel = catelabel.cuda()
        motion_input = torch.cat((motion_input, catelabel), 2)
        _, fake, _,_,_,_,_ = netG.sample_videos(motion_input, content_input)
        save_story_results(real_cpu, fake, batch['text'], '{:05d}'.format(i), save_path)
        save_images.append(fake.cpu().data.numpy())
        save_labels.append(catelabel.cpu().data.numpy())
    save_images = np.concatenate(save_images, 0)
    save_labels = np.concatenate(save_labels, 0)
    np.save(save_path + '/images.npy', save_images)
    np.save(save_path + '/labels.npy', save_labels)


def inference_samples(netG, dataloader, save_path, text_encoder, wordtoix):
    print('Generate and save images...')

    mkdir_p(save_path)
    mkdir_p('./Evaluation/ref')
    cnt_gen = 0
    cnt_ref = 0
    for i, batch in enumerate(tqdm(dataloader, desc='Saving')):
        real_cpu = batch['images']
        motion_input = batch['description'][:, :, :cfg.TEXT.DIMENSION]
        content_input = batch['description'][:, :, :cfg.TEXT.DIMENSION]
        catelabel = batch['labels']
        real_imgs = Variable(real_cpu)
        motion_input = Variable(motion_input)
        content_input = Variable(content_input)
        if cfg.CUDA:
            real_imgs = real_imgs.cuda()
            motion_input = motion_input.cuda()
            content_input = content_input.cuda()
            catelabel = catelabel.cuda()

        if 1:
            st_texts = None
            if 'text' in batch:
                st_texts = batch['text']

            new_list = []
            for idx in range(cfg.TRAIN.ST_BATCH_SIZE):
                for j in range(cfg.VIDEO_LEN):
                    cur = st_texts[j]
                    new_list.append(cur[idx])

            captions = []
            cap_lens = []
            for cur_text in new_list:
                        
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
            cap_lens = cap_lens[sorted_indices]

            cap_array = np.zeros((len(captions), max_len), dtype='int64')
            for pointer in range(len(captions)):
                idx_cap = sorted_indices[pointer]
                cap = captions[idx_cap]
                c_len = len(cap)
                cap_array[pointer, :c_len] = cap


            new_captions = cap_array
            batch_size = new_captions.shape[0]
            new_captions = Variable(torch.from_numpy(new_captions), volatile=True)
            cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

            new_captions = new_captions.cuda()
            cap_lens = cap_lens.cuda()

            batch_size = new_captions.size(0)
            hidden = text_encoder.init_hidden(batch_size)
            words_embs, sent_emb = text_encoder(new_captions, cap_lens, hidden)

            invert_sent_emb = Variable(torch.FloatTensor(sent_emb.size(0), sent_emb.size(1)).fill_(0)).cuda()
            invert_words_embs = Variable(torch.FloatTensor(words_embs.size(0), words_embs.size(1), words_embs.size(2)).fill_(0)).cuda()

            for pointer in range(len(sent_emb)):
                idx_cap = sorted_indices[pointer]
                cur_sent = sent_emb[pointer, :]
                cur_word = words_embs[pointer, :, :]
                invert_sent_emb[idx_cap, :] = cur_sent
                invert_words_embs[idx_cap, :, :] = cur_word
            invert_st_sent_emb = invert_sent_emb.view(-1, cfg.VIDEO_LEN, invert_sent_emb.size(1))

            motion_input = torch.cat((invert_st_sent_emb, catelabel), 2) 
            content_input = invert_st_sent_emb

        _, fake, _,_,_,_,_ = netG.sample_videos(motion_input, content_input, invert_words_embs, None)
        cnt_gen = save_all_img(fake, cnt_gen, save_path)
        cnt_ref = save_all_img(real_imgs, cnt_ref, './Evaluation/ref')

  
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count 


if __name__ == "__main__":
    test = torch.randn((14, 3, 5, 64,64))
    output, labels = create_random_shuffle(test)
    print(output.shape)
