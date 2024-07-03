import math

import numpy as np
import torch
import torch.nn.functional as F
import math
import random
import torch.nn.functional as F
import torch
import numpy as np
from scipy.stats import beta



def mixup(image, label, alpha=4.0):
    # 随机生成 mixup 参数
    lam = np.random.beta(alpha, alpha)
    #随机排序
    idx = torch.randperm(image.size(0))
    image1 = image[idx]
    label1 = label[idx]
    # 生成 mixup 图像和标签
    mixup_image = lam * image + (1 - lam) * image1
    mixup_label = lam * label + (1 - lam) * label1
    return mixup_image, mixup_label,label,label1,lam
def rand_bbox(size, lam):
    # 计算 cutmix 区域
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    # 随机生成坐标
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2
def cutmix(image, label,beta=1.0):
    # 随机生成 cutmix 参数
    lam = np.random.uniform(low=0,high=1)
    idx = torch.randperm(image.size(0))
    image1 = image[idx]
    label1 = label[idx]
    bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
    # 生成 cutmix 图像和标签
    cutmix_image = image.clone()
    cutmix_image[:, bbx1:bbx2, bby1:bby2] = image1[:, bbx1:bbx2, bby1:bby2]
    cutmix_label = label * (1 - (bbx2 - bbx1) * (bby2 - bby1) / (image.shape[1] * image.shape[2])) + label1 * ((bbx2 - bbx1) * (bby2 - bby1) / (image.shape[1] * image.shape[2]))
    return cutmix_image, cutmix_label,label,label1,lam

def fftfreqnd(h, w=None, z=None):
    """ Get bin values for discrete fourier transform of size (h, w, z)
    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    fz = fx = 0
    fy = np.fft.fftfreq(h)

    if w is not None:
        fy = np.expand_dims(fy, -1)

        if w % 2 == 1:
            fx = np.fft.fftfreq(w)[: w // 2 + 2]
        else:
            fx = np.fft.fftfreq(w)[: w // 2 + 1]

    if z is not None:
        fy = np.expand_dims(fy, -1)
        if z % 2 == 1:
            fz = np.fft.fftfreq(z)[:, None]
        else:
            fz = np.fft.fftfreq(z)[:, None]

    return np.sqrt(fx * fx + fy * fy + fz * fz)


def get_spectrum(freqs, decay_power, ch, h, w=0, z=0):
    """ Samples a fourier image with given size and frequencies decayed by decay power
    :param freqs: Bin values for the discrete fourier transform
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param ch: Number of channels for the resulting mask
    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    scale = np.ones(1) / (np.maximum(freqs, np.array([1. / max(w, h, z)])) ** decay_power)

    param_size = [ch] + list(freqs.shape) + [2]
    param = np.random.randn(*param_size)

    scale = np.expand_dims(scale, -1)[None, :]

    return scale * param


def make_low_freq_image(decay, shape, ch=1):
    """ Sample a low frequency image from fourier space
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param ch: Number of channels for desired mask
    """
    freqs = fftfreqnd(*shape)
    spectrum = get_spectrum(freqs, decay, ch, *shape)  # .reshape((1, *shape[:-1], -1))
    spectrum = spectrum[:, 0] + 1j * spectrum[:, 1]
    mask = np.real(np.fft.irfftn(spectrum, shape))

    if len(shape) == 1:
        mask = mask[:1, :shape[0]]
    if len(shape) == 2:
        mask = mask[:1, :shape[0], :shape[1]]
    if len(shape) == 3:
        mask = mask[:1, :shape[0], :shape[1], :shape[2]]

    mask = mask
    mask = (mask - mask.min())
    mask = mask / mask.max()
    return mask


def sample_lam(alpha, reformulate=False):
    """ Sample a lambda from symmetric beta distribution with given alpha
    :param alpha: Alpha value for beta distribution
    :param reformulate: If True, uses the reformulation of [1].
    """
    if reformulate:
        lam = beta.rvs(alpha + 1, alpha)
    else:
        lam = beta.rvs(alpha, alpha)

    return lam


def binarise_mask(mask, lam, in_shape, max_soft=0.0):
    """ Binarises a given low frequency image such that it has mean lambda.
    :param mask: Low frequency image, usually the result of `make_low_freq_image`
    :param lam: Mean value of final mask
    :param in_shape: Shape of inputs
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :return:
    """
    idx = mask.reshape(-1).argsort()[::-1]
    mask = mask.reshape(-1)
    num = math.ceil(lam * mask.size) if random.random() > 0.5 else math.floor(lam * mask.size)

    eff_soft = max_soft
    if max_soft > lam or max_soft > (1 - lam):
        eff_soft = min(lam, 1 - lam)

    soft = int(mask.size * eff_soft)
    num_low = num - soft
    num_high = num + soft

    mask[idx[:num_high]] = 1
    mask[idx[num_low:]] = 0
    mask[idx[num_low:num_high]] = np.linspace(1, 0, (num_high - num_low))

    mask = mask.reshape((1, *in_shape))
    return mask


def sample_mask(alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    """ Samples a mean lambda from beta distribution parametrised by alpha, creates a low frequency image and binarises
    it based on this lambda
    :param alpha: Alpha value for beta distribution from which to sample mean of mask
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :param reformulate: If True, uses the reformulation of [1].
    """
    if isinstance(shape, int):
        shape = (shape,)

    # Choose lambda
    lam = sample_lam(alpha, reformulate)

    # Make mask, get mean / std
    mask = make_low_freq_image(decay_power, shape)
    mask = binarise_mask(mask, lam, shape, max_soft)

    return lam, mask


def sample_and_apply(x, alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    """
    :param x: Image batch on which to apply fmix of shape [b, c, shape*]
    :param alpha: Alpha value for beta distribution from which to sample mean of mask
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :param reformulate: If True, uses the reformulation of [1].
    :return: mixed input, permutation indices, lambda value of mix,
    """
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)
    index = np.random.permutation(x.shape[0])

    x1, x2 = x * mask, x[index] * (1 - mask)
    return x1 + x2, index, lam
def fmix(images, labels, alpha=1, decay_power=3, shape=(32,32),max_soft = 0,reformulate=False):
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)
    index = torch.randperm(images.size(0)).to(images.device)
    mask = torch.from_numpy(mask).float().to(images.device)

    # Mix the images
    x1 = mask * images
    x2 = (1 - mask) * images[index]
    index = index
    lam = lam
    label = lam * labels + (1-lam)*labels[index]
    return x1 + x2,label,labels,labels[index],lam


def getbox_new(original_size, resized_size):
    # 计算裁剪区域的坐标
    m1 = random.randint(0, original_size[2] - resized_size[2])
    n1 = random.randint(0, original_size[3] - resized_size[3])
    m2 = m1 + resized_size[2]
    n2 = n1 + resized_size[3]
    return m1, n1, m2, n2


def resizemix(input, target, alpha=0.1, beta=0.8):
    batch_size = input.size(0)
    rate = random.uniform(alpha, beta)
    data_small = F.interpolate(input, scale_factor=rate, mode='bilinear', align_corners=False,recompute_scale_factor=True)
    m1, n1, m2, n2 = getbox_new(input.size(), data_small.size())
    rand_index = torch.randperm(input.size(0))

    for i in range(input.shape[0]):
        input[i, :, m1:m2, n1:n2] = data_small[rand_index[i]]

    lam = 1 - (data_small.size(2) * data_small.size(3)) / (input.size(2) * input.size(3))
    target_a = target
    target_b = target[rand_index]
    label = lam * target_a + (1-lam) * target_b

    return input, label,target_a,target_b, lam


def randommix(image, label, alpha=1.0, beta=1.0, decay_power=3.0, shape=(32, 32),replace = True):
    # 随机选择四种图像混合方式
    #return:list[mix1,mix2]
    mix_res = []
    #有放回的选择
    p = np.array([3,1,1,1], dtype=float)
    p /= p.sum()  # normalize
    mix_methods = np.random.choice(['mixup', 'cutmix', 'fmix', 'resizemix'], p=p,size=2,replace = False)

    for mix_method in mix_methods:
        if mix_method == 'mixup':
            img,label,label_a,label_b,lam = mixup(image, label, alpha=4)
        elif mix_method == 'cutmix':
            img,label,label_a,label_b,lam  = cutmix(image, label, beta=1)
        elif mix_method == 'fmix':
            img,label,label_a,label_b,lam  = fmix(image, label, alpha=1, decay_power=decay_power, shape=shape)
        else:
            img,label,label_a,label_b,lam  = resizemix(image, label,0.1,0.8)
        mix_res.append((img,label,label_a,label_b,lam))
    return mix_res
# if __name__ == '__main__':
#     img1 = torch.rand((100,3,32,32))
#     img2 = torch.rand((100, 3,32, 32))
#     label = torch.rand((100))
#     label1 = torch.rand((100))
#     a = randommix(img1,label)
