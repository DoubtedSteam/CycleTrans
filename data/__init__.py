import os
import copy

import math
import torch
import random
import torchvision.transforms as T
import numpy as np

from numpy import fft
from PIL import Image
from torch.utils.data import DataLoader
from data.dataset import SYSUDataset
from data.dataset import RegDBDataset
from data.dataset import MarketDataset

from data.sampler import RandomIdentitySampler

class FFTSketch(object):
    
    def __init__(self, probability=0.5, ratio=0.2, limit=0.5):
        self.probability = probability
        self.ratio = ratio
        self.limit = limit

    def __call__(self, img):

        # if np.random.rand() < self.probability:
        #     return img

        img = np.array(img)
        img = img / 255.
        raw_img = copy.copy(img)
        
        thread = np.random.rand() * self.ratio

        F_img = []
        for i in range(3):
            F_img.append(fft.fftshift(fft.fft2(img[:, :, i])))

        y, x = np.ogrid[0: img.shape[0], 0: img.shape[1]]
        radius = min(img.shape[0], img.shape[1]) * thread
        mask = (x - img.shape[1] // 2)**2 + (y - img.shape[0] // 2)**2 < radius**2
        mask = mask.astype('float')

        limit = (1 - self.limit) * np.random.rand() + self.limit
        F_result = []
        for i in range(3):
            F_result.append(F_img[i] * mask * limit + F_img[i] * (1 - mask))

        result = []
        for i in range(3):
            result.append(np.abs(fft.ifft2(fft.ifftshift(F_result[i]))[:, :, np.newaxis]))
            
        result = np.concatenate(result, axis=2)
        result = (raw_img + result) / 2 * 255
        result = Image.fromarray(result.astype('uint8')).convert('RGB')
        
        return result

class RandomLinear(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        
        mask = torch.ones(img.shape)
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    alphar = np.random.beta(0.5,0.5)
                    alphag = np.random.beta(0.5,0.5)
                    alphab = np.random.beta(0.5,0.5)
                    maxr = (1 / (torch.max(torch.abs(img[0, x1:x1+h, y1:y1+w]) + 1e-12)))
                    maxg = (1 / (torch.max(torch.abs(img[1, x1:x1+h, y1:y1+w]) + 1e-12)))
                    maxb = (1 / (torch.max(torch.abs(img[2, x1:x1+h, y1:y1+w]) + 1e-12)))
                    # print(maxr * alphar)
                    img[0, x1:x1+h, y1:y1+w] = img[0, x1:x1+h, y1:y1+w] * maxr * alphar
                    img[1, x1:x1+h, y1:y1+w] = img[1, x1:x1+h, y1:y1+w] * maxg * alphag
                    img[2, x1:x1+h, y1:y1+w] = img[2, x1:x1+h, y1:y1+w] * maxb * alphab
                    mask[0, x1:x1+h, y1:y1+w] = mask[0, x1:x1+h, y1:y1+w] * maxr * alphar
                    mask[1, x1:x1+h, y1:y1+w] = mask[1, x1:x1+h, y1:y1+w] * maxg * alphag
                    mask[2, x1:x1+h, y1:y1+w] = mask[2, x1:x1+h, y1:y1+w] * maxb * alphab
                else:
                    alpha = np.random.beta(0.5,0.5)
                    maxr = 1 / torch.max(img[0, x1:x1+h, y1:y1+w])
                    img[0, x1:x1+h, y1:y1+w] = img[0, x1:x1+h, y1:y1+w] * maxr * alpha
                    mask[0, x1:x1+h, y1:y1+w] = mask[0, x1:x1+h, y1:y1+w] * maxr * alpha
                min_flag = torch.min(mask.view(-1), 0)[0]
                if min_flag < 1e-6:
                    return img
                    
        return img

class RandomShake(object):
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if np.random.rand() < self.probability:
            return img

        # img = np.array(img)
        new_img = np.array(copy.deepcopy(img))
        raw_img = np.array(copy.deepcopy(img))

        for attempt in range(100):
            # target = np.random.random(size=256) * 4.0 - 1.0
            # target = np.cumsum(target)
            # target = target.astype('uint8')
            # target = np.clip(target, a_min=0, a_max=255)
            target = list(range(256))
            np.random.shuffle(target)
            target = np.array(target).astype('uint8')

            area = img.size[0] * img.size[1]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if w < img.size[0] and h < img.size[1]:
                x1 = random.randint(0, img.size[1] - h)
                y1 = random.randint(0, img.size[0] - w)

                nimg = target[new_img]
                new_img[x1:x1+h, y1:y1+w, :] = nimg[x1:x1+h, y1:y1+w, :]
                
                if np.abs(new_img - raw_img).mean() > 50:
                    # print(np.abs(new_img - raw_img).mean())
                    new_img = Image.fromarray(new_img)
                    # img.save('./img.png')
                    # new_img.save('./new_img.png')
                    # exit()
                    return new_img

        new_img = Image.fromarray(new_img)
        return new_img

class ScaleWithPadding(object):
    def __init__(self, imagesize):
        self.imagesize = (imagesize[1], imagesize[0])

    def __call__(self, img):
        
        size = img.size

        x_ratio = size[0] / self.imagesize[0]
        y_ratio = size[1] / self.imagesize[1]

        if x_ratio > y_ratio:
            ratio = x_ratio
        else:
            ratio = y_ratio

        new_img = Image.new(mode='RGB', size=(int(self.imagesize[0] * ratio + 0.5), int(self.imagesize[1] * ratio + 0.5)), color=(127, 127, 127))
        new_img.paste(img, box=(max(0, (new_img.size[0]-size[0]) // 2), max((new_img.size[1]-size[1]) // 2, 0)))

        return new_img

class ChannelRandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

class ChannelExchange(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, gray = 2):
        self.gray = gray

    def __call__(self, img):
    
        idx = random.randint(0, self.gray)
        
        if idx ==0:
            # random select R Channel
            img[1, :,:] = img[0,:,:]
            img[2, :,:] = img[0,:,:]
        elif idx ==1:
            # random select B Channel
            img[0, :,:] = img[1,:,:]
            img[2, :,:] = img[1,:,:]
        elif idx ==2:
            # random select G Channel
            img[0, :,:] = img[2,:,:]
            img[1, :,:] = img[2,:,:]
        else:
            tmp_img = 0.2989 * img[0,:,:] + 0.5870 * img[1,:,:] + 0.1140 * img[2,:,:]
            img[0,:,:] = tmp_img
            img[1,:,:] = tmp_img
            img[2,:,:] = tmp_img
        return img

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5, sl = 0.2, sh = 0.8, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img


def collate_fn(batch):  # img, label, cam_id, img_path, img_id
    samples = list(zip(*batch))

    data = [torch.stack(x, 0) for i, x in enumerate(samples) if i != 3]
    data.insert(3, samples[3])
    return data


def get_train_loader(dataset, root, sample_method, batch_size, p_size, k_size, image_size, random_flip=False, random_crop=False,
                     random_erase=False, color_jitter=False, padding=0, world_size=1, rank=0, num_workers=4):

    transform = T.Compose([
        T.Resize(image_size),
        RandomShake(probability = 0.5),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.5),
        T.Pad(padding, fill=0), 
        T.RandomCrop(image_size),
        T.ToTensor(), 
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # transform = T.Compose([
    #     # T.ToPILImage(),
    #     T.Resize(image_size),
    #     T.Pad(10),
    #     T.RandomGrayscale(p=0.5),
    #     T.RandomCrop(image_size),
    #     T.RandomHorizontalFlip(),
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     RandomErasing(probability = 0.5, sl = 0.2, sh = 0.8, r1 = 0.3, mean=[0.485, 0.456, 0.406]),
    # ])
    
    # dataset
    if dataset == 'sysu':
        train_dataset = SYSUDataset(root, mode='train', transform=transform)
    elif dataset == 'regdb':
        train_dataset = RegDBDataset(root, mode='train', transform=transform)
    elif dataset == 'market':
        train_dataset = MarketDataset(root, mode='train', transform=transform)

    if sample_method == 'group_sampler':
        batch_size = p_size * k_size
        sampler = RandomIdentitySampler(train_dataset, p_size * k_size, k_size, rank, world_size)
    else:
        batch_size = p_size * k_size
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    # loader
    train_loader = DataLoader(train_dataset, batch_size, sampler=sampler, shuffle=False, 
                              drop_last=True, pin_memory=False,
                              collate_fn=collate_fn, num_workers=num_workers)

    return train_loader, sampler


def get_test_loader(dataset, root, batch_size, image_size, num_workers=4):
    # transform
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # dataset
    if dataset == 'sysu':
        gallery_dataset = SYSUDataset(root, mode='gallery', transform=transform)
        query_dataset = SYSUDataset(root, mode='query', transform=transform)
    elif dataset == 'regdb':
        gallery_dataset = RegDBDataset(root, mode='gallery', transform=transform)
        query_dataset = RegDBDataset(root, mode='query', transform=transform)
    elif dataset == 'market':
        gallery_dataset = MarketDataset(root, mode='gallery', transform=transform)
        query_dataset = MarketDataset(root, mode='query', transform=transform)

    # dataloader
    query_loader = DataLoader(dataset=query_dataset,
                              batch_size=batch_size // 4,
                              shuffle=False,
                              pin_memory=False,
                              drop_last=False,
                              collate_fn=collate_fn,
                              num_workers=num_workers)

    gallery_loader = DataLoader(dataset=gallery_dataset,
                                batch_size=batch_size // 4,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=False,
                                collate_fn=collate_fn,
                                num_workers=num_workers)

    return gallery_loader, query_loader
