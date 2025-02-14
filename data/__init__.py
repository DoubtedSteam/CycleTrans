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


class RandomLinear(object):
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


def collate_fn(batch):  # img, label, cam_id, img_path, img_id
    samples = list(zip(*batch))

    data = [torch.stack(x, 0) for i, x in enumerate(samples) if i != 3]
    data.insert(3, samples[3])
    return data


def get_train_loader(dataset, root, sample_method, batch_size, p_size, k_size, image_size, random_flip=False, random_crop=False,
                     random_erase=False, color_jitter=False, padding=0, world_size=1, rank=0, num_workers=4):
    # data pre-processing
    t = [T.Resize(image_size)]

    # t.append(RandomShake())

    if random_flip:
        t.append(T.RandomHorizontalFlip())

    if color_jitter:
        t.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))

    if random_crop:
        t.extend([T.Pad(padding, fill=127), T.RandomCrop(image_size)])

    t.extend([T.ToTensor(), 
            #   T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
              T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # if random_erase:
    #     t.append(T.RandomErasing())

    t.append(RandomLinear())

    transform = T.Compose(t)

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
        # T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
