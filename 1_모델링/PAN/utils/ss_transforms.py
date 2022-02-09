# https://github.com/jfzhang95/pytorch-deeplab-xception.git

import torch
import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps

class RandomCrop(object):
    def __init__(self, size, padding=0):
        # 생성자로 들어오는 size가 숫자인지 확인
        # True이면 해당 사이즈를 튜플로 변환하여 size에 이미지 사이즈 저장
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
        # 숫자가 아니라면 그냥 저장.
            self.size = size # w, h
        self.padding = padding

    # 객체를 함수처럼 사용할 수 있도록 함. 객체에 x(a, b)와 같은 형태로 값을 넘겨주면 해당 함수가 호출됨
    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']

        ## padding만큼 경계를 채운다.
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        # 원본이미지 사이즈와 mask 사이즈가 다르면 에러를 발생시킨다.
        # 즉, 이미지 사이즈가 서로 다르면 아예 작동하지 않는다.
        assert img.size == mask.size
        w, h = img.size
        tw, th = self.size # target size

        # target 이미지와 원본 이미지 사이즈가 같다면 dictionary 형태로 데이터를 반환한다.
        if w == tw and h == th:
            return {'image': img,
                    'mask': mask}

        # target 이미지의 width or height가 원본에 비해 더 크다면 원본과 mask 이미지를 resize 한다.
        if w < tw or h < th:
            # image upscaling 과정
            img = img.resize((tw, th), Image.BILINEAR) # bilinear 방법을 사용
            mask = mask.resize((tw, th), Image.NEAREST) # nearest 방법을 사용.
            return {'image': img,
                    'mask': mask}

        # target이 원본에 비해 width나 height가 하나라도 더 크면 위의 조건문에서 마무리된다.
        # 따라서 target이 원본이미지보다 모두 작은 경우이다.
        random.seed(10)

        x1 = random.randint(0, w - tw) # 0과 차이값 사이의 임의의 정수를 생성
        y1 = random.randint(0, h - th) # 0과 차이값 사이의 임의의 정수를 생성
        img = img.crop((x1, y1, x1 + tw, y1 + th)) # (가로 시작점, 세로 시작점, 가로 범위, 세로 범위)
        mask = mask.crop((x1, y1, x1 + tw, y1 + th)) # raw data의 좌측, 위측을 잘라낸다.

        return {'image': img,
                'mask': mask}


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['mask']
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': img,
                'mask': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['mask']
        if random.random() < 0.5:
        # random값이 0.5보다 작으면 이미지를 왼쪽에서 오른쪽으로 뒤집는다.
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'mask': mask}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['mask']).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'mask': mask}

class MinMax(object):
    def __init__(self,max):
        self.max = max

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['mask']).astype(np.float32)
        img /= self.max
        mask /= self.max

        return {'image': img,
                'mask': mask}


class Normalize_cityscapes(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.)):
        self.mean = mean

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['mask']).astype(np.float32)
        img -= self.mean
        img /= 255.0

        return {'image': img,
                'mask': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # numpy형태의 image와 torch 형태의 image 규격이 다르기 때문에 torch에 맞게 이미지를 변형시켜준다.
        img = np.array(sample['image']).astype(np.float32).transpose((2, 0, 1))
        mask = np.expand_dims(np.array(sample['mask']).astype(np.float32),-1).transpose((2,0,1))

        img = torch.from_numpy(img).float().contiguous()
        mask = torch.from_numpy(mask).float().contiguous()

        return {'image': img,
                'mask': mask}


class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['mask']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'mask': mask}


class Scale(object):
    def __init__(self, scale):
        # if isinstance(size, numbers.Number):
        #     self.size = (int(size), int(size))
        # else:
        #     self.size = size
        self.scale = scale

    def __call__(self, sample):
        img = sample['image']
        mask = sample['mask']
        assert img.size == mask.size
        w, h = img.size

        # 너비가 높이와 같거나 크고, 입력받은 사이즈의 높이와 너비가 같다면
        # 높이가 너비보다 크거나 같고, 높이가 입력받은 사이즈의 너비와 같다면
        # if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
        #     return {'image': img,
        #             'mask': mask}

        # scaling해서 반환
        oh, ow = int(self.scale*h), int(self.scale*w)
        oh -= oh % 10 # 1의 자리 숫자 빼줌
        ow -= ow % 10
        assert oh > 0 and ow > 0, 'Scale is too small, resized images would have no pixel'

        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        return {'image': img,
                'mask': mask}


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['mask']
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                img = img.resize((self.size, self.size), Image.BILINEAR)
                mask = mask.resize((self.size, self.size), Image.NEAREST)

                return {'image': img,
                        'mask': mask}

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        sample = crop(scale(sample))
        return sample


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['mask']
        rotate_degree = random.random() * 2 * self.degree - self.degree
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'mask': mask}


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['mask']
        assert img.size == mask.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])
        # w = img.size[0]
        # h = img.size[1]

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
        sample = {'image': img, 'mask': mask}

        return self.crop(self.scale(sample))

class RescaleSized(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['mask']
        assert img.size == mask.size

        w, h = self.size[0], self.size[1]

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
        sample = {'image': img, 'mask': mask}

        return sample

class RandomScale(object):
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, sample):
        img = sample['image']
        mask = sample['mask']
        assert img.size == mask.size

        scale = random.uniform(self.limit[0], self.limit[1])
        w = int(scale * img.size[0])
        h = int(scale * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return {'image': img, 'mask': mask}