import torch
import torch.utils.data as data
from imageio import imread
from pathlib import Path
from PIL import Image
import numpy as np
import logging
from os import listdir
from os.path import splitext

class BasicDataset(data.Dataset):
    def __init__(self, data_path,masks_dir, scale=1.0, mask_suffix='', transform=None):
        self.data_path = Path(data_path)
        self.masks_dir = Path(masks_dir)
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.scale = scale

        self.ids = [splitext(file)[0] for file in listdir(data_path) if file.endswith('.jpg')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {data_path}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, index):
        # 데이터를 불러온 다음 transform 시켜서 반환할 수 있도록 하자.
        name = self.ids[index]
        mask_file = list(self.masks_dir.glob(name+self.mask_suffix+'.gif'))
        img_file = list(self.data_path.glob(name + '.jpg'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'

        # 이미지 데이터 로딩
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        # 전처리
        sample = {'image': img, 'mask': mask}
        if self.transform is not None:
            sample = self.transform(sample)

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        return sample

    def __len__(self):
        return len(self.ids)

class CarvanaDataset(BasicDataset):
    def __init__(self, data_path, masks_dir, scale=1.0, transform=None):
        super().__init__(data_path, masks_dir, scale=1.0, mask_suffix='_mask', transform=transform)
