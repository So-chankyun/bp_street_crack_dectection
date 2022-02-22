import torch
import json
import cv2
import random
import torch.utils.data as data
from imageio import imread
from pathlib import Path
from PIL import Image
import numpy as np
import logging
from os import listdir
from os.path import splitext

class BasicDataset(data.Dataset):
    def __init__(self, 
                 data_path,
                 masks_dir, 
                 scale=1.0,
                 transform=None,
                 thick=10,
                 data_num=-1,
                 mask_suffix=''):
        self.data_path = Path(data_path)
        self.masks_dir = Path(masks_dir)
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.scale = scale
        self.thick = thick

        if data_num > 0:
            full_ids = [splitext(file)[0] for file in listdir(data_path) if not file.startswith('.')]
            self.ids = random.sample(full_ids, data_num)
        else:
            self.ids = [splitext(file)[0] for file in listdir(data_path) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {data_path}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    @classmethod
    def load(cls, filename, thick):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        
        elif ext in ['.json']:
            with open(filename,"r",encoding='utf8') as f:
                contents = f.read()
                json_data = json.loads(contents)
            
            str_fname = str(filename)
            img_pth = Path(str_fname.replace("Annotations","Images").replace("_PLINE.json",".png").replace("annotations","images"))            
            load_img = np.array(Image.open(img_pth))    
            lbl = np.zeros((load_img.shape[0], load_img.shape[1]), np.int32)
        
            for idx in range(len(json_data["annotations"])):
                
                temp = np.array(json_data["annotations"][idx]["polyline"]).reshape(-1)
                try:
                    temp_round = np.apply_along_axis(np.round, arr=temp, axis=0)
                    temp_int = np.apply_along_axis(np.int32, arr=temp_round, axis=0)
                except:
                    t = json_data["annotations"][idx]["polyline"]
                    none_json = [[x for x in t[0] if x is not None]]
                    temp = np.array(none_json).reshape(-1)
                    temp_round = np.apply_along_axis(np.round , arr=temp, axis=0)
                    temp_int = np.apply_along_axis(np.int32, arr=temp_round, axis=0)
                    
                temp_re = temp_int.reshape(-1, 2)
                lbl = cv2.polylines(img=lbl,
                            pts=[temp_re],
                            isClosed=False,
                            color=(1),
                            thickness=thick)
            return Image.fromarray(lbl)
        else:
            return Image.open(filename)     # 이외의 확장자는 그냥 가져옴


    def __getitem__(self, index):
        # 데이터를 불러온 다음 transform 시켜서 반환할 수 있도록 하자.
        name = self.ids[index]
        mask_file = list(self.masks_dir.glob(name+self.mask_suffix+'.*'))
        img_file = list(self.data_path.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'

        # 이미지 데이터 로딩
        mask = self.load(mask_file[0], thick=self.thick)
        img = self.load(img_file[0], thick=self.thick)

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
    def __init__(self, data_path, masks_dir, scale=1.0, transform=None, thick=10, data_num=-1):
        super().__init__(data_path, masks_dir, scale, transform, thick, data_num, mask_suffix='_PLINE')
