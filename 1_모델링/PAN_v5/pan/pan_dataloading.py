import torch
import torch.utils.data as data
from imageio import imread
from pathlib import Path
from PIL import Image
import numpy as np
import logging
from os import listdir
from os.path import splitext
import random
import json
import cv2
import pandas as pd

class BasicDataset(data.Dataset):
    def __init__(self, data_path,masks_dir, scale=1.0, thick=5.0, data_num=-1, mask_suffix='',is_train=False, transform=None):
        self.data_path = Path(data_path) # image
        self.masks_dir = Path(masks_dir) # annotation
        self.mask_suffix = mask_suffix # anno_suffix
        self.thick = thick
        self.data_num = data_num
        self.transform = transform
        self.scale = scale
        self.is_train = is_train

        # data_num이 지정되지 않으면, 모든 데이터를 불러들여온다.

        # if data_num > 0:
        #     full_ids = [splitext(file)[0] for file in listdir(data_path) if not file.startswith('.')]
        #     self.ids = random.sample(full_ids, data_num)
        # else:
        #     self.ids = [splitext(file)[0] for file in listdir(data_path) if not file.startswith('.')]

        # 지금은 1만개라서 이렇게 지정하지만, 나중에는 test, train set 경로를 다르게 해주어 실시한다.
        val_ratio = 0.1
        img_name_list = pd.read_csv('./mask_ratio_over_five.csv')['file_name'].tolist()
        n_train = int(len(img_name_list) * (1-val_ratio))
        index = n_train if (n_train % 2) == 0 else n_train+1 # 홀수면 짝수로 만들어줌

        # 항상 일정한 순서의 데이터 리스트를 갖기 위해서 seed를 설정해준다.
        random.seed(10)
        data_list = [splitext(file)[0] for file in img_name_list if not file.startswith('.')]
        random.shuffle(data_list)

        # train set이라면
        if is_train:
            self.ids = data_list[:index]
        else:
            self.ids = data_list[index:]

        if not self.ids:
            raise RuntimeError(f'No input file found in {data_path}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    @classmethod
    def load(cls, filename,thick=None):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
            # json 데이터를 사전에 처리하지 않고 로딩하면서 처리하기
        elif ext in ['.json']:
            # read json file
            with open(filename, "r", encoding='utf8') as f:
                contents = f.read()
                json_data = json.loads(contents)

            str_fname = str(filename)
            img_pth = Path(
                str_fname.replace("Annotations", "Images").replace("_PLINE.json", ".png"))
            # array 형태로 이미지 로드
            load_img = np.array(Image.open(img_pth))
            # 검정색 색공간 생성.
            lbl = np.zeros((load_img.shape[0], load_img.shape[1]), np.int32) # 어차피 나중에 true mask를 float형태로 변환해주니 이 부분은 굳이 안바꿔도 될듯하다.

            # 차례대로 polylines 불러옴.
            for idx in range(len(json_data["annotations"])):

                temp = np.array(json_data["annotations"][idx]["polyline"]).reshape(-1)
                try:
                    temp_round = np.apply_along_axis(np.round, arr=temp, axis=0)
                    temp_int = np.apply_along_axis(np.int32, arr=temp_round, axis=0)
                except:
                    t = json_data["annotations"][idx]["polyline"]
                    none_json = [[x for x in t[0] if x is not None]]
                    temp = np.array(none_json).reshape(-1)
                    temp_round = np.apply_along_axis(np.round, arr=temp, axis=0)
                    temp_int = np.apply_along_axis(np.int32, arr=temp_round, axis=0)

                temp_re = temp_int.reshape(-1, 2)
                lbl = cv2.polylines(img=lbl,
                                    pts=[temp_re],
                                    isClosed=False,
                                    color=(255),
                                    thickness=thick)
            return Image.fromarray(lbl)
        else:
            return Image.open(filename)  # 이외의 확장자는 그냥 가져옴

    def __getitem__(self, index):
        # 데이터를 불러온 다음 transform 시켜서 반환할 수 있도록 하자.
        name = self.ids[index]
        mask_file = list(self.masks_dir.glob(name+self.mask_suffix+'.*'))
        img_file = list(self.data_path.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'

        # 이미지 데이터 로딩
        mask = self.load(mask_file[0],thick=self.thick)
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
    def __init__(self, data_path, masks_dir, scale=1.0, thick=5, data_num=-1, is_train=False, transform=None):
        super().__init__(data_path, masks_dir, scale=1.0, mask_suffix='_PLINE', thick=thick, data_num=data_num, is_train=is_train, transform=transform)
