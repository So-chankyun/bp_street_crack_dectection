import logging
import torch
import numpy as np
import json
import cv2

from PIL import Image
from pathlib import Path
from os import listdir
from os.path import splitext
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, img_path: str, ann_path: str, scale: float=1.0, ann_suffix: str= ''):
        self.img_path = Path(img_path)
        self.ann_path = Path(ann_path)
        
        assert 0 < scale <= 1, "Scale은 0과 1 사잇값이어야 합니다."
        self.scale = scale
        self.ann_suffix = ann_suffix
        
        self.ids = [splitext(file)[0] for file in listdir(img_path) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f"{img_path}\n 위 경로를 찾을 수 없습니다.")
        logging.info(f"Creating dataset with {len(self.ids)} examples")
        
    def __len__(self):
        return len(self.ids)
    
    @classmethod
    def preprocess(cls, pil_img, scale, is_ann):
        w, h = pil_img.size
        trs_w, trs_h = int(scale*w), int(scale*h)
        
        assert trs_w>0 and trs_h>0, "스케일이 너무 작습니다. 다시 조정해 주세요."
        pil_img = pil_img.resize((trs_w, trs_h), resample=Image.NEAREST if is_ann else Image.BICUBIC)
        
        """
        여기서부터 이해가 안됨
        """
        img_ndarray = np.asarray(pil_img)                   # 이미지 arr 생성
        
        if img_ndarray.ndim==2 and not is_ann:              # 이미지 arr 차원수==2 and "ann"이 아닐 때
            img_ndarray = img_ndarray[np.newaxis, ...]      # 이미지 arr = 이미지 arr에서 np.newaxis부터의 arr
            # --> 3차원으로 변환해 주는 것 같다 (np.newaxis는 차원을 변환할 때 사용됨)
            
        elif not is_ann:                                    # "ann"이 아닐 때
            img_ndarray = img_ndarray.transpose((2, 0, 1))  # 3차원이니깐 0,1,2 -> 2,0,1 순서로 변환
            # --> 정확히는 모르겠지만 (H,W,C) -> (C,H,W) 인 것 같기도,,? 나중에 정확히 확인
            
        # if not is_ann:
        img_ndarray = img_ndarray/255
            
        
        return img_ndarray
    
    @classmethod
    def load(cls, filename):                # 데이터 로드  함수
        ext = splitext(filename)[1]         # 확장자를 확인하는 라인
        if ext in ['.npz', '.npy']:         # .npz, .npy로 끝나면 numpy로 받아옴
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:        # .pt, .pth로 끝나면 torch로 받아와서 numpy로 변환                         
            return Image.fromarray(torch.load(filename).numpy())
        
        # json 데이터를 사전에 처리하지 않고 로딩하면서 처리하기
        elif ext in ['.json']:
            with open(filename,"r",encoding='utf8') as f:
                contents = f.read()
                json_data = json.loads(contents)
            
            str_fname = str(filename)
            img_pth = Path(str_fname.replace("Annotations","Images").replace("_PLINE.json",".png"))            
            load_img = np.array(Image.open(img_pth))    
            lbl = np.zeros((load_img.shape[0], load_img.shape[1]), np.uint8)
        
            for idx in range(len(json_data["annotations"])):
                temp = np.array(json_data["annotations"][idx]["polyline"]).reshape(-1)
                
                assert type(temp) != None, f"Convert .json Error: {ext}\t {list(temp)}"
                
                temp_int = np.apply_along_axis(np.int32, arr=temp, axis=0)
                temp_re = temp_int.reshape(-1, 2)
                
                lbl = cv2.polylines(img=lbl,
                            pts=[temp_re],
                            isClosed=False,
                            color=(255),
                            thickness=5)
            return Image.fromarray(lbl)
        else:
            return Image.open(filename)     # 이외의 확장자는 그냥 가져옴

    def __getitem__(self,idx):
        name = self.ids[idx]
        # ann_file = list(self.ann_path.glob(name+self.ann_suffix+'.png'))
        # img_file = list(self.img_path.glob(name+".png"))
        
        ann_file = list(self.ann_path.glob(name+self.ann_suffix+'.*'))
        img_file = list(self.img_path.glob(name+".*"))
        
        assert len(ann_file)==1, f"어노테이션이 없거나, 여러 개의 ID를 지녔습니다. {name}: {ann_file}"
        assert len(img_file)==1, f"이미지가 없거나, 여러 개의 ID를 지녔습니다. {name}: {img_file}"
        
        annot = self.load(ann_file[0])
        img = self.load(img_file[0])
        
        assert img.size == annot.size, \
            f'이미지와 어노테이션의 {name}은 반드시 같아야 합니다, 지금은 이미지 {img.size} 와 어노테이션 {annot.size}로 다릅니다.'
            
        img = self.preprocess(img, self.scale, is_ann=False)
        annot = self.preprocess(annot, self.scale, is_ann=True)
        
        return {
            "image" : torch.as_tensor(img.copy()).float().contiguous(),
            "mask" : torch.as_tensor(annot.copy()).long().contiguous()
        }
        
class CrackDataset(BaseDataset):
    def __init__(self, img_path, ann_path, scale=1):
        super().__init__(img_path, ann_path, scale, ann_suffix='_PLINE')