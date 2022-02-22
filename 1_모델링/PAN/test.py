from pathlib import Path
import os
import pandas as pd
import logging
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

from utils.dataloading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from unet import UNet
from pan.networks import PAN
import utils.ss_transforms as tr
from utils.dice_score import multiclass_dice_coeff, dice_coeff

def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

def test(model, data_loader, device):
    '''
    데이터를 받아서 prediction하고 결과를 출력한다.
    '''

    num_val_batches = len(data_loader)
    dice_score = 0
    loss = 0
    global_step = 0

    model.eval()

    # loss함수 및 모델에 필요한 생성자 초기화
    criterion = nn.CrossEntropyLoss()

    for batch in tqdm(data_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=True):
        images = batch['image']
        true_masks = batch['mask']

        # 결과이미지를 저장하는 프로세스는 나중에 추가하도록 하자.

        with torch.no_grad():
            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)
            true_masks = true_masks.long().squeeze(1)

            # 모델에 넣어서 예측
            pred_masks = model(images)

            # loss 계산
            loss += criterion(pred_masks, true_masks) + dice_loss(F.softmax(pred_masks, dim=1).float()
                                                                     , F.one_hot(true_masks, 2).permute(0, 3, 1, 2).float())

            # 예측 마스크 channel에서 값이 가장 큰 원소의 index를 가져옴. 이후 tensor형태로 차원변경
            pred_masks = F.one_hot(pred_masks.argmax(dim=1),2).permute(0,3,1,2).float()

            dice_score += multiclass_dice_coeff(pred_masks[:,1:,...]
                                                ,F.one_hot(true_masks, 2).permute(0,3,1,2).float()[:,1:,...],
                                                reduce_batch_first=False)

            # true_masks[0].float().cpu()


    if num_val_batches == 0:
        return {'Avg Loss': loss, 'Dice Score':dice_score/num_val_batches}
    return {'Avg Loss':loss/num_val_batches,'Dice Score':dice_score/num_val_batches}

if __name__ == '__main__':

    # 1. 경로 지정
    DATAPATH = "D:/crack data/도로장애물·표면 인지 영상(수도권)/Training/!CHANGE/CRACK/!changes/"
    dir_img = Path(DATAPATH.replace("!CHANGE", "Images").replace("!changes", "data"))
    dir_mask = Path(DATAPATH.replace("!CHANGE", "Annotations").replace("!changes", "data"))

    # 2. define transformer
    transform = transforms.Compose([tr.RescaleSized((640, 384)),
                                          tr.MinMax(255.0),
                                          tr.ToTensor()
                                          ])

    # 모델의 리스트를 출력해준다.
    print(10*'-'+' MODEL LIST '+'-'*10+'\n')

    MODEL_PATH = './model file'
    model_file_list = [name for name in os.listdir(MODEL_PATH) if name.endswith('.pth')]

    for idx, name in enumerate(model_file_list):
        print(f'{idx}. {name}')

    print('\n'+32 * '-'+'\n')

    # 모델의 번호를 입력받는다.
    model_index = input('Select model file(you have to input number) : ').split(' ')

    # 모델의 번호를 int형으로 변환한다.
    model_index = list(map(int, model_index))

    # 사용가능한 device 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_result = []

    # 3. Create dataset and Loading
    try:
        data_set = CarvanaDataset(dir_img, dir_mask, data_type='test', transform=transform)
    except (AssertionError, RuntimeError):
        data_set = BasicDataset(dir_img, dir_mask, data_type='test', transform=transform)

    for idx in model_index:
        file_name = model_file_list[idx]

        if file_name.startswith('PAN'):
            batch_size = 64
            model = PAN(backbone='resnet50', pretrained=True, n_class=2)
        elif file_name.startswith('Unet'):
            batch_size = 8
            model = UNet(n_channels=3, n_classes=2, bilinear=True)
        else:
            print('Nothing to use model file')
            break

        # load model
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, file_name)))
        model.to(device=device)

        # load data
        loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
        data_loader = DataLoader(data_set, **loader_args)

        # test
        print('Now Testing {}. {}'.format(idx, file_name))
        STORE_PATH = './result image/'+file_name

        # if not os.direxists(STORE_PATH):
        #     os.mkdir(STORE_PATH)
        result = test(model, data_loader, device) # 결과 저장. Avg Loss, Dice Score
        result['model'] = file_name.split('.')[0] # 모델명만 저장.

        # store result
        total_result.append(result)

        '''
        data를 테스트한 결과를 외부로 export하는 코드는 나중에 작성하도록 하자.
        '''

    # print result
    for idx, result_dict in enumerate(total_result):
        print(str(idx)+'. Model : {}'.format(result_dict['model']))
        print('    Avg Loss : {}'.format(result_dict['Avg Loss']))
        print('    Dice Score : {}\n'.format(result_dict['Dice Score']))

    # total_result.to_csv('model result.csv',index=False)








