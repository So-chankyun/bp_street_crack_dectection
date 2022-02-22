import argparse
<<<<<<< HEAD
import cv2
import torch
=======
import os
import cv2
import torch
import datetime
>>>>>>> b21dbb31461ff9a9454c5184dcdb918fd063a3c7
import numpy as np

from PIL import Image
from unet import UNet

"""
<<<<<<< HEAD
########### 모델 아이디어 및 테스트는 test_unet/mak_video_detect.ipynb 에서 확인할 수 있음 ###########
=======
########### 초기 모델 아이디어 및 테스트는 test_unet/mak_video_detect.ipynb 에서 확인할 수 있음 ###########
>>>>>>> b21dbb31461ff9a9454c5184dcdb918fd063a3c7

전처리 로직이 다르다면, preprocess 메서드 수정 필요
"""
def preprocess(img, scale):
    pil_img = Image.fromarray(img)
    trs_w, trs_h = int(1280*scale), int(720*scale)
    pil_img = pil_img.resize((trs_w, trs_h), resample=Image.BICUBIC)
    
    img_ndarray = np.asarray(pil_img)
    img_ndarray = img_ndarray.transpose((2, 0, 1))
    img_ndarray = img_ndarray[np.newaxis, ...]  
    img_ndarray = img_ndarray/255
    
    return torch.as_tensor(img_ndarray.copy())

def pred_frame(frame, net, DEV):
    images = preprocess(frame, .5).to(device=DEV, dtype=torch.float32)
    masks_pred = net(images)
    last_im = torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()
    return last_im.numpy()

def merge_img(frame, pred):
    re_frm = cv2.resize(frame, dsize=(640, 360), interpolation=cv2.INTER_CUBIC)
    add_mask = re_frm.copy()
    add_mask[pred[:,:]!=0]=[0,255,0]
    return add_mask

def get_args():
    
    """
    parser의 argument 정의
    
        --width :       이미지 너비
        --height :      이미지 높이
        --v_number :    비디오 번호 - input video 파일 명 예) sample_video9.mp4
        --frame :       output video 저장 시 프레임 수
        --model_dir :   모델이 있는 directory 이름 (main code 참조)
        --m_cam :       내장 카메라 사용 여부 (활성화 시 input video 대신 내장 카메라 작동)
        --save :        저장 여부 (활성화 시 save 경로로 저장 - save 경로는 main에서 정의)
<<<<<<< HEAD
        --frame_thred : Crack 비율의 경계 값 (예 : 1.0이면 1.0을 초과하는 frame의 ratio를 red로 표시)
=======
        --crack_thred : Crack 비율의 경계 값 (예 : 1.0이면 1.0을 초과하는 frame의 ratio를 red로 표시)
>>>>>>> b21dbb31461ff9a9454c5184dcdb918fd063a3c7
        
    """
    
    parser = argparse.ArgumentParser(description='Video Inference Learning Parameters')
    parser.add_argument('--width', '-W', type=int, default=640, help='Model Input Width')
    parser.add_argument('--height', '-H', type=int, default=360, help='Model Input Height')
    parser.add_argument('--v_number', '-vn', type=int, default=9, help='The Number of Road Video')
    parser.add_argument('--frame', '-fps', type=float, default=15.0, help='Output video Frame')
    parser.add_argument('--model_name', '-mn', type=str, default='UNet_b2th5dn200k', help='Input Model Name')
    parser.add_argument('--m_cam', action='store_true', default=False, help='Use Device Camera')
    parser.add_argument('--save', action='store_true', default=False, help='Save Video option')
    parser.add_argument('--crack_thred', '-crth', type=float, default=100, help='Frame Threshold Ratio')
    parser.add_argument('--bilinear', action='store_true', default=False, help='UNet use bilinear upscaling?')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    PATH = r"C:\Users\yunjc\_python_jupyter\bupyeonggu\bp_road_crack_detection\1_모델링\unet_result_pth\!model.pth"
    MODEL_PATH = PATH.replace("!model", args.model_name)
    INPUT_PATH = f"D:/data/sample/sample_video{args.v_number}.mp4"
    
    """
    초기 directory 구성

        D:/data (root를 지정하기 위해서 변경 필요)
            ├─ sample
            └─ output
                ├─ capture
                └─ video
    """
    
    try:
        os.makedirs(f'D:/data/output/{args.model_name}/vn{args.v_number}')
        SAVE_PATH = f'D:/data/output/{args.model_name}/vn{args.v_number}/{args.model_name}_vn{args.v_number}.avi'
    except:
        SAVE_PATH = f'D:/data/output/{args.model_name}/vn{args.v_number}/{args.model_name}_vn{args.v_number}.avi'
    
    try:
        os.makedirs(f'D:/data/output/{args.model_name}/vn{args.v_number}/capture/')
        CAP_PATH = f'D:/data/output/{args.model_name}/vn{args.v_number}/capture/'
    except:
        CAP_PATH = f'D:/data/output/{args.model_name}/vn{args.v_number}/capture/'
        
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    model = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device=DEVICE)
    model.eval()
    
    if args.m_cam:
        capture = cv2.VideoCapture(0)
    else:
        capture = cv2.VideoCapture(INPUT_PATH)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("재생할 파일 너비, 높이 : %d, %d"%(args.width, args.height))

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    if args.save:
        out = cv2.VideoWriter(SAVE_PATH, fourcc, args.frame, (int(args.width), int(args.height)))

    count = 0
    max_ratio, avg_ratio = 0.0, 0.0
    
    
    while cv2.waitKey(33) < 0:
        count+=1
        full_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        ret, frame = capture.read()
        if not ret:
            print("프레임을 수신할 수 없습니다. 종료 중 ...")
            break

        pred_img = pred_frame(frame, model, DEVICE)
        convert_img = merge_img(frame, pred_img)
        
        CRPF = np.round((pred_img.reshape(-1).sum()/(args.width*args.height))*100, 2)
        max_ratio = CRPF if CRPF > max_ratio else max_ratio
        avg_ratio = (avg_ratio*(count-1)+CRPF)/count
            
        """
        Text 표현
        
            CRPF :      지금 이미지에서 Crack 픽셀의 비율 이미지 (%)
            MAX_img :   현재 진행 상태에서 가장 큰 Crack 픽셀 비율 이미지 (%)
            AVG_img :   현재 진행 상태까지 평균 Crack 픽셀 비율 이미지 (%)
            PRG_rate :  현재 진행 비율 이미지 (%)
            
        """
        font=cv2.FONT_HERSHEY_SIMPLEX
        color = (50,50,165) if CRPF > args.crack_thred else (50,165,50)
        
        cv2.putText(convert_img, 'Crt Rate : {:.2f} {}'.format(CRPF, "%"), (5, 20), font, .6, color, 2)
        cv2.putText(convert_img, 'Max Rate : {:.2f} {}'.format(max_ratio, "%"), (5, 40), font, .6, (60,180,255), 2)
        cv2.putText(convert_img, 'Avg Rate : {:.2f} {}'.format(avg_ratio, "%"), (5, 60), font, .6, (60,180,255), 2)
        cv2.putText(convert_img, 
                    'Progress Rate : {:.1f} {}'.format((count/full_frame)*100, "%"), 
                    (5, 80), font, .6, (60,180,255), 2)
        
        # args.crack_thred 이상이 됐을 때 캡처
        if args.save and (CRPF > args.crack_thred):
            now = datetime.datetime.now().strftime("%d_%H-%M-%S")
            outfile = CAP_PATH + f"sample_video{args.v_number}_{str(now)}.jpg"
            cv2.imwrite(outfile, convert_img)
            
        cv2.imshow("Test Vidoe Crack Detect", convert_img)
        if args.save:
            out.write(convert_img)

    capture.release()
    if args.save:
        out.release()
    cv2.destroyAllWindows()