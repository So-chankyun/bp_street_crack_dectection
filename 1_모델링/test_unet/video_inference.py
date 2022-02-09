import argparse
import cv2
import torch
import numpy as np

from PIL import Image
from unet import UNet

"""
--- 모델 아이디어 및 테스트는 test_unet/mak_video_detect.ipynb 에서 확인할 수 있음 ---

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
    
        --width : 이미지 너비
        --height : 이미지 높이
        --v_number : 비디오 번호 - input video 파일 명 예) sample_video9.mp4
        --frame : output video 저장 시 프레임 수
        --model_dir : 모델이 있는 directory 이름 (main code 참조)
        --m_cam : 내장 카메라 사용 여부 (활성화 시 input video 대신 내장 카메라 작동)
        --save : 저장 여부 (활성화 시 save 경로로 저장 - save 경로는 main에서 정의)
        --frame_thred : Crack 비율의 경계 값 (예 : 1.0이면 1.0을 초과하는 frame의 ratio를 red로 표시)
        
    """
    
    parser = argparse.ArgumentParser(description='Video Inference Learning Parameters')
    parser.add_argument('--width', '-W', type=int, default=640, help='Model Input Width')
    parser.add_argument('--height', '-H', type=int, default=360, help='Model Input Height')
    parser.add_argument('--v_number', '-vn', type=int, default=9, help='The Number of Road Video')
    parser.add_argument('--frame', '-fps', type=float, default=30.0, help='Output video Frame')
    parser.add_argument('--model_dir', '-md', type=str, default='UNet_b2th5dn200k', help='Input checkpoint Directory')
    parser.add_argument('--m_cam', action='store_true', default=False, help='Use Device Camera')
    parser.add_argument('--save', action='store_true', default=False, help='Save Video option')
    parser.add_argument('--frame_thred', '-fth', type=float, default=100, help='Frame Threshold Ratio')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    PATH = r"C:\Users\yunjc\_python_jupyter\bupyeonggu\bp_road_crack_detection\1_모델링\unet_result_pth\!dir\checkpoint_epoch5.pth"
    MODEL_PATH = PATH.replace("!dir", args.model_dir)
    INPUT_PATH = f"D:/data/sample_video{args.v_number}.mp4"
    SAVE_PATH = f'D:/data/{args.model_dir}/output{args.v_number}.avi'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = UNet(n_channels=3, n_classes=2, bilinear=False)
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
    out = cv2.VideoWriter(SAVE_PATH, fourcc, args.frame, (int(args.width), int(args.height)))

    while cv2.waitKey(33) < 0:
        ret, frame = capture.read()
        if not ret:
            print("프레임을 수신할 수 없습니다. 종료 중 ...")
            break

        pred_img = pred_frame(frame, model, DEVICE)
        convert_img = merge_img(frame, pred_img)
        CRPF = np.round((pred_img.reshape(-1).sum()/(args.width*args.height))*100, 2)
        
        font=cv2.FONT_HERSHEY_SIMPLEX
        color = (50,50,165) if CRPF > args.frame_thred else (50,165,50)
        CRPF_img = cv2.putText(convert_img, 'Crack Ratio : {:.2f} {}'.format(CRPF, "%"), (5, 20), font, .6, color, 2)

        cv2.imshow("Test Vidoe Crack Detect", CRPF_img)
        if args.save:
            out.write(CRPF_img)

    capture.release()
    if args.save:
        out.release()
    cv2.destroyAllWindows()