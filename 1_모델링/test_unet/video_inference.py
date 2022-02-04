import argparse
import cv2
import torch
import numpy as np

from PIL import Image
from unet import UNet

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
    parser = argparse.ArgumentParser(description='Video Inference Learning Parameters')
    parser.add_argument('--width', '-w', metavar='W', type=int, default=640, help='Model Input Width')
    parser.add_argument('--height', '-h', metavar='H', type=int, default=360, help='Model Input Height')
    parser.add_argument('--v_number', '-vn', type=int, default=9, help='The Number of Road Video')
    parser.add_argument('--m_cam', action='store_true', default=False, help='Use Device Camera')
    parser.add_argument('--model_dir', '-md', type=str, default='UNet_b2th10dn50000', help='Input checkpoint Directory')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    PATH = r"C:\Users\yunjc\_python_jupyter\bupyeonggu\bp_road_crack_detection\1_모델링\unet_result_pth\!dir\checkpoint_epoch5.pth"
    MODEL_PATH = PATH.replace("!dir", args.model_dir)
    INPUT_PATH = f"D:/data/sample_video{args.v_num}.mp4"
    SAVE_PATH = f'D:/data/UNet_b2th10dn50000/output{args.v_num}.avi'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = UNet(n_channels=3, n_classes=2, bilinear=False)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device=DEVICE)
    model.eval()
    
    if args.m_cam == False:
        capture = cv2.VideoCapture(INPUT_PATH)
    else:
        capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("재생할 파일 넓이, 높이 : %d, %d"%(args.width, args.height))

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(SAVE_PATH, fourcc, 30.0, (int(args.width), int(args.height)))

    while cv2.waitKey(33) < 0:
        ret, frame = capture.read()
        if not ret:
            print("프레임을 수신할 수 없습니다. 종료 중 ...")
            break

        convert_img = merge_img(frame, pred_frame(frame, model, DEVICE))
        cv2.imshow("Test Vidoe Crack Detect", convert_img)
        out.write(convert_img)


    capture.release()
    out.release()
    cv2.destroyAllWindows()