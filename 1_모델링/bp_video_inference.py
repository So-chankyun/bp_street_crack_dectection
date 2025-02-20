import argparse
import os
import cv2
import torch
# import datetime
import numpy as np

from tqdm import tqdm
from PIL import Image
from UNet.unet import UNet
# from PAN.pan import 

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
    parser.add_argument('--width', '-W', type=int, default=640, help='Model Input Width')
    parser.add_argument('--height', '-H', type=int, default=360, help='Model Input Height')
    parser.add_argument('--frame', '-fps', type=float, default=30.0, help='Output video Frame')
    parser.add_argument('--save', action='store_true', default=False, help='Save Video option')
    parser.add_argument('--crack_thred', '-crth', type=float, default=100, help='Frame Threshold Ratio')
    return parser.parse_args()

if __name__ == '__main__':
    MODELS_PATH = './models/'
    VIDEOS = "./input/"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    args = get_args()
    for m in os.listdir(MODELS_PATH):
        print(f'============{m.replace(".pth","")}============')
        
        MODEL_PATH = MODELS_PATH+m
        
        if "bilinear" in m:
            print(" BILINEAR TRUE ")
            model = UNet(n_channels=3, n_classes=2, bilinear=True)
        else:
            print(" BILINEAR False ")
            model = UNet(n_channels=3, n_classes=2, bilinear=False)
            
        model.load_state_dict(torch.load(MODEL_PATH))
        model.to(device=DEVICE)
        model.eval()
        

    
        for video in tqdm(os.listdir(VIDEOS), total=len(os.listdir(VIDEOS))):
            try:
                os.makedirs(f'./output/{m.replace(".pth","")}/{video.replace(".mp4","")}')
                SAVE_PATH = f'./output/{m.replace(".pth","")}/{video.replace(".mp4","")}/{video.replace(".mp4","")}_out.avi'
            except:
                SAVE_PATH = f'./output/{m.replace(".pth","")}/{video.replace(".mp4","")}/{video.replace(".mp4","")}_out.avi'
            
            try:
                os.makedirs(f'./output/{m.replace(".pth","")}/{video.replace(".mp4","")}/capture/')
                CAP_PATH = f'./output/{m.replace(".pth","")}/{video.replace(".mp4","")}/capture/'
            except:
                CAP_PATH = f'./output/{m.replace(".pth","")}/{video.replace(".mp4","")}/capture/'  

            capture = cv2.VideoCapture(VIDEOS+video)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            if args.save:
                out = cv2.VideoWriter(SAVE_PATH, fourcc, args.frame, (int(args.width), int(args.height)))

            count, img_idx = 0, 0
            max_ratio, avg_ratio = 0.0, 0.0
            
            
            while cv2.waitKey(33) < 0:
                count+=1
                full_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                ret, frame = capture.read()
                if not ret:
                    break

                pred_img = pred_frame(frame, model, DEVICE)
                convert_img = merge_img(frame, pred_img)
                
                CRPF = np.round((pred_img.reshape(-1).sum()/(args.width*args.height))*100, 2)
                max_ratio = CRPF if CRPF > max_ratio else max_ratio
                avg_ratio = (avg_ratio*(count-1)+CRPF)/count
                    
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
                    # now = datetime.datetime.now().strftime("%d_%H-%M-%S")
                    outfile = CAP_PATH + f'{video.replace(".mp4","")}_{img_idx}.jpg'
                    img_idx+=1
                    cv2.imwrite(outfile, convert_img)
                    
                cv2.imshow("Test Vidoe Crack Detect", convert_img)
                if args.save:
                    out.write(convert_img)

            capture.release()
            if args.save:
                out.release()
            cv2.destroyAllWindows()