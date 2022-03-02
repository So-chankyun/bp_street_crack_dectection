# bp_road_crack_dectection

- [Setting](#setting)
    - [Hardware and Python](#hardware-and-python)
    - [필수 library](#%ED%95%84%EC%88%98-library)
- [Data](#data)
- [Train](#train)
    - [Wandb](#wandb)
    - [Data Path Setting](#data-path-setting)
- [Video Inference](#video-inference)
    - [How to Use](#how-to-use)
- [Output](#output)

<span style="color:RED"> !!! `README.md` 미완성 !!! </span>

## Setting

### Hardware and Python
해당 프로젝트를 수행하면서 사용된 하드 웨어 및 Python 환경은 다음과 같다. (구체적인 `lib` 구성은 `requirement.txt`를 참조 - 필수 `lib`은 별도 기재)

- **SET1** : Model - `UNet(50k data).pth`
    | HW/Python | Version                       |
    | :-------: | :---------------------------- |
    | CPU       | Intel 12th i5-12600kf         |
    | GPU       | Nvidia RTX 3060 D6 12GB       |
    | RAM       | Samsung DDR4-25600 32GB(Dual) |
    | SSD       | Google Drive Local Connect    |
    | Python    | 3.7.11                        |
    | CUDA      | 11.2                          |

- **SET2** : Model - `UNet_b6th5dn210k_bilinear.pth`
    | HW/Python | Version                       |
    | :-------: | :---------------------------- |
    | CPU       | Intel 12th i5-12600k          |
    | GPU       | Nvidia RTX 3070Ti D6x 8GB     |
    | RAM       | Samsung DDR4-25600 32GB(Dual) |
    | SSD       | SK Hynix P31 Gold (PCIe 3.0)  |
    | Python    | 3.8.8                         |
    | CUDA      | 11.6                          |

- **SET3** : Model - `UNet_b2th5dn200k.pth`
    | HW/Python | Version                           |
    | :-------: | :-------------------------------- |
    | CPU       | Intel 11th i7-11800h              |
    | GPU       | Nvidia RTX 3060 D6 Laptop GPU 6GB |
    | RAM       | Samsung DDR4-25600 32GB(Dual)     |
    | SSD       | SK Hynix P31 Gold (PCIe 3.0)      |
    | Python    | 3.8.8                             |
    | CUDA      | 10.2                              |


> 현재 github에는 **SET2**로 작성된 `UNet_b6th5dn210k_bilinear.pth` 를 예로 활용한다.

### 필수 library

```TEXT
argparse
os
cv2
torch (+ CUDA, cuDNN)
numpy
tqdm
PIL
```

## Data

활용 데이터는 [Ai Hub의 도로장애물/표면 인지 영상(수도권)](https://aihub.or.kr/aidata/34111) 데이터를 활용하였고, 구체적인 데이터 소개는 링크를 통해 확인할 수 있다. 제공되는 데이터 중 모델 구축에 활용한 데이터는 다음과 같다.

- **활용 데이터** 
    - Annotations : `/도로장애물·표면 인지 영상(수도권)/Training/Annotations/CRACK`
    - Images : `/도로장애물·표면 인지 영상(수도권)/Training/Images/CRACK`
- **제외한 데이터** :  `C_Mainroad_D01`, `C_Mainroad_D02`, `C_Mainroad_D03`

## Train

### Wandb

Training을 위해서 우선 `wandb`를 install하고 계정에 로그인하는 것을 추천한다. wandb를 활용한다면 별도의 evaluation과 validation에 대한 시각화 라인을 줄일 수 있다.(web에서 트래킹하는 것이니 네트워크 환경이 필요하다.) `wandb`에 관한 사항은 [여기](https://docs.wandb.ai/quickstart)를 통해 세팅할 수 있다.

### Data Path Setting

`train.py`를 원활히 실행하기 위해서 데이터 경로를 세팅한다. `train.py`의 상단부에 데이터 경로를 정의하는 코드 라인을 수정한다.

1. Ai Hub를 통해 받은 데이터에서 [Data](#data)를 참고하여 `제외한 데이터`를 제외한 폴더의 모든 압축을 풀고, 다음의 경로로 바꾸어준다.
    ```text
    images:     /도로장애물·표면 인지 영상(수도권)/Training/Images/CRACK/images
    annotations: /도로장애물·표면 인지 영상(수도권)/Training/Annotations/CRACK/annotations
    ```
2. `train.py`의 상단 코드라인을 수정한다.
    ```python
    DATAPATH = "<다운로드한 경로>/도로장애물·표면 인지 영상(수도권)/Training/!CHANGE/CRACK/!changes/"
    ```
    Data 경로를 위와 다른 경로로 설정한다면, 코드라인 하위의 images와 annotations의 경로를 지정하는 부분도 수정해야 한다.

## Video Inference

추출된 모델 형태를 활용하여, video 형태에 기능을 입히는 단계이다. input은 `.mp4` format이 사용되고 output으로 `.avi` 파일과 Crack Pixel Ratio가 지정한 Threshold를 넘는 Frame의 캡처 파일을 `.avi`로 반환하게 된다.

### How to Use

1. cmd 혹은 anaconda prompt 를 연다.(필요에 따라 가상환경 활성화)
2. git을 clone한 위치로 이동하고, `1_모델링` directory로 이동한다.
    ```cmd
    cd <git을 clone한 위치>/1_모델링
    ```
3. 검출할 영상을 `.mp4`의 형태로 `1_모델링/input/`으로 모두 옮긴다.
4. `bp_video_inference.py`를 옵션에 맞춰 실행한다
    ```cmd
    python bp_video_inference.py --save -crth 5
    ```
    - 해당 옵션은 smaple을 통한 output을 저장하고(--save)
    - Crack Pixel Ratio에 따른 캡처 기준을 5%(-crth 5)로 설정하여 실행한다는 의미
5. `1_모델링/output/`에 결과물 확인
    - 반환되는 결과물은 다음과 같음

    > ⅰ. *(Crack이 차지하는 Pixel 수)\*100 / (전체 픽셀 수)* 가 5% 이상인 Frame 이미지 캡처  
    > ⅱ. Crack Detection이 포함된 `.avi` 비디오 파일

## Output

### Video Output

![Video_Out](https://github.com/chaaaning/bp_road_crack_detection/blob/main/_imgs/video_out.gif?raw=true)