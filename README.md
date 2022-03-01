# bp_road_crack_dectection

- [Data](#Data)
- [Train](#Train)
- [Video Inference](#Video_inference)
- [Output](#Output)

## Setting

해당 프로젝트를 수행하면서 사용된 하드 웨어 환경은 다음과 같다.

- **SET1**
    | HW/Python | Version                       | Model                  |
    | :-------: | :---------------------------- | :--------------------: |
    | CPU       | Intel 12th i5-12600kf         | UNet(50k data).pth     |
    | GPU       | Nvidia RTX 3060 D6 12GB       |                       ^|
    | RAM       | Samsung DDR4-25600 32GB(Dual) |                       ^|
    | SSD       | Google Drive Local Connect    |                       ^|
    | Python    | 3.7.11                        |                       ^|
    | CUDA      | 11.2                          |                       ^|

- **SET2**
    | HW/Python | Version                       |
    | :-------: | :---------------------------- |
    | CPU       | Intel 12th i5-12600k          |
    | GPU       | Nvidia RTX 3070Ti D6x 8GB     |
    | RAM       | Samsung DDR4-25600 32GB(Dual) |
    | SSD       | SK Hynix P31 Gold (PCIe 3.0)  |
    | Python    | 3.8.8                         |
    | CUDA      | 11.6                          |

- **SET3**
    | HW/Python | Version                           |
    | :-------: | :-------------------------------- |
    | CPU       | Intel 11th i7-11800h              |
    | GPU       | Nvidia RTX 3060 D6 Laptop GPU 6GB |
    | RAM       | Samsung DDR4-25600 32GB(Dual)     |
    | SSD       | SK Hynix P31 Gold (PCIe 3.0)      |
    | Python    | 3.8.8                             |
    | CUDA      | 10.2                              |

## Data

활용 데이터는 [Ai Hub의 도로장애물/표면 인지 영상(수도권)](https://aihub.or.kr/aidata/34111) 데이터를 활용하였고, 구체적인 데이터 소개는 링크를 통해 확인할 수 있다.

---

## Train



---

## Video Inference

추출된 모델 형태를 활용하여, video 형태에 기능을 입히는 단계이다. input은 `.mp4` format이 사용되고 output으로 `.avi` 파일과 Crack Pixel Ratio가 지정한 Threshold를 넘는 Frame의 캡처 파일을 `.avi`로 반환하게 된다.

---

## Output



---