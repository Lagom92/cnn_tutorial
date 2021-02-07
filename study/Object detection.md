# Object Detection

# 의료용 마스크 탐지 모델 구축

## 내용 정리



<br/>

## 1. 객체 탐지 소개

객체 탐지(Object Detection)는 컴퓨터 비전 기술의 세부 분야 중 하나로써 주어진 이미지내 사용자가 관심 있는 객체를 담지하는 기술이다.



<br/>

### 1.1. 바운딩 박스

특정 사물을 탐지하여 모델을 효율적으로 학습 할 수 있도록 도움을 주는 방법

객체 탐지 모델에서 타겟의 위치를 특정하기 위해서 사용된다.

타겟 위치를 x와 y축을 이용하여 사각형으로 표현한다.

(x 최소값, y 최소값, x 최대값, y 최대값)

<br/>

![](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/bc2.PNG)

<br/>

x와 y 값은 픽셀값으로 효율적인 연산을 위해서는 최대값을 1로 변환해줘야 한다.



<br/>

### 1.2. 모델 형태

객체 탐지 모델은 크게 one-stage 모델과 two-stage 모델로 구분할 수 있다.

<br/>

![](https://github.com/Pseudo-Lab/Tutorial-Book/blob/master/book/pics/ch1img04.PNG?raw=true)

<br/>

**Classification과 Region Proposal의 개념 이해**

Classification: 특정 물체에 대해 어떤 물체인지 분류

Region Proposal: 물체가 있을만한 영역을 빠르게 찾아내는 알고리즘



<br/>

#### 1.2.1. One-Stage Detector

One-Stage Detector는 Classificatio. Regional Proposal을 동시에 수행하여 결과를 얻는 방법이다.

이미지를 모델에 입력 후, Conv Layer를 사용하여 이미지 특징을 추출

<br/>

![](https://drive.google.com/uc?id=1850eKsb59NtgQEcM0fXji7cD13TEsDo3)



<br/>

#### 1.2.2. Two-Stage Detector

Two-Stage Detector는 Classification, Region Proposal을 순차적으로 수행하여 결과를 얻는 방법이다.

Region Porposal과 Classification을 순차적으로 실행하는 것을 알 수 있다.

<br/>

![](https://drive.google.com/uc?id=1qACO-vEahSiz2Zb5jmAW1jbw94g1ThZO)



<br/>

### 1.3. 모델 구조

One-Stage Detector: YOLO, SSD, RetinaNet, etc

Twor-Stage Detector: R-CNN, Fast R-CNN, Faster R-CNN, etc



<br/>

#### 1.3.1. R-CNN

![](https://drive.google.com/uc?id=1qCmgiqH45lkpzADBk3zh29RFPPrRVKC4)



<br/>

R-CNN은 Selective Search를 이용해 이미지에 대한 후보영역(Region Porposal)을 생성한다.

성성된 각 후보영역을 고정된 크기로 wrapping하여 CNN의 input으로 사용한다.

CNN에서 나온 Feature map으로 SVM을 통해 분류, Regressor을 통해 Bounding box를 조정한다.

강제로 크기를 맞추기 위한 wrapping으로 이미지의 변형이나 손실이 일어나고 후보영역 만큼 CNN을 돌려야하기 때문에 큰 저장공간을 요구하고 느리다는 단점이 있다.



<br/>

#### 1.3.2. Fast R-CNN

![](https://drive.google.com/uc?export=view&id=1zKLOeKylk2SjRMQmfJ3ZJdaJSs0q1fIw)



<br/>

각 후보영역에 CNN을 적용하는 R-CNN과 달리 이미지 전체에 CNN을 적용하여 생성된 Feature map에서 후보영역을 생성한다.

생성된 후보영역은 ROI Pooling을 통해 고정 사이즈의 Feature vector로 추출한다.

Feature vector에 FC layer를 거쳐 Softmax를 통해 분류, Regressor를 통해 Bounding box를 조정한다.



<br/>

#### 1.3.3. Faster R-CNN

![](https://drive.google.com/uc?export=view&id=1O5sRVhjcVR8J8zFDhVNwWHmEo069j876)



![](https://drive.google.com/uc?export=view&id=18PW63VbzIdODeCRGSW0G9UF78Zmd5QY0)



<br/>

Selective Search 부분을 딥러닝으로 바꾼 Region Proposal Network(RPN)을 사용한다.

RPN은 Feature map에서 CNN 연산시 Sliding-window가 찍은 지점마다 Anchor-box로 후보영역을 예측한다.

Anchor-box란 미리 지정해놓은 여러개의 비유로가 크기의 Bounding box이다.

RPN에서 얻은 후보영역을 IoU순으로 정렬하여 Non-Maximum Suppression(NMS) 알고리즘을 통해 최종 후보영역을 선택한다.

선택된 후보영역의 크기를 맞추기 위해 RoI Pooling을 거치고 이후 Fast R-CNN과 동일하게 진행된다.



<br/>

#### 1.3.4. YOLO

![](https://drive.google.com/uc?export=view&id=1mVIx26ewRcNab82TGkFj1ODnvPSoQjoD)



<br/>

Bounding box와 Class probability를 하나의 문제로 간주하여 객체의 종류와 위치를 한번에 예측한다.

이미지를 일정 크기의 그리드로 나눠 각 그리드에 대한 Bounding box를 예측한다.

Bounding box의 confidence score와 그리드셀의  class score의 값으로 학습하게 된다.

간단한 처리과정으로 속도가 매우 빠르지만 작은 객체에 대해서는 상대적으로 정확도가 낮다.



<br/>

#### 1.3.5. SSD

![](https://drive.google.com/uc?export=view&id=11kZDrVy5qi4VIAPXfGLmaGV5l_5h68rO)

<br/>

각 Convolution layer 이후에 나오는 Feature map 마다 Bounding box의 class 점수와 Offset(위치좌표)를 구하고, NMS 알고리즘을 통해 최종 Bounding box를 결정한다.

이는 각 Feature map마다 스케일이 다르기 때문에 작은 물체와 큰 물체를 모두 탐지할 수 있다는 장점이 있다.



<br/>

#### 1.3.6. RetinaNet

![](https://github.com/Pseudo-Lab/Tutorial-Book/blob/master/book/pics/ch1img13.PNG?raw=true)



<br/>

RetinaNet은 모델 학습 시 계산하는 손실 함수(Loss function)에 변화를 주어 기존 One-Stage Detector들이 지닌 낮은 성능을 개선했다.

One-Stage Detector는 많게는 십만개 까지의 후보군 제시를 통해 학습을 진행한다.

그 중 실제 객체인것은 일반적으로 10개 이내 이고, 다수의 후보군이 background 클래스로 잡힌다.

상대적으로 분류하기 쉬운 background 후보군들에 대한 loss값을 줄여줌으로써 분류하기 어려운 실제 객체들의 loss 비중을 높이고, 그에 따라 실제 객체들에 대한 학습에 집중하게한다.

RetinaNet은 속도도 빠르면서 Two-Stage Detector와 유사한 성능을 보인다.



<br/><br/>

## 2. 데이터 탐색













<br/>

<br/><br/>

------------

### Reference

- https://pseudo-lab.github.io/Tutorial-Book/chapters/object-detection/Ch1-Object-Detection.html

- https://jdselectron.tistory.com/101
- 

<br/>