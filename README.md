# TensorflowEDA
텐서 플로를 이용한 EDA 심장병 분류 데이터셋

심장마비와 뇌졸중은 심혈관 질환으로 인한 사망 5명 중 4명 이상을 차지합니다. 이 중 3분의 1은 70세 이전에 발생합니다.

전 세계적으로 심혈관 질환(CVDs)은 주요 사망 원인 중 하나입니다. 심혈관 질환에는 관상동맥 질환, 뇌혈관 질환, 류마티스성 심장 질환 및 기타 심장 및 혈관 문제들이 포함됩니다. 세계보건기구에 따르면, 매년 1,790만 명이 심혈관 질환으로 인해 사망합니다. 심장마비와 뇌졸중은 심혈관 질환 사망의 80% 이상을 차지하며, 이 중 3분의 1은 70세 이전에 발생합니다. 심장마비에 기여하는 요인들에 대한 포괄적인 데이터베이스가 구축되었습니다.

주요 목적은 심장마비의 특성 또는 그것에 기여하는 요인을 수집하는 것입니다.
데이터셋의 크기는 1,319개의 샘플로, 아홉 개의 필드로 구성되어 있습니다. 여기서 여덟 개의 필드는 입력 필드이고, 한 개의 필드는 출력 필드입니다. 나이, 성별(여성은 0, 남성은 1), 심박수(impulse), 수축기 혈압(pressurehight), 이완기 혈압(pressurelow), 혈당(glucose), CK-MB (kcm), 그리고 Test-Troponin (troponin)은 입력 필드를 나타냅니다. 출력 필드는 심장마비의 존재 여부(클래스)와 관련이 있으며, 이는 두 가지 카테고리(음성과 양성)로 나뉩니다; 음성은 심장마비가 없음을 나타내며, 양성은 심장마비가 있음을 나타냅니다.

데이터셋 출처 URL:https://www.kaggle.com/datasets/bharath011/heart-disease-classification-dataset

<div id="import">

## 텐서 플로를 활용한 심장병 EDA

이번 텐서 플로를 이용한 모델링 및 최적화 적용 분석을 통해 많은 것을 배울 수 있었고, 이를 활용할 수 있는 수많은 사례에 대한 궁금증이 생겼다. 그리고 코드를 작성하며 부족했던 점이 많았는데 이는 다음과 같다.

### 배울 수 있었던 점:

* 기본적인 모델링: 텐서플로를 사용하여 딥러닝 모델을 어떻게 구축하는지에 대한 기본적인 이해를 얻었다.
* 특성 공학: 데이터의 특성을 잘 이해하고 이를 모델에 어떻게 적용하는지 배웠다.
* 최적화 전략: 다양한 최적화 전략과 방법들을 사용하며, 모델의 성능을 향상시키기 위한 방법을 학습하였다.
* 콜백의 활용: 텐서플로의 콜백 기능을 활용하여 모델 학습 과정 중 원하는 작업을 자동화하고, 모델의 학습 상태를 더욱 효과적으로 모니터링할 수 있었다.

### 활용 가능한 점:

* 다양한 문제 해결: 배운 기술을 다양한 딥러닝 문제에 적용할 수 있다.
* 빠른 프로토타이핑: 텐서플로의 간편한 API를 활용하여 빠르게 모델 프로토타입을 구축하고 테스트할 수 있다.
* 커스텀 모델 및 층 구축: 텐서플로의 유연성을 활용하여 사용자 정의 모델 및 층을 만들 수 있다.
* 전이 학습: 이미 학습된 모델을 활용하여 새로운 문제를 더 빠르게 해결할 수 있다.

### 부족했던 부분:

* 고급 최적화 기법: 몇몇 고급 최적화 전략이나 기법들에 대해 아직까지 깊이 있게 다루지 못했다.
* 과적합 대응: 모델이 과적합될 가능성이 있으며, 이를 해결하기 위한 다양한 전략들에 대한 심도 있는 이해가 필요하다.
* 모델 해석: 모델이 왜 그런 예측을 하는지에 대한 해석능력을 더 발전시켜야 한다.

# 목차

| 번호 | 내용                                             |
|------|--------------------------------------------------|
| 1  | [라이브러리 임포트](#import)                             |
| 2  | [데이터 로드](#load)                                   |
| 3  | [데이터 확인](#check)                   |
| 4  | [데이터 준비](#prepare)       |
|     | [데이터 표준화](#scaling)       |
|     | [성능 향상을 위한 기능 엔지니어링 및 분할](#engineering)       |
| 5  | [모델 학습 준비](#para)                           |
|     | [유틸리티 함수로 모델 구축](#utility)                   |
|     | [유틸리티 함수: 곡선 플로팅](#utility-plot)                   |
|     | [Baseline Model 설정](#Baseline)                           |
|     | [최적화 모델](#opt)                |
|     | [최적화 모델 추출](#opt-extract)                       |
|     | [모델 최적화 적용 후 그래프 ](#opt-extract-active)          |
       
# 1.라이브러리 임포트

</div>

```python
# 표준 라이브러리
import numpy as np # 선형 대수
import pandas as pd # 데이터 처리, CSV 파일 입출력 (예: pd.read_csv)
import warnings # warnings는 경고를 무시
warnings.simplefilter('ignore')
import gc # gc는 메모리 관리

# 학습 이력 시각화
import matplotlib.pyplot as plt
%matplotlib inline

# 텐서플로
import tensorflow as tf  # 텐서플로를 가져옵니다. keras는 딥러닝 모델을 구축하기 위해 사용
from tensorflow import keras
from tqdm.keras import TqdmCallback  # TqdmCallback은 학습 진행 상황을 표시

# 옵튜나 hyperparameters를 최적화
import optuna

# 스케일러
from sklearn.preprocessing import LabelEncoder # LabelEncoder는 레이블을 인코딩
from sklearn.model_selection import train_test_split 
#train_test_split은 데이터를 학습 세트와 테스트 세트로 분할
```
<div id="load">
       
# 2.데이터 로드

</div>


```python
# 데이터를 불러옵니다.
CSV_PATH = './Heart_Attack.csv'
df = pd.read_csv(CSV_PATH)
```
<div id="check">
# 3.데이터 확인
</div>
```python
df.head()
df.columns
df.isna().sum()
df['class'].unique()
pd.value_count(df['class'])
```

df.coulms 결과

```python
Index(['age', 'gender', 'impluse', 'pressurehight', 'pressurelow', 'glucose',
       'kcm', 'troponin', 'class'],
      dtype='object')
```

결측값은 없었고, class의 경우 확인 결과 0,1로만 이루어진 이진수로 심장병의 유무에 대한 열이었습니다.



```python
age              0
gender           0
impulse          0
pressurehight    0
pressurelow      0
glucose          0
kcm              0
troponin         0
class            0
dtype: int64
```
클래스 0, 1 이진수 확인, 값의 빈도 확인
```python
array([0, 1])

class
1    810
0    509
Name: count, dtype: int64
```
<div id="prepare">
# 데이터 준비
</div>
맥박, 즉 심박수 'impluse' 에 대한 오타가 존재하니 이를 rename 합니다.

```python
df = df.rename(columns={'impluse': 'impulse'})
```
<div id="scaling">

## 데이터 표준화

</div>

```python

le = LabelEncoder()
df['class'] = le.fit_transform(df['class']) #수치 코드로 변환

df.head()
```

```python
age	gender	impulse	pressurehight	pressurelow	glucose	kcm	troponin	class
0	64	1	66	160	83	160.0	1.80	0.012	0
1	21	1	94	98	46	296.0	6.75	1.060	1
2	55	1	64	160	77	270.0	1.99	0.003	0
3	64	1	70	120	55	270.0	13.87	0.122	1
4	55	1	64	112	65	300.0	1.08	0.003	0
```
<div id="engineering">
       
## 성능 향상을 위한 기능 엔지니어링 및 분할

</div>

* 학습 세트와 테스트 세트로 데이터 분리

```python
# 특성 공학 및 데이터 분할
x = df.iloc[:, :-1] #마지막 열은 타겟 변수 'class' 심장병 여부이므로 제외
# 설명 변수(x)와 타겟 변수(y)로 분리
y= df[['class']]
```

데아터 세트 분할

```python
TEST_SIZE = 0.2  # 테스트 세트의 비율을 20%로 설정

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=0)

print(f"Training Records = {x_train.shape[0]} ; Validation Record {x_test.shape[0]} ")
# 학습 데이터와 테스트 데이터의 크기를 출력

Training Records = 1055 ; Validation Record 264 
```
Output

훈련 세트 1055, 테스트 세트 264 

<div id="para">

# 모델 학습 준비
       
## 매개변수 & 기능

</div>

텐서 플로 활용을 위한 매개변수 지정

* 입력층 뉴런 개수 설정 1
* 이진수 분류 뉴런 개수는 1개
* 은닉층 뉴런 개수 결정 입/출력 중간값
* 최적의 성능을 위한 배치 크기
* 전체 학습 데이터셋 학습 빈도 (에포크 150)
* 실험 횟수 25

```python
# 입력층의 크기 & x_train 데아터 특성 개수
INPUT_SHAPE = x_train.shape[1]
# 출력층의 크기 > 이진 분류이므로 크기 1
OUTPUT_SHAPE = 1
# 은닉층의 크기 > 입력층, 출력층 크기의 중간
NODES = np.ceil(2 * INPUT_SHAPE / 3 + OUTPUT_SHAPE).astype(int)
# 배치 크기 > 모델 학습 한번에 처리 데이터 양
BATCH_SIZE = np.ceil(len(x_train) / 16).astype(int)
# 에포크 수 > 모델 학습하는 동안 전체 데이터셋 한번 학습 과정
EPOCHS = 150
# 실험 횟수
TRIALS = 25
```

<div id="utility">

## 유틸리티 함수로 모델 구축

</div>


```python
def build_model(batch_size=None, nodes=None, input_shape=None, output_shape=None):
    # 모델을 생성
    model = keras.models.Sequential()

    # 첫 번째 은닉층을 추가
    model.add(keras.layers.Dense(nodes, input_shape=(input_shape,), activation='relu'))

    # 두 번째 은닉층을 추가
    model.add(keras.layers.Dense(nodes, activation='relu'))

    # 드롭아웃을 추가
    model.add(keras.layers.Dropout(0.5))

    # 출력층을 추가
    model.add(keras.layers.Dense(output_shape, activation='sigmoid'))

    return model
```

<div id="utility-plot">

## 유틸리티 함수: 곡선 플로팅

</div>

```python
# 유틸리티 함수: 곡선 플로팅
def plot_curves(hist):
    
    # 서브플롯 생성
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
    
    # 정확도 곡선 플로팅
    ax[0].plot(hist.history['accuracy'])
    ax[0].plot(hist.history['val_accuracy'])
    ax[0].set_title('model accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('accuracy')
    ax[0].legend(['train', 'val'], loc='best')
    
    # 손실 곡선 플로팅
    ax[1].plot(hist.history['loss'])
    ax[1].plot(hist.history['val_loss'])
    ax[1].set_title('model loss')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('loss')
    ax[1].legend(['train', 'val'], loc='upper right')    
    
    plt.show()
```

Output description


* build_model() 함수는 모델을 구축하는 함수
* plot_curves() 함수는 곡선을 플로팅하는 함수
* batch_size 매개변수는 배치 크기를 지정
* nodes 매개변수는 은닉층의 크기를 지정
* input_shape 매개변수는 입력층의 크기를 지정
* output_shape 매개변수는 출력층의 크기를 지정
* hist 매개변수는 모델의 훈련 이력을 지정
* 
<div id="Baseline">
       
## Baseline Model 설정

</div>

비교 기준을 위한 모델의 기본 성능 설정

```python
# 기본 모델
baseline_model = build_model(BATCH_SIZE, NODES, INPUT_SHAPE, OUTPUT_SHAPE)

# 모델을 컴파일합니다.
baseline_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #loss 손실함수, optimizer 최적화 알고리즘, metrics 평가 지표

# 모델 요약
baseline_model.summary()

```

Baseline Model에 대한 결과는 다음과 같고 3가지 dense 레이어와 1개의 dropout 레이어로 이루어져 있다.
각각의 레이어는 이진수를 포함하는 dense_2 레이어를 제외하고 7개의 뉴런을 가지고 있다.

Output

```python
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 7)                 63        
                                                                 
 dense_1 (Dense)             (None, 7)                 56        
                                                                 
 dropout (Dropout)           (None, 7)                 0         
                                                                 
 dense_2 (Dense)             (None, 1)                 8         
                                                                 
=================================================================
Total params: 127 (508.00 Byte)
Trainable params: 127 (508.00 Byte)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```


```python
# 세션 초기화
tf.keras.backend.clear_session()

# 모델 학습
baseline_hist = baseline_model.fit(x_train
          , y_train
          , epochs = EPOCHS
          , batch_size=BATCH_SIZE
          ,shuffle=True
          ,validation_data=(x_test, y_test)
          ,verbose=0
          ,callbacks=[TqdmCallback(verbose=0)]
          )

# 그래프 그리기
plot_curves(baseline_hist)
```

모델 학습에 대한 결과는 다음과 같고, 
* 학습 데이터에 대한 손실(loss)은 0.643
* 학습 데이터에 대한 정확도(accuracy)는 62.5%
* 검증 데이터에 대한 손실(val_loss)은 1.5
* 검증 데이터에 대한 정확도(val_accuracy)는 58%

모델은 학습 데이터에 대해 어느 정도 잘 동작하지만, 검증 데이터에 대해서는 성능이 떨어지는 결과로, 데이터 수가 적음에 따라 과적합이 의심된다.

Output

![image](https://github.com/plintAn/TensorflowEDA/assets/124107186/0854d9bf-8fab-43ac-9662-288b9093c3ec)

```python
150/150 [00:07<00:00, 23.10epoch/s, loss=0.643, accuracy=0.625, val_loss=1.5, val_accuracy=0.58]
```

<div id="opt">

## 최적화 모델

</div>


```python
def create_model(trial):
    # 층의 수, 해당 층의 유닛 및 학습률을 최적화
    n_layers = trial.suggest_int("n_layers", 1, 5)
    n_nodes = trial.suggest_int('n_nodes', 8, 64)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    
    # 모델 생성
    model = keras.models.Sequential()
    
    # 입력 층 추가
    model.add(keras.layers.Dense(BATCH_SIZE, input_shape=(INPUT_SHAPE,), activation='relu'))
    
    # 시도를 통해 동적으로 은닉층 추가
    for i in range(n_layers):
        model.add(keras.layers.Dense(n_nodes, activation='relu'))
    
    # 드롭아웃 층 추가
    model.add(keras.layers.Dropout(0.5))
    
    # 출력 층 추가
    model.add(keras.layers.Dense(OUTPUT_SHAPE,activation='sigmoid'))
    
    # 모델 컴파일
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    
    return model

```

```python
# 목적 함수
def objective(trial):
    
    # 모델 인스턴스화
    model_opt = create_model(trial)
    
    # 모델 학습
    model_opt.fit(x_train
                  ,y_train
                  ,epochs = EPOCHS
                  ,batch_size=BATCH_SIZE
                  ,shuffle=True
                  ,validation_data=(x_test, y_test)
                  ,verbose=0
                  ,callbacks=[TqdmCallback(verbose=0)])
    
    # 정확도 점수 계산
    acc_score = model_opt.evaluate(x_test, y_test, verbose=0)[1]
    
    return acc_score
```

```python
# 최적화 실행
study = optuna.create_study(direction="maximize", study_name="baseline model optimization")
study.optimize(objective, n_trials=TRIALS, n_jobs=-1)
```
Output

```python
150/150 [01:09<00:00, 2.48epoch/s, loss=0.48, accuracy=0.758, val_loss=0.672, val_accuracy=0.792]

150/150 [01:10<00:00, 2.04epoch/s, loss=0.491, accuracy=0.737, val_loss=1.22, val_accuracy=0.716]

150/150 [01:14<00:00, 1.84epoch/s, loss=0.663, accuracy=0.624, val_loss=0.685, val_accuracy=0.576]

150/150 [01:13<00:00, 2.22epoch/s, loss=0.647, accuracy=0.624, val_loss=0.672, val_accuracy=0.576]

150/150 [01:10<00:00, 2.22epoch/s, loss=0.477, accuracy=0.753, val_loss=0.799, val_accuracy=0.674]

150/150 [01:15<00:00, 1.59epoch/s, loss=0.386, accuracy=0.794, val_loss=0.839, val_accuracy=0.758]

150/150 [01:10<00:00, 2.32epoch/s, loss=0.62, accuracy=0.643, val_loss=0.758, val_accuracy=0.625]

150/150 [01:12<00:00, 2.12epoch/s, loss=0.495, accuracy=0.724, val_loss=0.798, val_accuracy=0.788]

150/150 [01:14<00:00, 1.81epoch/s, loss=0.662, accuracy=0.624, val_loss=0.686, val_accuracy=0.576]

150/150 [01:12<00:00, 2.28epoch/s, loss=0.631, accuracy=0.624, val_loss=0.645, val_accuracy=0.576]

150/150 [01:13<00:00, 1.97epoch/s, loss=0.656, accuracy=0.624, val_loss=0.672, val_accuracy=0.576]

150/150 [01:13<00:00, 2.27epoch/s, loss=0.538, accuracy=0.704, val_loss=0.715, val_accuracy=0.617]

150/150 [01:13<00:00, 2.16epoch/s, loss=0.463, accuracy=0.762, val_loss=0.621, val_accuracy=0.663]

150/150 [01:11<00:00, 1.90epoch/s, loss=0.417, accuracy=0.775, val_loss=1.03, val_accuracy=0.746]

150/150 [01:13<00:00, 2.24epoch/s, loss=0.432, accuracy=0.774, val_loss=0.64, val_accuracy=0.705]

150/150 [01:14<00:00, 1.73epoch/s, loss=0.371, accuracy=0.814, val_loss=0.64, val_accuracy=0.723]

150/150 [01:09<00:00, 2.42epoch/s, loss=0.663, accuracy=0.624, val_loss=0.686, val_accuracy=0.576]

150/150 [01:11<00:00, 4.29epoch/s, loss=0.41, accuracy=0.804, val_loss=1.05, val_accuracy=0.742]

150/150 [01:09<00:00, 2.54epoch/s, loss=0.575, accuracy=0.688, val_loss=0.731, val_accuracy=0.644]

150/150 [01:11<00:00, 8.94epoch/s, loss=0.578, accuracy=0.664, val_loss=0.646, val_accuracy=0.659]

150/150 [01:06<00:00, 2.66epoch/s, loss=0.656, accuracy=0.624, val_loss=0.678, val_accuracy=0.576]

150/150 [01:05<00:00, 2.62epoch/s, loss=0.604, accuracy=0.624, val_loss=0.594, val_accuracy=0.572]

150/150 [01:08<00:00, 4.53epoch/s, loss=0.487, accuracy=0.752, val_loss=0.938, val_accuracy=0.739]

150/150 [01:06<00:00, 2.51epoch/s, loss=0.609, accuracy=0.624, val_loss=0.735, val_accuracy=0.576]

```

다음 vallidation 정확도에 대한 시각화

![image](https://github.com/plintAn/TensorflowEDA/assets/124107186/b77ec524-6394-4004-8318-12544081ce8f)


```python
# 결과 추출
print('*'*100)
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("Best Params:")
print(study.best_params)
print('*'*100)
```
Output

```python
Number of finished trials:  25
Best trial:
  Value:  0.7916666865348816
Best Params:
{'n_layers': 1, 'n_nodes': 54, 'learning_rate': 0.005951303078196145}
```

<div id="opt-extract">

## 최적화 모델 추출

</div>

```python
# 최적화 모델 추출
baseline_model_opt = create_model(study.best_trial)
baseline_model_opt.summary()
```

Output

```python
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 66)                594       
                                                                 
 dense_1 (Dense)             (None, 54)                3618      
                                                                 
 dropout (Dropout)           (None, 54)                0         
                                                                 
 dense_2 (Dense)             (None, 1)                 55        
                                                                 
=================================================================
Total params: 4267 (16.67 KB)
Trainable params: 4267 (16.67 KB)
Non-trainable params: 0 (0.00 Byte)
```

```python
# 세션 정리
tf.keras.backend.clear_session()

# train 모델
baseline_hist_opt = baseline_model_opt.fit(x_train
                                           ,y_train
                                           ,epochs = EPOCHS
                                           ,batch_size=BATCH_SIZE
                                           ,shuffle=True
                                           ,validation_data=(x_test, y_test)
                                           ,verbose=0
                                           ,callbacks=[TqdmCallback(verbose=0)])

# 시각화
plot_curves(baseline_hist_opt)
```

<div id="opt-extract-active">

## 모델 최적화 적용 후 그래프 

</div>

Output

```python
150/150 [00:13<00:00, 12.34epoch/s, loss=0.438, accuracy=0.773, val_loss=0.795, val_accuracy=0.784]
```
![image](https://github.com/plintAn/TensorflowEDA/assets/124107186/dc2bfa55-2452-471a-949d-b39ab369ce21)


최적화 모델 적용 전과 후의 비교

![image](https://github.com/plintAn/TensorflowEDA/assets/124107186/462a61ae-dca7-4011-9a5a-33e62a57aefb)





