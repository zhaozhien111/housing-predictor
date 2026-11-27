# 데이터 기반 예측 모델 및 서비스 개발하기

## 주제
주택 가격 예측 — 공개 주택 데이터셋으로 회귀 모델을 학습시키고, 예측 모델을 활용한 CLI 프로그램을 개발한다.

## 세부 기능 (3가지)
1. **예상 주택 가격 계산** : 사용자가 입력한 주택 정보를 바탕으로 모델이 집값을 예측한다.
2. **평균 대비 가격 수준 평가** : 예측 가격을 학습 데이터의 중위값과 비교해 높은 편·비슷한 편·낮은 편으로 나눈다.
3. **예상 월 대출금 계산** : 예측 가격을 기준으로 보증금 비율·이자율·대출 기간을 입력받아 월 대출금(원리금 균등상환)을 계산한다.

## 데이터셋
- 캘리포니아 주택 가격 데이터셋 (California Housing)
- scikit-learn 라이브러리 `fetch_california_housing()`으로 자동 다운로드
- 참고: Pace, R. Kelley and Ronald Barry, *Sparse Spatial Autoregressions*, 1997
- 원본 데이터: https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
- scikit-learn 설명: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset

> 수업에서 사용한 타이타닉·붓꽃·연비 데이터셋은 과제에서 사용하지 않았다.

## 파일 구성
| 파일 | 역할 |
| --- | --- |
| `housing.csv` | 공개 주택 데이터셋 (train.py가 읽어들이는 데이터 파일) |
| `train.py` | 데이터 불러오기, 전처리, 여러 알고리즘 비교, 모델 학습 및 저장 |
| `app.py` | 저장된 모델을 불러와 3가지 기능을 제공하는 응용 SW |
| `model.joblib` | 학습이 끝난 모델 파일 (train.py 실행 시 생성) |
| `meta.joblib` | 예측에 필요한 부가 정보 파일 (train.py 실행 시 생성) |

## 실행 방법
1. 패키지 설치
   ```
   pip install -r requirements.txt
   ```
2. 모델 학습 및 저장
   ```
   python train.py
   ```
3. 응용 프로그램 실행
   ```
   python app.py
   ```

## 모델 성능 비교 (테스트 데이터)
| 알고리즘 | R² | MAE |
| --- | --- | --- |
| LinearRegression | 0.5758 | 0.5332 |
| RandomForest | 0.8053 | 0.3274 |
| HistGradientBoosting | 0.8355 | 0.3096 |

- 최종 선택 모델: HistGradientBoostingRegressor (R² = 0.8355)

## GitHub 업로드 안내
- 세부 기능을 개발할 때마다 독립적으로 commit & push 한다.
- 커밋 메시지에는 기능 내용을 충분히 설명하는 글을 적는다.
  - 예: `feat: 주택 가격 예측 기능 추가`
  - 예: `feat: 평균 대비 가격 수준 평가 기능 추가`
  - 예: `feat: 월 대출금 계산 기능 추가`
- 레포지토리를 공개한 뒤, 레포지토리 주소를 과제 보고서에 넣어 LMS에 제출한다.
