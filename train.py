# train.py
# 공개 데이터셋으로 여러 회귀 알고리즘을 학습한 뒤, 성능이 가장 좋은 모델을
# joblib 파일로 저장하는 코드입니다. 응용 SW(app.py)와는 분리해서 작성합니다.

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1. 공개 데이터셋 불러오기
# 캘리포니아 주택 가격 데이터셋 (타이타닉·붓꽃·연비 데이터셋은 사용 금지 규정에 해당하지 않음)
# 네트워크 다운로드 대신 폴더 안 housing.csv 파일을 직접 읽어온다.
df = pd.read_csv("housing.csv")
X = df.drop(columns=["MedHouseVal"]).values
y = df["MedHouseVal"].values
feature_names = [c for c in df.columns if c != "MedHouseVal"]

# 2. 학습 데이터와 테스트 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 비교할 알고리즘 목록
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
}

results = []
best_name = None
best_score = -np.inf

for name, model in models.items():
    # 스케일링과 모델을 하나의 파이프라인으로 묶어 학습시킨다
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    results.append((name, r2, mae))

    if r2 > best_score:
        best_score = r2
        best_name = name

print("=== 모델 성능 비교 (테스트 데이터) ===")
for name, r2, mae in results:
    print(f"{name:22s} R²={r2:.4f}  MAE={mae:.4f}")
print(f"가장 성능이 좋은 모델: {best_name} (R²={best_score:.4f})")

# 4. 가장 좋은 모델로 최종 파이프라인을 다시 만들어 저장한다
final_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", models[best_name]),
])
final_pipe.fit(X_train, y_train)

# 5. 예측에 필요한 부가 정보(특징 이름, 집값 중위값 등)도 함께 저장
meta = {
    "feature_names": feature_names,
    "target_mean": float(np.mean(y_train)),
    "target_median": float(np.median(y_train)),
    "best_model_name": best_name,
    "test_r2": float(best_score),
}

joblib.dump(final_pipe, "model.joblib")
joblib.dump(meta, "meta.joblib")
print("model.joblib, meta.joblib 저장 완료")
