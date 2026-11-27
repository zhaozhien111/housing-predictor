# app.py
# train.py가 저장한 모델 파일을 불러와 집값을 예측하고,
# 하나의 주제(주택 가격 예측) 안에서 3가지 세부 기능을 제공하는 CLI 프로그램입니다.

import joblib
import numpy as np

# 저장된 모델과 부가 정보 불러오기
model = joblib.load("model.joblib")
meta = joblib.load("meta.joblib")

# 입력받을 특징과 안내 문구
INPUT_GUIDE = [
    ("MedInc", "중위 소득 (만 달러 단위, 예: 3.5)"),
    ("HouseAge", "주택 연식 (예: 25)"),
    ("AveRooms", "평균 방 수 (예: 5.2)"),
    ("AveBedrms", "평균 침실 수 (예: 1.0)"),
    ("Population", "지역 인구 (예: 1200)"),
    ("AveOccup", "평균 거주 인원 (예: 2.5)"),
    ("Latitude", "위도 (예: 37.5)"),
    ("Longitude", "경도 (예: -122.3)"),
]


def input_features():
    """사용자에게 8개 특징 값을 입력받아 모델에 넣을 numpy 배열로 돌려준다."""
    values = []
    print("주택 정보를 입력해 주세요.")
    for name, guide in INPUT_GUIDE:
        while True:
            raw = input(f"- {guide}: ").strip()
            try:
                values.append(float(raw))
                break
            except ValueError:
                print("숫자로 다시 입력해 주세요.")
    return np.array([values])


def predict_price(x):
    """모델로 예측한 집값(10만 달러 단위)을 돌려준다."""
    return float(model.predict(x)[0])


def read_number(prompt):
    """빈 칸이나 잘못된 입력이 들어오면 다시 묻는 숫자 입력 처리."""
    while True:
        raw = input(prompt).strip()
        try:
            return float(raw)
        except ValueError:
            print("숫자로 다시 입력해 주세요.")


def read_integer(prompt):
    """정수만 받는 입력 처리."""
    while True:
        raw = input(prompt).strip()
        try:
            return int(raw)
        except ValueError:
            print("정수로 다시 입력해 주세요.")


def feature1_price():
    # 세부 기능 1: 예상 주택 가격 계산
    x = input_features()
    pred = predict_price(x)
    dollars = pred * 100000
    print(f"예상 주택 가격: 약 {pred:.3f} (10만 달러 단위), 약 ${dollars:,.0f}")


def feature2_level():
    # 세부 기능 2: 학습 데이터 평균 대비 가격 수준 평가
    x = input_features()
    pred = predict_price(x)
    median = meta["target_median"]

    if pred >= median * 1.15:
        level = "평균보다 높은 편"
    elif pred <= median * 0.85:
        level = "평균보다 낮은 편"
    else:
        level = "평균과 비슷한 편"

    print(f"예측 가격 {pred:.3f}, 학습 데이터 중위값 {median:.3f} → {level}")


def feature3_monthly_payment():
    # 세부 기능 3: 예상 월 대출금 계산 (원리금 균등상환 방식)
    x = input_features()
    pred = predict_price(x)
    price = pred * 100000

    down_ratio = read_number("보증금 비율(%, 예: 30): ") / 100
    interest = read_number("연 이자율(%, 예: 4.5): ") / 100
    years = read_integer("대출 기간(년, 예: 30): ")

    loan = price * (1 - down_ratio)
    r = interest / 12
    n = years * 12

    if r == 0:
        monthly = loan / n
    else:
        monthly = loan * r * (1 + r) ** n / ((1 + r) ** n - 1)

    print(f"대출금: 약 ${loan:,.0f}, 예상 월 대출금: 약 ${monthly:,.0f}")


def main():
    print("=== 주택 가격 예측 서비스 ===")
    while True:
        print("\n1. 예상 주택 가격 계산")
        print("2. 평균 대비 가격 수준 평가")
        print("3. 예상 월 대출금 계산")
        print("0. 종료")
        choice = input("원하는 기능 번호를 입력하세요: ").strip()

        if choice == "1":
            feature1_price()
        elif choice == "2":
            feature2_level()
        elif choice == "3":
            feature3_monthly_payment()
        elif choice == "0":
            print("이용해 주셔서 감사합니다.")
            break
        else:
            print("1, 2, 3, 0 중에서 입력해 주세요.")


if __name__ == "__main__":
    main()
