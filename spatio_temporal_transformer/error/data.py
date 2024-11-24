import pandas as pd
import numpy as np

# 랜덤 시드 설정
np.random.seed(42)

# 샘플 데이터 생성 함수
def generate_sample_data(n_samples):
    data = {
        'area': np.random.randint(50, 300, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'age': np.random.randint(0, 50, n_samples),
        'distance_to_center': np.random.uniform(0, 20, n_samples),
        'school_rating': np.random.uniform(1, 10, n_samples),
        'crime_rate': np.random.uniform(0, 100, n_samples),
    }
    
    # 가격 생성 (특성들의 조합으로)
    price = (
        data['area'] * 1000 +
        data['bedrooms'] * 50000 +
        data['bathrooms'] * 30000 -
        data['age'] * 1000 -
        data['distance_to_center'] * 10000 +
        data['school_rating'] * 20000 -
        data['crime_rate'] * 500 +
        np.random.normal(0, 50000, n_samples)  # 노이즈 추가
    )
    
    data['price'] = np.maximum(price, 50000)  # 최소 가격 설정
    
    return pd.DataFrame(data)

# 훈련 데이터 생성 및 저장
train_data = generate_sample_data(1000)
train_data.to_csv('train.csv', index=False)

# 테스트 데이터 생성 및 저장
test_data = generate_sample_data(200)
test_data.to_csv('test.csv', index=False)

print("훈련 데이터 샘플:")
print(train_data.head())
print("\n테스트 데이터 샘플:")
print(test_data.head())
print("\n데이터 통계:")
print(train_data.describe())