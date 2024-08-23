import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope
import numpy as np

def cleanse_data(df, is_train=True):
    # 모든 값이 결측치인 열 제거
    df = df.dropna(axis=1, how='all')
    
    # 모든 값이 같은 열 제거
    df = df.loc[:, df.apply(pd.Series.nunique) != 1]

    # 타겟 레이블을 이진 정수로 변환
    if is_train:
        df['target'] = df['target'].replace({'Normal': 1, 'AbNormal': 0})

    return df

# CSV 파일 불러온 후 cleanse_data 호출
train_data = pd.read_csv('train.csv')
train_data = cleanse_data(train_data)

test_data = pd.read_csv('test.csv')
test_data = cleanse_data(test_data, is_train=False)

# 특징과 라벨 분리
X = train_data.drop('target', axis=1)  # 'target'은 예측하려는 타겟 변수 이름으로 수정해야 합니다.
y = train_data['target']

# 범주형 변수에 대한 원-핫 인코딩 적용
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# ColumnTransformer를 사용하여 범주형 변수는 OneHotEncoder를 적용하고 수치형 변수는 StandardScaler를 적용합니다.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# k-fold 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 하이퍼파라미터 공간 정의
space = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1500, 50)),
    'max_depth': scope.int(hp.quniform('max_depth', 3, 15, 1)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.1)),
    'num_leaves': scope.int(hp.quniform('num_leaves', 20, 200, 10)),
    'min_child_samples': scope.int(hp.quniform('min_child_samples', 10, 100, 5)),
    'subsample': hp.uniform('subsample', 0.3, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1),
    'min_child_weight': hp.loguniform('min_child_weight', np.log(1e-3), np.log(1e2)),
    'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-3), np.log(1)),
    'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-3), np.log(1)),
    'max_bin': 63,
    'gpu_use_dp': 'false',  # 단일 정밀도 사용
    'device': 'gpu',  # GPU 사용
    'gpu_platform_id': 1,
    'gpu_device_id': 0,
    'verbose': -1
}

# 목적 함수 정의
def objective(params):
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LGBMClassifier(**params))])
    score = cross_val_score(model, X, y, cv=kf, scoring='accuracy', n_jobs=-1)
    return {'loss': -score.mean(), 'status': STATUS_OK}

# Hyperopt을 이용한 하이퍼파라미터 최적화
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials, rstate=np.random.default_rng(42))

# 필요한 하이퍼파라미터를 정수형으로 변환
best = {k: int(v) if k in ['n_estimators', 'max_depth', 'num_leaves', 'min_child_samples'] else v for k, v in best.items()}

# 최적의 하이퍼파라미터로 모델 학습
best_model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LGBMClassifier(**best))])
best_model.fit(X, y)

# 테스트 데이터 전처리 및 예측 수행
y_pred = best_model.predict(test_data)

# 결과를 DataFrame으로 저장
submission = pd.DataFrame({
    'Set ID': test_data.index,  # 각 데이터의 ID 열이 있을 경우 사용, 없으면 수정 필요
    'target': y_pred
})

# 타겟 레이블을 원래 값으로 변환
submission['target'] = submission['target'].replace({1: 'Normal', 0: 'AbNormal'}) 

# 결과를 CSV로 저장
submission.to_csv('submission.csv', index=False)

print("Prediction complete. Results saved to 'submission.csv'")
