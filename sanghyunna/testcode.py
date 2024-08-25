import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np
from catboost import CatBoostClassifier

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
X = train_data.drop('target', axis=1)
y = train_data['target']

# 범주형 변수 리스트 추출
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# k-fold 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 하이퍼파라미터 공간 정의
space = {
    'iterations': scope.int(hp.quniform('iterations', 100, 1500, 50)),
    'depth': scope.int(hp.quniform('depth', 3, 15, 1)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.1)),
    'l2_leaf_reg': hp.loguniform('l2_leaf_reg', np.log(1e-3), np.log(10)),
    'border_count': scope.int(hp.quniform('border_count', 32, 255, 1)),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'random_strength': hp.uniform('random_strength', 1e-9, 10),
    'bagging_temperature': hp.uniform('bagging_temperature', 0.0, 1.0),
    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1)
}

# 목적 함수 정의
def objective(params):
    # CatBoost의 파라미터 타입 맞추기
    params['iterations'] = int(params['iterations'])
    params['depth'] = int(params['depth'])
    params['border_count'] = int(params['border_count'])
    
    # 모델 파이프라인 정의
    model = ImbPipeline(steps=[
        ('smote', SMOTE(random_state=42)),
        ('classifier', CatBoostClassifier(cat_features=categorical_cols, **params, task_type='GPU', verbose=0))
    ])
    
    score = cross_val_score(model, X, y, cv=kf, scoring='f1', n_jobs=-1)
    return {'loss': -score.mean(), 'status': STATUS_OK}

# Hyperopt을 이용한 하이퍼파라미터 최적화
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials, rstate=np.random.default_rng(42))

# 필요한 하이퍼파라미터를 정수형으로 변환
best = {k: int(v) if k in ['iterations', 'depth', 'border_count'] else v for k, v in best.items()}
print("Best Hyperparameters:", best)

# 최적의 하이퍼파라미터로 모델 학습
best_model = ImbPipeline(steps=[
    ('smote', SMOTE(random_state=42)),
    ('classifier', CatBoostClassifier(cat_features=categorical_cols, **best, task_type='GPU', verbose=0))
])
best_model.fit(X, y)

# 최적의 하이퍼파라미터의 f1 점수 출력
print("Best F1 Score:", -trials.best_trial['result']['loss'])

# 테스트 데이터 전처리 및 예측 수행
y_pred = best_model.predict(test_data)

# 'Set ID' 칼럼이 있는지 확인하고 없으면 추가
if 'Set ID' in test_data.columns:
    set_id = test_data['Set ID']
else:
    raise ValueError("'Set ID' Not found")

# 결과를 DataFrame으로 저장
submission = pd.DataFrame({
    'Set ID': set_id,
    'target': y_pred
})

# 타겟 레이블을 원래 값으로 변환
submission['target'] = submission['target'].replace({1: 'Normal', 0: 'AbNormal'}) 

# 결과를 CSV로 저장
submission.to_csv('submission.csv', index=False)

print("Prediction complete. Results saved to 'submission.csv'")
