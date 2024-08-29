import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier, Pool
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# 데이터 전처리 함수
def process_workorder(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).fillna('NaN')

    df['Workorder_Fill1'] = df['Workorder_Fill1'].apply(tailing_zero_remover)
    df['Workorder_Fill2'] = df['Workorder_Fill2'].apply(tailing_zero_remover)
    df['Workorder_Dam'] = df['Workorder_Dam'].apply(tailing_zero_remover)
    df['Workorder_AutoClave'] = df['Workorder_AutoClave'].apply(tailing_zero_remover)

    df['Workorder_Fill1_1'] = df['Workorder_Fill1'].str[0:2]
    df['Workorder_Fill1_2'] = df['Workorder_Fill1'].str[2:4]
    df['Workorder_Fill1_3'] = df['Workorder_Fill1'].str[9:10]

    df['Workorder_Fill2_1'] = df['Workorder_Fill2'].str[0:2]
    df['Workorder_Fill2_2'] = df['Workorder_Fill2'].str[2:4]
    df['Workorder_Fill2_3'] = df['Workorder_Fill2'].str[9:10]

    df['Workorder_Dam_1'] = df['Workorder_Dam'].str[0:2]
    df['Workorder_Dam_2'] = df['Workorder_Dam'].str[2:4]
    df['Workorder_Dam_3'] = df['Workorder_Dam'].str[9:10]

    df['Workorder_AutoClave_1'] = df['Workorder_AutoClave'].str[0:2]
    df['Workorder_AutoClave_2'] = df['Workorder_AutoClave'].str[2:4]
    df['Workorder_AutoClave_3'] = df['Workorder_AutoClave'].str[9:10]

    df.drop(columns=['Workorder_Fill1', 'Workorder_Fill2', 'Workorder_Dam', 'Workorder_AutoClave'], inplace=True)

    categorical_cols = [
        'Workorder_Fill1_1', 'Workorder_Fill1_2', 'Workorder_Fill1_3',
        'Workorder_Fill2_1', 'Workorder_Fill2_2', 'Workorder_Fill2_3',
        'Workorder_Dam_1', 'Workorder_Dam_2', 'Workorder_Dam_3',
        'Workorder_AutoClave_1', 'Workorder_AutoClave_2', 'Workorder_AutoClave_3'
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('object')

    df = df.fillna('NaN')

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    def convert_to_bool(series):
        tt_col = ['train', 'test']
        unique_values = series.unique()
        if len(unique_values) != 2:
            return series
        elif unique_values[0] in tt_col and unique_values[1] in tt_col:
            return series
        mapping = {unique_values[0]: False, unique_values[1]: True}
        return series.map(mapping)

    for col in df.columns:
        if df[col].nunique() == 2:
            df[col] = convert_to_bool(df[col])

    return df

# 문자열 전처리 함수
def tailing_zero_remover(input_string):
    parts = input_string.split('-')
    parts[1] = str(int(parts[1]))
    return '-'.join(parts)

# 데이터 로딩
ROOT_DIR = "./"
RANDOM_STATE = 110

train_data = pd.read_csv(os.path.join(ROOT_DIR, "train.csv"))
test_data = pd.read_csv(os.path.join(ROOT_DIR, "test.csv"))

# 통합 df 생성
train_data['tt'] = 'train'
test_data['tt'] = 'test'

train_target = train_data['target']
test_id = test_data['Set ID']
test_target = test_data['target']

test_data = test_data.drop(columns=['Set ID', 'target'])
train_data = train_data.drop(columns = ['target'])

integ_df = pd.concat([train_data, test_data], ignore_index=True)

# 결측치 비율이 90% 이상인 열 삭제
threshold = 90
missing_values_ratio = (integ_df.isnull().sum() / integ_df.shape[0]) * 100
integ_df= integ_df.drop(columns=missing_values_ratio[missing_values_ratio >= threshold].index)

# 상수 열 삭제
constant_columns = [col for col in integ_df.columns if integ_df[col].nunique() == 1]
integ_df.drop(columns=constant_columns, inplace=True)

# 상관계수가 높은 피처 삭제
numeric_df = integ_df.select_dtypes(include=[float, int])
correlation_matrix = numeric_df.corr()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
high_corr_features  = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]
integ_df_reduced = integ_df.drop(columns=high_corr_features)

# 범주형 열 처리 및 스케일링
def preprocess_data(df):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # 범주형 변수 원핫 인코딩
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # 수치형 변수 스케일링
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    return df

integ_df_reduced = preprocess_data(integ_df_reduced)

# 데이터 분리 및 타겟 인코딩
train = integ_df_reduced[integ_df_reduced['tt_train'] == True].copy()
test = integ_df_reduced[integ_df_reduced['tt_train'] == False].copy()
train.drop(columns=['tt_train'], inplace=True)
test.drop(columns=['tt_train'], inplace=True)
train['target'] = train_target.values

# 불균형 데이터 처리
normal_ratio = 2.0
df_normal = train[train["target"] == "Normal"]
df_abnormal = train[train["target"] == "AbNormal"]
df_normal = df_normal.sample(n=int(len(df_abnormal) * normal_ratio), replace=False, random_state=RANDOM_STATE)
train_df = pd.concat([df_normal, df_abnormal], axis=0).reset_index(drop=True)

# 타겟 인코딩
mapping = {'Normal': 1, 'AbNormal': 0}
train_df['target'] = train_df['target'].map(mapping)
X = train_df.drop('target', axis=1)
y = train_df['target']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 범주형 변수 인덱스 찾기
categorical_features_indices = [
    i for i, col in enumerate(X_train.columns)
    if X_train[col].dtype.name in ['category', 'object']
]

# CatBoost 및 Logistic Regression의 Optuna 하이퍼파라미터 최적화
def objective_catboost(trial):
    iterations = trial.suggest_int('iterations', 500, 1000)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.3, log=True)
    depth = trial.suggest_int('depth', 4, 10)
    l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1e-2, 100.0, log=True)
    rsm = trial.suggest_float('rsm', 0.5, 1.0) ##
    border_count = trial.suggest_int('border_count', 32, 255)

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        rsm=rsm, ##
        border_count=border_count,
        eval_metric='F1',
        random_seed=42,
        logging_level='Silent',
        thread_count=-1,
        task_type="CPU",
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        train_pool = Pool(data=X_fold_train, label=y_fold_train, cat_features=categorical_features_indices)
        val_pool = Pool(data=X_fold_val, label=y_fold_val, cat_features=categorical_features_indices)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, verbose=False)
        y_pred_encoded = model.predict(X_fold_val)
        fold_f1_score = f1_score(y_fold_val, y_pred_encoded)
        f1_scores.append(fold_f1_score)

    return np.mean(f1_scores)

catboost_model = None
if os.path.exists(os.path.join(ROOT_DIR, 'best_catboost_model.pkl')):
  with open(os.path.join(ROOT_DIR, 'best_catboost_model.pkl'), 'rb') as f:
      catboost_model = pickle.load(f)
else:
  study_catboost = optuna.create_study(direction='maximize')
  study_catboost.optimize(objective_catboost, n_trials=50)
  best_params_catboost = study_catboost.best_trial.params

  # 최적의 CatBoost 모델
  catboost_model = CatBoostClassifier(
      iterations=best_params_catboost['iterations'],
      learning_rate=best_params_catboost['learning_rate'],
      depth=best_params_catboost['depth'],
      l2_leaf_reg=best_params_catboost['l2_leaf_reg'],
      rsm=best_params_catboost['rsm'], ##
      border_count=best_params_catboost['border_count'],
      eval_metric='F1',
      random_seed=42,
      cat_features=categorical_features_indices,
      verbose=100,
      early_stopping_rounds=100
  )
  catboost_model.fit(X_train, y_train, cat_features=categorical_features_indices)
  # 모델을 pkl 파일로 저장
  with open(os.path.join(ROOT_DIR, 'best_catboost_model.pkl'), 'wb') as f:
      pickle.dump(catboost_model, f)

# Logistic Regression 최적화 함수
def objective_logistic(trial):
    # C 파라미터를 최적화
    C = trial.suggest_float('C', 1e-4, 1e2, log=True)

    # Logistic Regression 모델 정의
    model = make_pipeline(
        StandardScaler(),  # 데이터 스케일링 추가
        LogisticRegression(
            penalty='l2',               # L2 정규화 사용
            C=C,
            solver='saga',              # saga 솔버 사용
            class_weight='balanced',    # 불균형 데이터셋에 대해 클래스 가중치 적용
            random_state=42,
            max_iter=1000
        )
    )

    # 교차 검증 설정
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    # 교차 검증 루프
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_fold_train, y_fold_train)
        y_pred_encoded = model.predict(X_fold_val)
        fold_f1_score = f1_score(y_fold_val, y_pred_encoded)
        f1_scores.append(fold_f1_score)

    return np.mean(f1_scores)

# Logistic Regression 모델 최적화 및 학습
logistic_model = None
if os.path.exists(os.path.join(ROOT_DIR, 'best_logistic_model.pkl')):
    with open(os.path.join(ROOT_DIR, 'best_logistic_model.pkl'), 'rb') as f:
        logistic_model = pickle.load(f)
else:
    study_logistic = optuna.create_study(direction='maximize')
    study_logistic.optimize(objective_logistic, n_trials=50)
    best_params_logistic = study_logistic.best_trial.params

    # 최적의 Logistic Regression 모델
    logistic_model = make_pipeline(
        StandardScaler(),  # 데이터 스케일링 추가
        LogisticRegression(
            penalty='l2',               # L2 정규화 사용
            C=best_params_logistic['C'],
            solver='saga',              # saga 솔버 사용
            class_weight='balanced',    # 불균형 데이터셋에 대해 클래스 가중치 적용
            random_state=42,
            max_iter=1000
        )
    )
    logistic_model.fit(X_train, y_train)

    # 모델을 pkl 파일로 저장
    with open(os.path.join(ROOT_DIR, 'best_logistic_model.pkl'), 'wb') as f:
        pickle.dump(logistic_model, f)

# 앙상블 모델 정의 (Soft Voting)
ensemble_model = VotingClassifier(
    estimators=[
        ('catboost', catboost_model),
        ('logistic', logistic_model),
    ],
    voting='soft'
)

# 앙상블 모델 학습 및 예측
ensemble_model.fit(X_train, y_train)
y_val_pred = ensemble_model.predict(X_val)

# F1 점수 계산
f1 = f1_score(y_val, y_val_pred)
print(f"Final F1 score with ensemble: {f1}")

# 테스트 데이터 예측
test_pred = ensemble_model.predict(test)

# 디코딩
mapping = {1: 'Normal', 0: 'AbNormal'}
test_pred = np.vectorize(mapping.get)(test_pred)

# 제출 파일 생성
df_sub = pd.read_csv(os.path.join(ROOT_DIR, "submission.csv"))
df_sub["target"] = test_pred
df_sub.to_csv(os.path.join(ROOT_DIR, "submission_ensemble.csv"), index=False)
