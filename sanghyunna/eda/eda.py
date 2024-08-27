import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../train.csv')
test_df = pd.read_csv('../test.csv')
train_df_wo_target = df.drop('target', axis=1)
integrated_df = pd.concat([train_df_wo_target, test_df], axis=0)

# print(f"\n--------------\nShape: {df.shape}")

# Drop Columns with entirely missing values
df = df.dropna(axis=1, how='all')
# print(f"\n--------------\nShape after dropping null columns: {df.shape}")

# Drop Columns with entirely same values
df = df.loc[:, df.apply(pd.Series.nunique) != 1]
# print(f"\n--------------\nShape after dropping columns with same values: {df.shape}")

# # Check for missing values
# print("\n--------------\nMissing values: ")
# print(df.isnull().sum())

# # Check for duplicates
# print("\n--------------\nDuplicates: ")
# print(df.duplicated().sum())

# # Check for outliers
# print("\n--------------\nOutliers: ")
# print(df.describe())

# print("\n--------------\nHeatmap: ")
# numeric_columns = df.select_dtypes(include=[np.number])
# sns.heatmap(numeric_columns.corr(), cmap='coolwarm')
# plt.show()

# list of columns
# print("\n--------------\nColumns: ")

# for c in sorted(list(df.columns)):
#     print(c)

print(df['Workorder_AutoClave'].nunique())
print("--------------")
print(df['Workorder_Dam'].nunique())
print("--------------")
print(df['Workorder_Fill1'].nunique())
print("--------------")
print(df['Workorder_Fill2'].nunique())

import numpy as np

# 주어진 ndarray
column_to_check = "Fill2"
arr = list(integrated_df[f'Workorder_{column_to_check}'])

# 각 자리의 글자 집합을 찾는 함수
def find_character_sets(arr):
    n = len(arr[0])  # 문자열의 길이
    result = [set() for _ in range(n)]  # 각 자리별로 빈 집합 생성

    for string in arr:
        for i, char in enumerate(string):
            try:
                result[i].add(char)  # 각 자리의 집합에 문자 추가
            except:
                print(f"Error at {i} in {string}")
        
    # sort each set
    for i in range(n):
        result[i] = sorted(list(result[i]))

    return result

def transform_string(input_string):
    # 문자열을 '-'로 분리
    parts = input_string.split('-')
    
    # '-'로 분리된 두 번째 부분을 정수로 변환하여 앞의 0을 제거
    parts[1] = str(int(parts[1]))
    
    # 다시 합쳐서 반환
    return '-'.join(parts)

# 결과 계산
print(f"Column: Workorder_{column_to_check}")
print("--------------")
arr = [transform_string(string) for string in arr]
result_sets = find_character_sets(arr)
for i, s in enumerate(result_sets):
    print(f"Position [{i}]: {s}")

# sort the arr
arr = sorted(arr)

# Convert ndarray to DataFrame
df_output = pd.DataFrame(arr)

# Save DataFrame to CSV
df_output.to_csv('./output.csv', index=False)