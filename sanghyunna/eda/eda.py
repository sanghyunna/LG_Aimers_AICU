import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../train.csv')
test_df = pd.read_csv('../test.csv')
train_df_wo_target = df.drop('target', axis=1)
integrated_df = pd.concat([train_df_wo_target, test_df], axis=0)

# print(f"\n--------------\nShape: {df.shape}")

# dam = []
# fill1 = []
# fill2 = []
# autoclave = []
# for c in sorted(list(integrated_df.columns)):
#     if c.endswith('Dam'):
#         dam.append(c)
#     elif c.endswith('Fill1'):
#         fill1.append(c)
#     elif c.endswith('Fill2'):
#         fill2.append(c)
#     elif c.endswith('AutoClave'):
#         autoclave.append(c)

# print("Dam: ")
# for c in dam:
#     print(c)
# print("--------------")
# print("Fill1: ")
# for c in fill1:
#     print(c)
# print("--------------")
# print("Fill2: ")
# for c in fill2:
#     print(c)
# print("--------------")
# print("AutoClave: ")
# for c in autoclave:
#     print(c)



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

# print(df['Workorder_AutoClave'].nunique())
# print("--------------")
# print(df['Workorder_Dam'].nunique())
# print("--------------")
# print(df['Workorder_Fill1'].nunique())
# print("--------------")
# print(df['Workorder_Fill2'].nunique())

# 주어진 ndarray
# column_to_check = "Fill2"
# arr = list(integrated_df[f'Workorder_{column_to_check}'])

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
# print(f"Column: Workorder_{column_to_check}")
# print("--------------")
# arr = [transform_string(string) for string in arr]
# result_sets = find_character_sets(arr)
# for i, s in enumerate(result_sets):
#     print(f"Position [{i}]: {s}")

# # sort the arr
# arr = sorted(arr)

# # Convert ndarray to DataFrame
# df_output = pd.DataFrame(arr)

# # Save DataFrame to CSV
# df_output.to_csv('./output.csv', index=False)

col = integrated_df['Production Qty Collect Result_Dam']

# sort the ndarray
col_sorted = np.sort(col)

# convert numpy array to pandas Series
col_series = pd.Series(col_sorted)

unique_values = col_series.unique()

abnormal_percentage = []

for value in unique_values:
    subset = df[df['Production Qty Collect Result_Dam'] == value]
    total_count = len(subset)
    if total_count == 0:
        continue
    abnormal_count = len(subset[subset['target'] == 'AbNormal'])
    percentage = (abnormal_count / total_count) * 100
    abnormal_percentage.append((value, percentage))

abnormal_percentage = sorted(abnormal_percentage, key=lambda x: x[1], reverse=True)

# create plot for abnormal percentage
plt.figure(figsize=(20, 10))
plt.bar([x[0] for x in abnormal_percentage], [x[1] for x in abnormal_percentage])
plt.xlabel('Production Qty Collect Result_Dam')
plt.ylabel('Abnormal Percentage')
plt.title('Abnormal Percentage by Production Qty Collect Result_Dam')
plt.show()
