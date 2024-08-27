import pandas as pd
import sweetviz as sv

df = pd.read_csv('train.csv')
df.dropna(axis=1, how='all', inplace=True)
df = df.loc[:, df.apply(pd.Series.nunique) != 1]

df['target'] = df['target'].replace({'Normal': 1, 'AbNormal': 0})
report = sv.analyze(df, target_feat='target')
report.show_html()