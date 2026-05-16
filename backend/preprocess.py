import pandas as pd
import numpy as np
def preprocess_data(df):
  df = df.copy()
  # clean columns & cells
  df.columns = df.columns.str.strip()
  for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].str.strip()
  
  # missing values marker
  df.replace('?', np.nan, inplace=True)
  
  # fill missing
  for col in df.columns:
    if df[col].isnull().any():
      if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
      else:
        df[col] = df[col].fillna(df[col].mode()[0])
  
  # outliers handling
  if 'hours-per-week' in df.columns:
    Q1 = df['hours-per-week'].quantile(0.25)
    Q3 = df['hours-per-week'].quantile(0.75)
    IQR = Q3 - Q1
    df['hours-per-week'] = df['hours-per-week'].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
  
  categorical_cols = ['marital-status','workclass','occupation','relationship']
  df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
  # capital-net
  if 'capital-gain' in df.columns and 'capital-loss' in df.columns:
    df['capital-net'] = df['capital-gain'] - df['capital-loss']
  # has capital
  if 'capital-gain' in df.columns and 'capital-loss' in df.columns:
    df['has-capital'] = ((df['capital-gain'] > 0) | (df['capital-loss'] > 0)).astype(int)
  # country
  if 'native-country' in df.columns:
    df['is_us'] = df['native-country'].apply(lambda x: 1 if x == 'United-States' else 0)
  if all(col in df.columns for col in ['education-num', 'hours-per-week']):
    df['socio_economic_score'] = (
      df['education-num'] * 2 +
      df['hours-per-week'] * 0.5 +
      df.get('capital-net', 0) * 0.01 +
      df.get('marital-status', 0) * 3 +
      df.get('is_us', 0) * 2+
      df.get('age', 0) * 2
    )
  
  # drop columns
  drop_cols = ['fnlwgt', 'education', 'race', 'native-country']
  df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
  
  # target clean
  if 'Income' in df.columns:
    df['Income'] = df['Income'].str.replace('.', '', regex=False).str.strip()
    df.dropna(subset=['Income'], inplace=True)
  return df