# 2. Visualization
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
output_folder = './analytics/'
if not os.path.exists(output_folder):
  os.makedirs(output_folder)
def plot_target_distribution(df):
  if 'Income ' not in df.columns:
    return

  plt.figure(figsize=(6, 6))
  df['Income '].value_counts().plot(
    kind='pie',
    color=['#4CAF50', '#FF5722'],
    autopct='%1.1f%%'
  )
  plt.title('Income Distribution')
  plt.xlabel('Income Class')
  plt.ylabel('Count')
  plt.tight_layout()
  plt.savefig('analytics/target_distribution_pie.png')
  plt.close()


def plot_categorical_vs_income(df, col):
  if col not in df.columns or 'Income ' not in df.columns:
    return
  
  cross = pd.crosstab(df[col], df['Income '])
  cross.plot(
    kind='bar',
    stacked=True,
    figsize=(10, 5),
    colormap='viridis'
  )
  plt.title(f'{col} vs Income (Stacked)')
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.savefig(f'analytics/{col}_stacked.png')
  plt.close()


def plot_numerical_vs_income(df, col):
  if col not in df.columns or 'Income ' not in df.columns:
    return

  plt.figure(figsize=(10, 5))
  sns.histplot(
    data=df,
    x=col,
    hue='Income ',
    kde=True,
    bins=30
  )
  plt.title(f'{col} Distribution (Histogram + KDE)')
  plt.tight_layout()
  plt.savefig(f'analytics/{col}_hist.png')
  plt.close()


def chi_square_test(df, col):
  if col not in df.columns or 'Income ' not in df.columns:
    return

  contingency = pd.crosstab(df[col], df['Income '])
  chi2, p, dof,excepted= chi2_contingency(contingency)
  print(f"\n===== Chi-Square Test: {col} vs Income =====")
  print(f"Chi2 Statistic : {chi2:.2f}")
  print(f"P-value        : {p:.6f}")
  print(f"Degrees Freedom: {dof}")
  if p < 0.05:
    print("Strong relationship with Income")
  else:
    print("Weak or no significant relationship")


def plot_correlation_heatmap(df):
  numeric_df = df.select_dtypes(include=['number'])
  if numeric_df.empty:
    return

  plt.figure(figsize=(12, 8))
  corr = numeric_df.corr()
  sns.heatmap(
    corr,
    annot=True,
    cmap='inferno',
    fmt=".2f",
    linewidths=0.5
  )
  plt.title('Correlation Heatmap')
  plt.tight_layout()
  plt.savefig('analytics/correlation_heatmap.png')
  plt.close()


def plot_pairplot(df):
  sample_df = df.sample(min(500, len(df)))
  sns.pairplot(
    sample_df,
    hue='Income ',
  )
  plt.savefig('analytics/pairplot.png')
  plt.close()


def deep_analysis_summary(df):
  print("\n===== Analysis Insights =====")
  if 'education' in df.columns:
    edu = pd.crosstab(df['education'], df['Income '], normalize='index')
    edu_sorted = edu.sort_values(by=edu.columns[1], ascending=False)
    print("\nTop education levels leading to >50K:")
    print(edu_sorted.head())
    print("\nLowest education levels:")
    print(edu_sorted.tail())

  if 'marital-status' in df.columns:
    marital = pd.crosstab(df['marital-status'], df['Income '], normalize='index')
    marital_sorted = marital.sort_values(by=marital.columns[1], ascending=False)
    print("\nMarital status impact:")
    print(marital_sorted)

  if 'hours-per-week' in df.columns:
    hours = df.groupby('Income ')['hours-per-week'].mean()
    print("\nAverage working hours:")
    print(hours)
    if hours[' >50K'] > hours[' <=50K']:
      print("People with higher income tend to work more hours.")
    else:
      print("Working hours are not a strong factor.")

  if 'age' in df.columns:
    age = df.groupby('Income ')['age'].mean()
    print("\nAverage age:")
    print(age)
    if age[' >50K'] > age[' <=50K']:
      print("Higher income group tends to be older.")
    else:
      print("Age is not a strong factor.")


def create_capital_category(df):
  if 'capital-gain' not in df.columns or 'capital-loss' not in df.columns:
    return df
  def label_row(row):
    if row['capital-gain'] > 0:
      return 'Gain'
    elif row['capital-loss'] > 0:
      return 'Loss'
    else:
      return 'Neutral'
  df = df.copy()
  df['capital-category'] = df.apply(label_row, axis=1)
  return df


def plot_capital_category_vs_income(df):
  if 'capital-category' not in df.columns or 'Income ' not in df.columns:
    return
  cross = pd.crosstab(df['capital-category'], df['Income '])
  cross.plot(
    kind='bar',
    stacked=True,
    figsize=(8, 5),
    colormap='viridis'
  )
  plt.title('Capital Category vs Income')
  plt.xticks(rotation=0)
  plt.tight_layout()
  plt.savefig('analytics/capital_category_vs_income.png')
  plt.close()


def run_full_analysis(df):
  plot_target_distribution(df)
  cat_features = ['sex', 'education', 'marital-status', 'occupation']
  for col in cat_features:
    plot_categorical_vs_income(df, col)
    chi_square_test(df, col)
  for col in cat_features:
    plot_categorical_vs_income(df, col)

  num_features = ['age', 'hours-per-week']
  for col in num_features:
    plot_numerical_vs_income(df, col)
    
  df = create_capital_category(df)
  plot_capital_category_vs_income(df)

  plot_correlation_heatmap(df)
  plot_pairplot(df)
  deep_analysis_summary(df)


def corr_between_each_col_and_income(correlation_matrix):
  income_correlation = correlation_matrix[['Income']].sort_values(by='Income',ascending=False)
  plt.figure(figsize=(14, 10))
  sns.heatmap(
    income_correlation,
    annot=True,
    cmap='viridis',
    fmt=".2f",
    linewidths=0.5
  )
  plt.title("Correlation Between Features and Income", fontsize=16)
  plt.tight_layout()
  plt.savefig("./analytics/income_correlation_heatmap.png", dpi=300, bbox_inches='tight')
  plt.close()