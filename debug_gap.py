import pandas as pd

df = pd.read_csv(r'aoee_output\aoee_unified_dataset.csv')

print("Update_Gap statistics:")
print(df['Update_Gap'].describe())
print("\nNegative values:", (df['Update_Gap'] < 0).sum())
print("\nSample data:")
print(df[['state', 'district', 'Update_Gap', 'Total_Updates']].head(10))
