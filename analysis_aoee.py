import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def run_analysis():
    base_path = r"c:\Users\param\OneDrive\Desktop\Data Hackathon UDAI"
    data_path = os.path.join(base_path, 'aoee_output', 'aoee_unified_dataset.csv')
    plots_dir = os.path.join(base_path, 'aoee_output', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Loading unified dataset...")
    df = pd.read_csv(data_path)
    
    # 1. Visualizations
    print("Generating visualizations...")
    sns.set_theme(style="whitegrid")
    
    # Uni-variate: Mandatory Update Backlog (Age 5-17)
    plt.figure(figsize=(10, 6))
    # Combining bio and demo 5-17 counts for backlog
    df['Backlog_5_17'] = df['bio_age_5_17'] + df['demo_age_5_17']
    sns.histplot(df['Backlog_5_17'], bins=30, kde=True, color='orange')
    plt.title('Distribution of Mandatory Update Backlog (Age 5-17)')
    plt.xlabel('Volume of Pending Updates')
    plt.savefig(os.path.join(plots_dir, 'univariate_backlog_5_17.png'))
    plt.close()
    
    # Bivariate: Pincode vs Failure Rate (Hardware Hotspots)
    # Using 'Auth_Failure_Rate' generated in process step
    # Scatter plot for Top 500 Pincodes by Activity
    top_pincodes = df.sort_values('Total_Updates', ascending=False).head(500)
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=top_pincodes, x='pincode', y='Auth_Failure_Rate', hue='state', legend=False, palette='viridis')
    plt.title('Hardware Hotspots: Pincode vs Auth Failure Rate')
    plt.xticks(rotation=45) 
    plt.ylabel('Failure Rate (%)')
    plt.close()

    # Trivariate 1: Auth Success Rate Heatmap (State x Month)
    # Metric: Weighted Average Success Rate (100 - Failure Rate)
    df['Success_Rate'] = 100 - df['Auth_Failure_Rate']

    # Explicit Grid Aggregation to prevent "middle rows"
    # Group by State and Month first to ensure uniqueness
    success_agg = df.groupby(['state', 'Month'])['Success_Rate'].mean().reset_index()
    heatmap_success = success_agg.pivot(index='state', columns='Month', values='Success_Rate')
    
    plt.figure(figsize=(16, 12)) # Expanded height
    ax1 = sns.heatmap(heatmap_success, cmap="RdYlGn", annot=True, annot_kws={'size': 9}, 
                     fmt='.1f', linewidths=0.5)
    plt.title('Aadhaar Auth Success Rate: State-Wise Performance by Month (2024)')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('')
    # Rotate labels
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0) # Months are short, keep horizontal
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'trivariate_success_heatmap.png'))
    plt.close()

    # Trivariate 2: Activity Volume Heatmap (State x Month) - Finalist Grade
    # Aggregating all activity (Updates) to State-Month level
    # Using 'Total_Updates' as the metric for Operational Load
    state_monthly = df.groupby(['state', 'Month'])['Total_Updates'].sum().unstack().fillna(0)

    # Custom formatter for human-readable numbers
    def format_large_numbers(val):
        if val >= 1e6:
            return f'{val/1e6:.1f}M'
        elif val >= 1e3:
            return f'{val/1e3:.0f}K'
        else:
            return f'{val:.0f}'
    
    plt.figure(figsize=(16, 10))
    # Apply custom formatter to each cell
    annot_data = state_monthly.applymap(format_large_numbers)
    
    ax2 = sns.heatmap(state_monthly, 
                      annot=annot_data,
                      fmt='',  # Empty because we're passing pre-formatted strings
                      annot_kws={"size": 8}, 
                      cmap="YlGnBu", 
                      linewidths=0.5)
    
    plt.title("Total Activity Updates by State and Month, 2024 (Volume)")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'trivariate_volume_heatmap.png'))
    plt.close()
    
    # Anomaly Detection: 2-Sigma Baseline
    # Aggregate monthly updates
    monthly_trend = df.groupby('Month')['Total_Updates'].sum().reset_index()
    
    mean_val = monthly_trend['Total_Updates'].mean()
    std_val = monthly_trend['Total_Updates'].std()
    
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_trend['Month'], monthly_trend['Total_Updates'], marker='o', label='Actual Updates')
    plt.axhline(y=mean_val, color='g', linestyle='--', label='Mean')
    plt.axhline(y=mean_val - 2*std_val, color='r', linestyle='--', label='Lower Control Limit (2-Sigma)')
    plt.fill_between(monthly_trend['Month'], mean_val - 2*std_val, mean_val + 2*std_val, color='gray', alpha=0.1)
    
    plt.title('Operational Stability: 2-Sigma Anomaly Detection')
    plt.ylabel('Total Updates')
    plt.xlabel('Month')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'anomaly_2sigma.png'))
    plt.close()

    print(f"Advanced visualizations saved to {plots_dir}")

if __name__ == "__main__":
    run_analysis()
