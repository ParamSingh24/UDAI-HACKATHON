import pandas as pd
import os
import glob
import numpy as np
import re

def process_aoee_data():
    base_path = r"c:\Users\param\OneDrive\Desktop\Data Hackathon UDAI"
    cleaned_dir = os.path.join(base_path, 'cleaned_data')
    output_dir = os.path.join(base_path, 'aoee_output')
    os.makedirs(output_dir, exist_ok=True)

    print("Loading cleaned datasets...")
    try:
        bio_df = pd.read_csv(os.path.join(cleaned_dir, 'cleaned_biometric.csv'))
        demo_df = pd.read_csv(os.path.join(cleaned_dir, 'cleaned_demographic.csv'))
        enroll_df = pd.read_csv(os.path.join(cleaned_dir, 'cleaned_enrollment.csv'))
    except FileNotFoundError:
        print("Error: Cleaned datasets not found. Please run clean_aadhar_data.py first.")
        return

    # --- 1. Enhanced Data cleaning & Merging ---
    
    # Validation Helper: Strict 6-digit Pincode
    def is_valid_pincode(p):
        s = str(p).split('.')[0] # Handle float strings
        return bool(re.match(r'^[1-9][0-9]{5}$', s))

    # Clean Pincodes & States
    # Store references in a list to iterate, but we need to update the original variables or the list contents
    datasets = [bio_df, demo_df, enroll_df]
    cleaned_datasets = []

    # Standardize State Names (Robust Cleaning)
    state_map = {
        '100000': 'Unknown',
        'Andaman & Nicobar Islands': 'Andaman And Nicobar Islands',
        'Andhra Pr': 'Andhra Pradesh',
        'Balanagar': 'Telangana',
        'Chhatisgarh': 'Chhattisgarh',
        'Dadra & Nagar Haveli': 'Dadra And Nagar Haveli',
        'Dadra And Nagar Haveli': 'Dadra And Nagar Haveli And Daman And Diu',
        'Daman & Diu': 'Daman And Diu',
        'Daman And Diu': 'Dadra And Nagar Haveli And Daman And Diu',
        'Darbhanga': 'Bihar',
        'Jaipur': 'Rajasthan',
        'Jammu & Kashmir': 'Jammu And Kashmir',
        'Madanapalle': 'Andhra Pradesh',
        'Nagpur': 'Maharashtra',
        'Orissa': 'Odisha',
        'Pondicherry': 'Puducherry',
        'Puttenahalli': 'Karnataka',
        'Raja Annamalai Puram': 'Tamil Nadu',
        'Tamilnadu': 'Tamil Nadu',
        'The Dadra And Nagar Haveli And Daman And Diu': 'Dadra And Nagar Haveli And Daman And Diu',
        'Uttaranchal': 'Uttarakhand',
        'West  Bengal': 'West Bengal',
        'West Bangal': 'West Bengal',
        'West Bengli': 'West Bengal',
        'Westbengal': 'West Bengal'
    }

    for df in datasets:
        # Force to string first
        df['pincode'] = df['pincode'].astype(str).str.split('.').str[0]
        # Keep only valid pincodes
        df = df[df['pincode'].apply(is_valid_pincode)].copy()

        # Title Clean first
        df['state'] = df['state'].str.title().str.strip()
        # Apply Map
        df['state'] = df['state'].replace(state_map)
        
        cleaned_datasets.append(df)
    
    # Unpack back
    bio_df, demo_df, enroll_df = cleaned_datasets
        
    # Aggregate by Pincode, State, District, Month, Year
    group_cols = ['state', 'district', 'pincode', 'Year', 'Month']
    
    print("Aggregating datasets...")
    bio_agg = bio_df.groupby(group_cols)[['bio_age_5_17', 'bio_age_17_', 'Total_Activity']].sum().reset_index()
    bio_agg.rename(columns={'Total_Activity': 'Biometric_Updates'}, inplace=True)
    
    demo_agg = demo_df.groupby(group_cols)[['demo_age_5_17', 'demo_age_17_', 'Total_Activity']].sum().reset_index()
    demo_agg.rename(columns={'Total_Activity': 'Demographic_Updates'}, inplace=True)
    
    enroll_agg = enroll_df.groupby(group_cols)[['age_0_5', 'age_5_17', 'age_18_greater', 'Total_Activity']].sum().reset_index()
    enroll_agg.rename(columns={'Total_Activity': 'Total_Enrollment'}, inplace=True)

    # Merge
    print("Merging datasets...")
    merged_df = pd.merge(enroll_agg, bio_agg, on=group_cols, how='outer')
    merged_df = pd.merge(merged_df, demo_agg, on=group_cols, how='outer')
    
    # Fill NaN with 0
    count_cols = ['age_0_5', 'age_5_17', 'age_18_greater', 'Total_Enrollment', 
                  'bio_age_5_17', 'bio_age_17_', 'Biometric_Updates',
                  'demo_age_5_17', 'demo_age_17_', 'Demographic_Updates']
    merged_df[count_cols] = merged_df[count_cols].fillna(0)

    # --- 2. Advanced Feature Engineering ---
    print("Engineering features (Advanced)...")
    
    merged_df['Total_Updates'] = merged_df['Biometric_Updates'] + merged_df['Demographic_Updates']
    
    # A. Service Desert Score (Population / Activity) ~ Accessibility Score
    # "High score = High Demand, Low Supply"
    # We use Total_Enrollment as a proxy for "Eligible Population" base.
    merged_df['Service_Desert_Score'] = merged_df['Total_Enrollment'] / (merged_df['Total_Updates'] + 1)
    
    # B. GUP Index (Gender Update Parity)
    # Source data has no gender. Simulating for Prototype Demo.
    # Logic: Randomly assign split roughly 48-52% for Female updates.
    np.random.seed(42)
    female_ratio = np.random.normal(0.48, 0.05, size=len(merged_df))
    female_ratio = np.clip(female_ratio, 0.3, 0.7) # Clip extreme bounds
    
    merged_df['Female_Updates'] = (merged_df['Total_Updates'] * female_ratio).astype(int)
    merged_df['Male_Updates'] = merged_df['Total_Updates'] - merged_df['Female_Updates']
    
    # GUP Index = Female / Male (Ideal ~ 1.0)
    merged_df['GUP_Index'] = merged_df['Female_Updates'] / (merged_df['Male_Updates'] + 1)
    
    # C. Update Gap (FIXED: Should be positive when there's actual gap)
    # Logic: Children (age 5-17) need mandatory updates every 5/15 years
    # Estimate: 20% of age_5_17 population needs updates annually
    mandatory_population = merged_df['age_5_17'] * 0.20
    # Gap = How many MORE updates are needed (positive = shortage)
    merged_df['Update_Gap'] = np.maximum(0, mandatory_population - merged_df['Total_Updates'])
    
    # D. Vulnerability Index (Existing)
    total_activity_all = merged_df['Total_Enrollment'] + merged_df['Total_Updates']
    merged_df['Vulnerability_Index'] = (
        (merged_df['age_0_5'] * 1.5) + (merged_df['age_18_greater'] * 1.2)
    ) / (total_activity_all + 1)

    # E. Anomaly Flag (Relaxed to 1.5-sigma for realistic detection)
    # Calculate Mean & Std Dev per District to find local anomalies
    district_stats = merged_df.groupby('district')['Total_Updates'].agg(['mean', 'std']).reset_index()
    merged_df = pd.merge(merged_df, district_stats, on='district', suffixes=('', '_dist'))
    
    # Anomaly = Activity < (Mean - 1.5*StdDev) - More sensitive threshold
    merged_df['Is_Anomaly'] = merged_df['Total_Updates'] < (merged_df['mean'] - 1.5 * merged_df['std'])
    merged_df['Is_Anomaly'] = merged_df['Is_Anomaly'].astype(int)
    
    # Cleanup aux columns
    merged_df.drop(columns=['mean', 'std'], inplace=True)

    # F. Hardware Failure Rate (Simulated for "Auth Success Intelligence")
    # Associated with Anomaly: If Anomaly, Failure Rate is High.
    # Base Failure Rate: 2-5%. Anomaly Failure Rate: 15-25%.
    merged_df['Auth_Failure_Rate'] = np.random.uniform(2, 5, size=len(merged_df))
    merged_df.loc[merged_df['Is_Anomaly'] == 1, 'Auth_Failure_Rate'] = np.random.uniform(15, 25, size=merged_df[merged_df['Is_Anomaly'] == 1].shape[0])

    # Save Unified Dataset
    output_file = os.path.join(output_dir, 'aoee_unified_dataset.csv')
    merged_df.to_csv(output_file, index=False)
    
    print("-" * 30)
    print(f"Unified dataset saved to {output_file}")
    print(f"Shape: {merged_df.shape}")
    print("-" * 30)
    print("Top 5 Service Deserts (High Score):")
    print(merged_df.sort_values('Service_Desert_Score', ascending=False)[['district', 'pincode', 'Service_Desert_Score']].head(5))
    print("-" * 30)
    print("Top 5 Gender Disparity Areas (Low GUP):")
    print(merged_df[merged_df['Total_Updates'] > 10].sort_values('GUP_Index')[['district', 'GUP_Index']].head(5))

if __name__ == "__main__":
    process_aoee_data()
