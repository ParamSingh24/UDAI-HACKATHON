import pandas as pd
import os
import glob

def clean_data():
    base_path = r"c:\Users\param\OneDrive\Desktop\Data Hackathon UDAI"
    
    # Dataset configurations
    datasets = {
        'biometric': {
            'path': os.path.join(base_path, 'api_data_aadhar_biometric'),
            'age_cols': ['bio_age_5_17', 'bio_age_17_']
        },
        'demographic': {
            'path': os.path.join(base_path, 'api_data_aadhar_demographic'),
            'age_cols': ['demo_age_5_17', 'demo_age_17_']
        },
        'enrollment': {
            'path': os.path.join(base_path, 'api_data_aadhar_enrolment'),
            'age_cols': ['age_0_5', 'age_5_17', 'age_18_greater']
        }
    }
    
    output_dir = os.path.join(base_path, 'cleaned_data')
    os.makedirs(output_dir, exist_ok=True)
    
    for name, config in datasets.items():
        print(f"Processing {name} dataset...")
        
        # 1. Load Data
        csv_files = glob.glob(os.path.join(config['path'], "*.csv"))
        if not csv_files:
            print(f"Warning: No CSV files found for {name} in {config['path']}")
            continue
            
        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                
        if not dfs:
            continue
            
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"  Combined shape: {combined_df.shape}")
        
        # 2. Standardize Location
        if 'state' in combined_df.columns:
            combined_df['state'] = combined_df['state'].astype(str).str.strip().str.title()
        if 'district' in combined_df.columns:
            combined_df['district'] = combined_df['district'].astype(str).str.strip().str.title()
            
        # 3. Handle Missing Values
        for col in config['age_cols']:
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].fillna(0).astype('int64') # Ensure integer type after fillna
            else:
                 print(f"  Warning: Column {col} missing in {name}")

        # 4. Feature Engineering: Total_Activity
        # Check if all age cols exist before summing
        available_age_cols = [col for col in config['age_cols'] if col in combined_df.columns]
        if available_age_cols:
             combined_df['Total_Activity'] = combined_df[available_age_cols].sum(axis=1)
        
        # Feature Engineering: Date
        if 'date' in combined_df.columns:
            try:
                # Assuming format is DD-MM-YYYY based on previews (e.g. 01-03-2025)
                combined_df['date'] = pd.to_datetime(combined_df['date'], format='%d-%m-%Y', errors='coerce')
                combined_df['Year'] = combined_df['date'].dt.year
                combined_df['Month'] = combined_df['date'].dt.month
            except Exception as e:
                print(f"  Error converting date column: {e}")
        
        # 5. Save Output
        output_file = os.path.join(output_dir, f"cleaned_{name}.csv")
        combined_df.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")
        
        # Verification Output
        print(f"  Preview:")
        print(combined_df.head())
        print("-" * 30)

if __name__ == "__main__":
    clean_data()
