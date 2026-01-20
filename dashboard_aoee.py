import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="Aadhaar Impact Engine (AOEE)", layout="wide")

# Paths
BASE_PATH = r"c:\Users\param\OneDrive\Desktop\Data Hackathon UDAI"
DATA_PATH = os.path.join(BASE_PATH, 'aoee_output', 'aoee_unified_dataset.csv')
MODEL_PATH = os.path.join(BASE_PATH, 'aoee_output', 'aoee_model.pkl')
FEATURES_PATH = os.path.join(BASE_PATH, 'aoee_output', 'model_features.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features

def main():
    st.title("🇮🇳 Aadhaar Operational Efficiency Engine (AOEE)")
    st.markdown("### Strategic Impact & What-If Planning Dashboard (Advanced)")

    try:
        df = load_data()
        model, model_features = load_model()
    except Exception as e:
        st.error(f"Error loading data/model: {e}")
        st.write("Please run 'process_aoee_data.py' and 'train_aoee_model.py' first.")
        return

    # Sidebar: Global Controls
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "What-If Simulator", "ROI Predictor", "Auth Success Intelligence"])

    if page == "Overview":
        st.header("Executive Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Biometric Updates", f"{df['Biometric_Updates'].sum():,.0f}")
        col2.metric("Avg GUP Index (Gender Parity)", f"{df['GUP_Index'].mean():.2f}")
        col3.metric("Service Desert Score (Avg)", f"{df['Service_Desert_Score'].mean():.0f}")
        col4.metric("Avg Update Gap", f"{df['Update_Gap'].mean():,.0f}")

        st.subheader("Service Desert Map (High Demand, Low Supply)")
        # Filter High Service Desert Score
        deserts = df.sort_values('Service_Desert_Score', ascending=False).head(10)
        st.dataframe(deserts[['state', 'district', 'pincode', 'Service_Desert_Score', 'Update_Gap']])

    elif page == "What-If Simulator":
        st.header("🔮 Impact Prediction Engine (Random Forest)")
        st.markdown("Simulate the impact of **Mobile Van Deployment** on reducing the **Update Gap**.")

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Simulation Parameters")
            
            # Create State + District combined selector
            df['state_district'] = df['state'] + ' - ' + df['district']
            location_options = sorted(df['state_district'].unique())
            
            selected_location = st.selectbox("Select Location (State - District)", location_options)
            
            # Extract state and district
            selected_state = selected_location.split(' - ')[0]
            selected_district = selected_location.split(' - ')[1]
            
            # Slider for Mobile Van Deployment
            vans = st.slider("Deploy Mobile Vans (Units)", 0, 50, 5)
            capacity_per_van = 500 
            
        with col2:
            st.subheader("Predicted Outcome")
            
            # Filter Data by both state and district
            dist_df = df[(df['state'] == selected_state) & (df['district'] == selected_district)].copy()
            if dist_df.empty:
                st.warning("No data for selected location.")
            else:
                current_gap = dist_df['Update_Gap'].sum()
                current_updates = dist_df['Total_Updates'].sum()
                
                # Apply Boost
                added_capacity = vans * capacity_per_van
                new_updates = current_updates + added_capacity
                
                # Recalculate Features for Model
                dist_df['Service_Desert_Score'] = dist_df['Total_Enrollment'] / (new_updates + 1)
                
                # Predict Demand Change (using Random Forest)
                X_pred = dist_df[model_features].fillna(0)
                predicted_demand = model.predict(X_pred).sum()
                
                # Update Gap Reduction
                gap_reduction = min(added_capacity, max(0, current_gap))
                new_gap = max(0, current_gap - gap_reduction)
                
                st.metric(label="Projected Update Gap", value=f"{new_gap:,.0f}", delta=f"-{gap_reduction:,.0f}", delta_color="inverse")
                st.metric(label="Modeled Demand (Biometric)", value=f"{predicted_demand:,.0f}")

                # Improved Chart
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(["Current\nGap", "Projected\nGap"], [current_gap, new_gap], 
                              color=['#e74c3c', '#27ae60'], width=0.6, edgecolor='black', linewidth=1.5)
                ax.set_ylabel("Update Gap Volume", fontsize=12, fontweight='bold')
                ax.set_title(f"Impact of {vans} Mobile Vans in {selected_state}", fontsize=14, fontweight='bold')
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                
                # Add value labels
                for i, (bar, v) in enumerate(zip(bars, [current_gap, new_gap])):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{v:,.0f}',
                           ha='center', va='bottom', fontsize=11, fontweight='bold')
                
                st.pyplot(fig)
                plt.close(fig)

    elif page == "ROI Predictor":
        st.header("💰 Intervention ROI Predictor")
        st.markdown("Calculate **Cost per New Enrollment** vs. **Social Benefit Value**.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Cost Inputs")
            cost_per_van = st.number_input("Monthly Cost per Van (₹)", value=50000)
            enrollments_per_van = st.number_input("Est. Enrollments per Van", value=150)
            
        with col2:
            st.subheader("Benefit Inputs")
            subsidy_saving_per_person = st.number_input("Subsidy Saving / Person (₹)", value=2000)
            multiplier = st.slider("Economic Multiplier", 1.0, 3.0, 1.5)
            
        # Calculation
        total_cost = cost_per_van
        total_benefit = enrollments_per_van * subsidy_saving_per_person * multiplier
        
        roi = ((total_benefit - total_cost) / total_cost) * 100
        cpa = total_cost / enrollments_per_van if enrollments_per_van else 0
        
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("ROI", f"{roi:.1f}%", delta="Positive" if roi>0 else "Negative")
        c2.metric("Cost Per Acquisition (CPA)", f"₹{cpa:,.0f}")
        c3.metric("Net Social Value", f"₹{(total_benefit - total_cost):,.0f}")

    elif page == "Auth Success Intelligence":
        st.header("🔐 Authentication Success Intelligence")
        st.markdown("Immediate Hardware Inspection List (Based on ML Anomalies).")
        
        # Filter Anomalies (Is_Anomaly = 1)
        anomalies = df[df['Is_Anomaly'] == 1].copy()
        
        if anomalies.empty:
            st.success("✅ No critical anomalies detected! All systems operating normally.")
        else:
            st.warning(f"⚠️ Found {len(anomalies)} anomalous records requiring attention")
            
            # Show top 20 anomalies
            st.subheader("Top 20 Hardware Failure Hotspots")
            top_anomalies = anomalies[['state', 'district', 'pincode', 'Auth_Failure_Rate', 'Total_Updates']].sort_values('Auth_Failure_Rate', ascending=False).head(20)
            st.dataframe(top_anomalies, use_container_width=True)
            
            # State-level aggregation for better visualization
            st.subheader("State-Level Failure Rate Analysis")
            st.caption("Average authentication failure rate by state (anomalous regions only)")
            
            state_failures = anomalies.groupby('state')['Auth_Failure_Rate'].agg(['mean', 'count']).reset_index()
            state_failures.columns = ['State', 'Avg Failure Rate (%)', 'Anomaly Count']
            state_failures = state_failures.sort_values('Avg Failure Rate (%)', ascending=False)
            
            # Create better visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.barh(state_failures['State'], state_failures['Avg Failure Rate (%)'], 
                          color='#e74c3c', edgecolor='black', linewidth=0.8)
            ax.set_xlabel('Average Failure Rate (%)', fontsize=12, fontweight='bold')
            ax.set_ylabel('State', fontsize=12, fontweight='bold')
            ax.set_title('Authentication Failure Rates by State (Anomalous Regions)', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, state_failures['Avg Failure Rate (%)'])):
                ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                       ha='left', va='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Show summary table
            st.subheader("State Summary")
            st.dataframe(state_failures, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
