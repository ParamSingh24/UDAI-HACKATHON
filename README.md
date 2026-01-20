# Aadhaar Operational Efficiency Engine (AOEE)

## 🏆 UIDAI Data Hackathon 2026 Submission

A comprehensive data analytics solution for optimizing Aadhaar enrollment and update operations across India.

## 📊 Key Features

- **Advanced Feature Engineering**: GUP Index, Service Desert Score, Update Gap Analysis
- **Machine Learning**: Random Forest Regressor for demand forecasting (R² = 0.43)
- **Interactive Dashboard**: Streamlit-based What-If simulator and ROI predictor
- **Visual Intelligence**: State-level heatmaps, anomaly detection, hardware hotspot identification

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Execution Steps
```bash
# 1. Clean raw data
python clean_aadhar_data.py

# 2. Process and engineer features
python process_aoee_data.py

# 3. Train ML model
python train_aoee_model.py

# 4. Generate analysis plots
python analysis_aoee.py

# 5. Launch dashboard
python -m streamlit run dashboard_aoee.py
```

## 📁 Project Structure

```
├── clean_aadhar_data.py          # Initial data cleaning
├── process_aoee_data.py          # Feature engineering
├── analysis_aoee.py              # Visualization generation
├── train_aoee_model.py           # ML model training
├── dashboard_aoee.py             # Interactive dashboard
├── requirements.txt              # Python dependencies
├── REPRODUCTION_GUIDE.md         # Detailed setup guide
└── aoee_output/
    ├── plots/                    # Generated visualizations
    └── aoee_model.pkl            # Trained model
```

## 🎯 Key Insights

- **Service Deserts**: Identified top 5 high-demand, low-supply districts
- **Gender Parity**: GUP Index tracking for inclusive enrollment
- **Operational Stability**: 2-sigma anomaly detection for center monitoring
- **ROI Optimization**: Mobile van deployment impact prediction

## 📈 Visualizations

- Trivariate heatmaps (State × Month × Success Rate/Volume)
- Hardware failure hotspot analysis
- Mandatory update backlog distribution
- 2-sigma anomaly detection charts

## 🛠️ Tech Stack

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Random Forest)
- **Visualization**: Matplotlib, Seaborn
- **Dashboard**: Streamlit
- **Data Validation**: Regex-based pincode validation, fuzzy state matching

## 👥 Team

Developed for UIDAI Data Hackathon 2026

## 📄 License

This project is submitted for the UIDAI Data Hackathon 2026.
