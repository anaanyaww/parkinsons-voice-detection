"""
Quick test to verify installation is successful
Run this after installing requirements
"""

print("Testing imports...")

try:
    import streamlit
    print("✓ Streamlit:", streamlit.__version__)
except:
    print("✗ Streamlit failed")

try:
    import librosa
    print("✓ Librosa:", librosa.__version__)
except:
    print("✗ Librosa failed")

try:
    import xgboost
    print("✓ XGBoost:", xgboost.__version__)
except:
    print("✗ XGBoost failed")

try:
    import sklearn
    print("✓ Scikit-learn:", sklearn.__version__)
except:
    print("✗ Scikit-learn failed")

print("\n✅ All dependencies installed successfully!")
print("Run: streamlit run app.py")
