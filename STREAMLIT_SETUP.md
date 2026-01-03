# Streamlit Setup Guide for EasyWIF

## ✅ Configuration Complete

The app is now properly configured for Streamlit deployment. Here's what was set up:

### Files Structure
```
easywif/
├── app.py                    # Main application file
├── requirements.txt          # Python dependencies
├── .streamlit/
│   ├── config.toml          # Streamlit configuration (committed)
│   └── secrets.toml          # Password/secrets (NOT committed)
└── run_app.bat              # Windows batch file to run locally
```

## 🚀 Running Locally

### Option 1: Using the batch file
Double-click `run_app.bat` or run:
```bash
run_app.bat
```

### Option 2: Using command line
```bash
streamlit run app.py
```

### Option 3: Using Python module
```bash
python -m streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## ☁️ Deploying to Streamlit Cloud

1. **Push to GitHub**: Make sure your code is pushed to a GitHub repository
   - ✅ `app.py` is committed
   - ✅ `requirements.txt` is committed
   - ✅ `.streamlit/config.toml` is committed
   - ❌ `.streamlit/secrets.toml` is NOT committed (contains password)

2. **Connect to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file to: `app.py`
   - Click "Deploy"

3. **Set Secrets in Streamlit Cloud**:
   - In the Streamlit Cloud dashboard, go to "Settings" → "Secrets"
   - Add the password:
   ```toml
   password = "Intel2025"
   ```

## 📋 Requirements

All dependencies are listed in `requirements.txt`:
- streamlit>=1.28.0
- pandas>=2.0.0
- numpy>=1.24.0
- plotly>=5.15.0
- openpyxl>=3.1.0
- pyarrow>=12.0.0

## 🔐 Authentication

The app uses password protection. Default password: `Intel2025`

To change the password, update `.streamlit/secrets.toml` (locally) or the Secrets section in Streamlit Cloud.

## ✨ Latest Features

All recent changes are included:
- ✅ Updated changelog format ("impacted" instead of "increase/decrease")
- ✅ Fixed project view bar updates
- ✅ Fixed Resource_Type widget error
- ✅ Improved chart spacing
- ✅ Removed "Profile" from chart legend
- ✅ Reordered filters (Workstream before Project)
- ✅ Updated subtitle
- ✅ Combined Save and Export functionality
- ✅ Dark theme HTML export

## 🐛 Troubleshooting

If the app doesn't start:
1. Check that all dependencies are installed: `pip install -r requirements.txt`
2. Verify `.streamlit/config.toml` exists
3. Verify `.streamlit/secrets.toml` exists (for local runs)
4. Check for any error messages in the terminal

