# Interactive Command Center - Quick Start

## 🚀 5-Minute Setup

### 1. Install
```bash
cd /path/to/your/zanalytics
unzip zanalytics_command_center.zip
cd zanalytics_command_center
python integrate_command_center.py
```

### 2. Start Services
```bash
# Terminal 1 - API Service
python zanalytics_api_service.py

# Terminal 2 - Dashboard
streamlit run dashboards/Home.py
```

### 3. Access Strategy Editor
- Open browser to http://localhost:8501
- Click "🔧 Strategy Editor" in sidebar

### 4. Edit Your First Strategy
1. Select a strategy from dropdown
2. Change a parameter (e.g., position_size)
3. Click "✅ Validate"
4. Click "💾 Save Changes"

## 🎯 Key Features at a Glance

| Feature | Description | Shortcut |
|---------|-------------|----------|
| Visual Editor | Form-based editing | Default mode |
| YAML Editor | Direct code editing | Toggle radio button |
| Validation | Check before save | ✅ button |
| Backups | Auto-saved versions | 📋 button |
| Templates | Quick start strategies | ➕ button |

## 📝 Example: Modify Risk Settings

1. Select your strategy
2. Go to "💰 Risk Management" tab
3. Adjust position size (e.g., 0.02 = 2%)
4. Set max positions (e.g., 2)
5. Validate and Save

## ⚡ Pro Tips

- **Ctrl+S** doesn't work - use Save button
- **Validate first** - catches errors before save
- **Use templates** - faster than starting blank
- **Check backups** - easy rollback if needed

---

Ready to transform your trading dashboard into a command center!
