# 📁 Recommended Project Structure

## Clean Structure
```
zanalytics/
├── 📁 _dash_v_2/           # ← USE THIS AS MAIN
│   ├── 📄 zanflow_start.py  # New unified starter
│   ├── 📄 run_system.py     # Original starter
│   ├── 📄 api_server.py     # Flask API
│   ├── 📄 redis_server.py   # Redis utilities
│   ├── 📄 main.py           # Main dashboard
│   └── 📄 config_helper.py  # Configuration
│
├── 📁 pages/                # Dashboard pages
│   ├── 📄 1_  🌍 Market Intelligence.py
│   ├── 📄 2_ 📰 MACRO & NEWS.py
│   ├── 📄 3_ 🎓 Wyckoff.py
│   └── ... (other pages)
│
├── 📁 mt5/                  # MT5 Expert Advisors
│   ├── 📄 HttpsJsonSender_Simple.mq5
│   └── 📄 TickDataFileWriter.mq5
│
├── 📁 _archive/             # Old versions (create this)
│   └── ... (move old files here)
│
└── 📄 README.md
```

## Cleanup Steps
1. Create `_archive` folder
2. Move all duplicate/old files to `_archive`
3. Keep only the `_dash_v_2` version as main
4. Copy `zanflow_start.py` to `_dash_v_2`
5. Use `_dash_v_2` as your working directory

## Running the System
```bash
cd zanalytics-main/_dash_v_2
python zanflow_start.py
```
