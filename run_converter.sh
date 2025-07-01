cd /Users/tom/Documents/_trade/_exports/_tick
# Activate venv directly via python binary
/Users/tom/Documents/GitHub/venv2/bin/python \
  /Users/tom/Documents/_trade/_exports/_tick/convert_final_enhanced_smc_ULTIMATE.py \
  --output /Users/tom/Documents/GitHub/zanalytics/dashboard/data 

  /Users/tom/Documents/GitHub/venv2/bin/python \
  /Users/tom/Documents/_trade/_exports/_tick/convert_final_enhanced_smc_ULTIMATE.py \
  --output /Users/tom/Documents/GitHub/zanalytics/data 
  
/Users/tom/Documents/GitHub/venv2/bin/python \
  /Users/tom/Documents/_trade/_exports/_tick/ncOS_ultimate_microstructure_analyzer.py \
  --output /Users/tom/Documents/GitHub/zanalytics/dashboard/data


/Users/tom/Documents/GitHub/venv2/bin/python \
  /Users/tom/Documents/_trade/_exports/_tick/zanflow_microstructure_analyzer.py --folder /Users/tom/Documents/_trade/_exports/_tick/ \
  --limit 350 --json \
  --output /Users/tom/Documents/GitHub/zanalytics/dashboard/data --no-dashboard

/Users/tom/Documents/GitHub/venv2/bin/python \
  /Users/tom/Documents/_trade/_exports/_tick/ncOS_ultimate_microstructure_analyzer_DEFAULTS.py \
  --directory ./ \
  --no_compression \
  --output_dir /Users/tom/Documents/GitHub/zanalytics/dashboard/data \
  --max_candles 150 \
  --json_dir /Users/tom/Documents/GitHub/zanalytics/dashboard/data \
  --csv_dir /Users/tom/Documents/GitHub/zanalytics/dashboard/data \
    --file GBPUSD_M1_bars.csv

    /Users/tom/Documents/GitHub/venv2/bin/python \
  /Users/tom/Documents/_trade/_exports/_tick/ncOS_ultimate_microstructure_analyzer_DEFAULTS.py \
  --directory ./ \
  --no_compression \
  --output_dir /Users/tom/Documents/GitHub/zanalytics/dashboard/data/EURUSD \
  --max_candles 150 \
  --file EURUSD_M1_bars.csv

    /Users/tom/Documents/GitHub/venv2/bin/python \
  /Users/tom/Documents/_trade/_exports/_tick/ncOS_ultimate_microstructure_analyzer_DEFAULTS.py \
  --directory ./ \
  --no_compression \
  --output_dir /Users/tom/Documents/GitHub/zanalytics/dashboard/data/XAUUSD \
  --max_candles 150 \
  --file XAUUSD_M1_bars.csv

      /Users/tom/Documents/GitHub/venv2/bin/python \
  /Users/tom/Documents/_trade/_exports/_tick/ncOS_ultimate_microstructure_analyzer_DEFAULTS.py \
  --directory ./ \
  --no_compression \
  --output_dir /Users/tom/Documents/GitHub/zanalytics/dashboard/data/BTCUSD \
  --max_candles 150 \
  --file BTCUSD_M1_bars.csv \
  --json_dir /Users/tom/Documents/GitHub/zanalytics/dashboard/data/BTCUSD

      /Users/tom/Documents/GitHub/venv2/bin/python \
 /Users/tom/Documents/_trade/_exports/_tick/ncos_enhanced_analyzer.py \
    --file XAUUSD_M1_bars.csv \
    --timeframes_parquet M1,M5,M15,M30,H1,H4,D1 \
    --max_candles 500 \
    --file XAUUSD_M1_bars.csv \
    --output_dir ./midas_analysis \
    --no_compression 


 cp /Users/tom/Documents/_trade/_exports/_tick/*_bars.csv /Users/tom/Documents/GitHub/zanalytics/dashboard/data/

  

'''
  --file ETHUSD.csv \



 --file XAUUSD_M1_bars.csv 
  --json_dir /Users/tom/Documents/GitHub/zanalytics/dashboard/data \
  --csv_dir /Users/tom/Documents/GitHub/zanalytics/dashboard/data \

/Users/tom/Documents/GitHub/venv2/bin/python \
  /Users/tom/Documents/_trade/_exports/_tick/ncos_envcOS_ultimate_microstructure_analyzer_DEFAULTS.py \
  --file ETHUSD.csv \
  --no_compression \
  --output_dir /Users/tom/Documents/GitHub/zanalytics/data/ETHUSD

  /Users/tom/Documents/GitHub/venv2/bin/python \
  /Users/tom/Documents/_trade/_exports/_tick/ncOS_ultimate_microstructure_analyzer_DEFAULTS.py \
  --file BTCUSD_ticks.csv \
  --no_compression  \
  --output_dir /Users/tom/Documents/GitHub/zanalytics/data/BTCUSD

/Users/tom/Documents/GitHub/venv2/bin/python \
  /Users/tom/Documents/_trade/_exports/_tick/ncOS_ultimate_microstructure_analyzer_DEFAULTS.py \
  --file XRPUSD_ticks.csv \
  --no_compression \
  --output_dir /Users/tom/Documents/GitHub/zanalytics/data/XRPUSD
  '''