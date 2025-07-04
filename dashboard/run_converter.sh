
cp -R /Volumes/_bars/* /Users/tom/Documents/_trade/_exports/_tick/_bars
cp -R /Volumes/_processed/* /Users/tom/Documents/_trade/_exports/_tick/midas_analysis
cp -R /Volumes/_ticks/* /Users/tom/Documents/_trade/_exports/_tick/_raw
cp -R /Volumes/out/* /Users/tom/Documents/_trade/_exports/_tick/parquet
cp -R /Volumes/_tick_csv/* /Users/tom/Documents/_trade/_exports/_tick/_raw

'''
    finnhub_api_key = "d1fiar1r01qig3h21l20d1fiar1r01qig3h21l2g"
    newsapi_key = "713b3bd82121482aaa0ecdc9af77b6da"
    trading_economics_api_key = "1750867cdfc34c6:288nxdz64y932qq"
    fred_api_key = "6a980b8c2421503564570ecf4d765173"
    data_directory = "/Users/tom/Documents/GitHub/zanalytics/dashboard/data"
    raw_data_directory = "/Users/tom/Documents/_trade/_exports/_tick/_raw"
    JSONdir = "/Users/tom/Documents/_trade/_exports/_tick/midas_analysis"
    bar_data_directory = "/Users/tom/Documents/_trade/_exports/_tick/_bars"
    data_path = "/Users/tom/Documents/_trade/_exports/_tick"
    BAR_DATA_DIR = "/Users/tom/Documents/_trade/_exports/_tick/_bars"
    PARQUET_DATA_DIR = "/Users/tom/Documents/_trade/_exports/_tick/parquet"
    raw_data = "/Users/tom/Documents/_trade/_exports/_tick"
    enriched_data = "/Users/tom/Documents/_trade/_exports/_tick/parquet" 
    tick_data = "/Users/tom/Documents/_trade/_exports/_tick/_raw"


cd /Users/tom/Documents/_trade/_exports/_tick
# Activate venv directly via python binary
/Users/tom/Documents/GitHub/venv2/bin/python \
  /Users/tom/Documents/_trade/_exports/_tick/convert_final_enhanced_smc_ULTIMATE.py 
  --output /Users/tom/Documents/GitHub/zanalytics/dashboard/data 

  /Users/tom/Documents/GitHub/venv2/bin/python \
  /Users/tom/Documents/_trade/_exports/_tick/convert_final_enhanced_smc_ULTIMATE.py 
  --output /Users/tom/Documents/GitHub/zanalytics/data 
  
/Users/tom/Documents/GitHub/venv2/bin/python \
  /Users/tom/Documents/_trade/_exports/_tick/ncOS_ultimate_microstructure_analyzer.py 
  --output /Users/tom/Documents/GitHub/zanalytics/dashboard/data

/Users/tom/Documents/GitHub/venv2/bin/python \
  /Users/tom/Documents/_trade/_exports/_tick/zanflow_microstructure_analyzer.py --folder /Users/tom/Documents/_trade/_exports/_tick/ \
  --limit 250 --json \
  --output /Users/tom/Documents/GitHub/zanalytics/dashboard/data --no-dashboard

/Users/tom/Documents/GitHub/venv2/bin/python \
  /Users/tom/Documents/_trade/_exports/_tick/ncOS_ultimate_microstructure_analyzer.py \
  --directory ./ \
  --no_compression \
  --output_dir /Users/tom/Documents/GitHub/zanalytics/dashboard/data \
  --max_candles 200 \
  --csv_dir /Users/tom/Documents/GitHub/zanalytics/dashboard/data \
  --file GBPUSD_M1_bars.csv 

    /Users/tom/Documents/GitHub/venv2/bin/python \
  /Users/tom/Documents/_trade/_exports/_tick/ncOS_ultimate_microstructure_analyzer_DEFAULTS.py \
  --directory ./ \
  --no_compression \
  --output_dir /Users/tom/Documents/GitHub/zanalytics/dashboard/data/EURUSD \
  --max_candles 200 \
  --file EURUSD_M1_bars.csv 

cd /Users/tom/Documents/_trade/_exports/_tick 

    /Users/tom/Documents/GitHub/venv2/bin/python \
  /Users/tom/Documents/_trade/_exports/_tick/ncOS_ultimate_microstructure_analyzer_DEFAULTS.py \
  --directory ./ \
  --no_compression \
  --output_dir /Users/tom/Documents/GitHub/zanalytics/dashboard/data/XAUUSD \
  --max_candles 200 \
  --file XAUUSD_M1_bars.csv

      /Users/tom/Documents/GitHub/venv2/bin/python \
 /Users/tom/Documents/_trade/_exports/_tick/ncos_enhanced_analyzer.py \
    --file XAUUSD_M1_bars.csv \
    --timeframes_parquet M1,M5,M15,M30,H1,H4,D1 \
    --max_candles 250 \
    --file XAUUSD_M1_bars.csv \
    --output_dir ./midas_analysis \
    --no_compression 


 cp /Users/tom/Documents/_trade/_exports/_tick/*_bars.csv /Users/tom/Documents/GitHub/zanalytics/dashboard/data/


  --file ETHUSD.csv \

 /Users/tom/Documents/_trade/_exports/_tick/
 
 python ncos_enhanced_analyzer.py --timeframes_parquet 1min,5min,15min,30min,1h,4h,1d \
    --max_candles 250 \
    --output_dir "/Users/tom/Documents/_tick_data/_processed" \
    --no_compression \
    --directory "/Users/tom/Library/Application Support/net.metaquotes.wine.metatrader5/drive_c/Program Files/MetaTrader 5/MQL5/Files/_tick_data/_bars" \
    --json_dir  /Users/tom/Documents/_tick_data/_processed/json 




      /Users/tom/Documents/GitHub/venv2/bin/python \
  /Users/tom/Documents/_trade/_exports/_tick/ncOS_ultimate_microstructure_analyzer_DEFAULTS.py \
  --directory ./ \
  --no_compression \
  --output_dir /Users/tom/Documents/GitHub/zanalytics/dashboard/data/BTCUSD \
  --max_candles 200 \
  --file BTCUSD_M1_bars.csv \
  --json_dir /Users/tom/Documents/GitHub/zanalytics/dashboard/data/BTCUSD 



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