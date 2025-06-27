cd /Users/tom/Documents/_trade/_exports/_tick
# Activate venv directly via python binary
/Users/tom/venvs/ncos_env/bin/python \
  /Users/tom/Documents/_trade/_exports/_tick/convert_final_enhanced_smc_ULTIMATE.py \
  --output /Users/tom/Documents/_trade/_exports/_tick/data  --bar-limit 300 --no-tick
  
/Users/tom/venvs/ncos_env/bin/python \
  /Users/tom/Documents/_trade/_exports/_tick/ \
  --1min-bars 500 --5min-bars 500 --15min-bars 500 --30min-bars 600 \
  --output /Users/tom/Documents/_trade/_exports/_tick/data

/Users/tom/venvs/ncos_env/bin/python \
  /Users/tom/Documents/_trade/_exports/_tick/zanflow_microstructure_analyzer.py --folder /Users/tom/Documents/_trade/_exports/_tick/ \
  --limit 150 --json \
  --output /Users/tom/Documents/_trade/_exports/_tick/data --no-dashboard