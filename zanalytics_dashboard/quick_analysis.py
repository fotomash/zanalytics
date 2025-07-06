#!/usr/bin/env python3
import sys
sys.path.append('data_pipeline')

try:
    from data_enricher import DataEnricher

    config = {
        'data_path': './data',
        'refresh_interval': 60
    }

    print("Running analysis...")
    enricher = DataEnricher(config)
    enricher.process_all_data()
    print("Complete!")

except Exception as e:
    print(f"Error: {e}")
    print("Run setup_system.py first")
