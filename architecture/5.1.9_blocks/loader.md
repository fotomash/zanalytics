# **/zanalytics/loader/ \- Data Ingestion Modules**

This directory contains Python modules responsible for loading raw market data from various sources and formats into the Zanzibar Analytics system. The goal of this layer is to abstract the complexities of different data source formats and provide a consistent, preprocessed data stream (typically pandas DataFrames or lists of ZBAR™ objects) to the downstream analytical engines.

## **Purpose**

* **Data Acquisition:** Connect to and read data from files (CSV, TSV, etc.) and, in the future, potentially from APIs or live data streams.  
* **Format Normalization:** Handle different delimiters, encodings, header names, and timestamp formats from various data providers (e.g., MetaTrader exports, Sierra Chart exports).  
* **Initial Preprocessing:** Perform essential cleaning steps like standardizing column headers, ensuring correct data types (especially for timestamps, price, and volume), and handling combined date/time columns.  
* **Microstructure Construction (Tick Processing):** Aggregate raw tick data into meaningful bar structures (ZBAR™ objects), including the construction of intra-bar price ladders and calculation of accurate bar delta.

## **Key Modules**

* **csv\_loader.py (Version: 4.0 \- Policy Aligned)**  
  * **Responsibility:** Loads M1 (or other timeframe) bar data from CSV or TSV files.  
  * **Functionality:**  
    * Auto-detects delimiters (comma or tab) and file encodings (using chardet).  
    * Uses column\_map profiles defined in config/config.yaml to map source column headers (e.g., \<DATE\>, \<TIME\>, \<OPEN\>, \<VOL\>) to standard internal field names (timestamp, open, volume, etc.).  
    * Handles combined date and time columns, merging them into a single, timezone-aware (UTC) timestamp column.  
    * Performs basic header cleaning (stripping \<\> and extra spaces).  
    * Outputs a pandas DataFrame with standardized column names, ready for mapping to ZBAR™ objects by zanzibar.utils.zbar\_mapper.  
  * **Policy Alignment:** Adheres to the data\_ingestion\_policy.md regarding format detection, column mapping, and timestamp handling for bar data.  
* **tick\_processor.py (Version: 2.0 \- Implemented Aggregation & Metrics)**  
  * **Responsibility:** Processes raw tick-level data into fully populated ZBAR™ objects.  
  * **Functionality:**  
    * load\_tick\_data(): Loads raw tick data from files (CSV/TSV), standardizes column names (for timestamp, price, volume, and optional bid, ask, flags) based on config.yaml profiles, and ensures timestamps are UTC.  
    * aggregate\_ticks\_to\_zbars(): Takes the standardized tick DataFrame and uses the internal TickAggregator class to:  
      * Group ticks into bars based on a specified interval (e.g., "1min", future: volume/tick-based bars).  
      * Construct the intra-bar price\_ladder (a Dict\[float, MarketOrderData\]) for each ZBAR™.  
      * Infer trade aggression (buyer vs. seller initiated) for each tick using configurable logic (use\_l1\_quote, use\_flags, lee\_ready\_simple) to accurately populate bid\_volume, ask\_volume, and delta within the price\_ladder.  
      * Calls ZBAR.calculate\_derived\_metrics\_from\_ladder() to compute the final bar\_delta, poc\_price, poi\_price, bid\_volume\_total, and ask\_volume\_total for each ZBAR™.  
  * **Output:** A list of fully populated ZBAR objects.  
  * **Strategic Importance:** This module is critical for unlocking deep microstructure analysis and providing accurate inputs for VSA and indicators like CVD.

## **Configuration**

The behavior of these loaders (especially csv\_loader.py and tick\_processor.py) is heavily driven by profiles defined in config/config.yaml under the data\_loader and tick\_processor sections respectively. This includes:

* active\_profile: Specifies which named profile to use.  
* profiles: A dictionary where each key is a profile name. Each profile contains:  
  * delimiter: For file parsing.  
  * encoding: For file reading.  
  * column\_map: Crucial for mapping source column names to standard internal names (e.g., timestamp\_source, price\_source, volume\_source, bid\_source, ask\_source, flags\_source).  
  * timestamp\_format: Optional, for pd.to\_datetime if inference fails.  
  * default\_tz: For tick\_processor.  
  * default\_bar\_interval: For tick\_processor.  
  * tick\_side\_logic: For tick\_processor to determine how to infer trade aggression.  
  * flags\_buy\_value, flags\_sell\_value: Used if tick\_side\_logic is use\_flags.

## **Future Enhancements**

* Support for additional data formats (Parquet, JSON, direct API connections).  
* More sophisticated error handling and reporting for data quality issues.  
* Integration with a ZBarRepository for incremental loading and deduplication based on the storage configuration in config.yaml.