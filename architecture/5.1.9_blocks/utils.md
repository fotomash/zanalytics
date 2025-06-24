# **/zanalytics/utils/ \- Shared Utilities**

This directory contains utility modules that provide common, reusable functions and classes supporting various parts of the Zanzibar Analytics system. These utilities help maintain clean code, reduce redundancy, and encapsulate specific functionalities like data mapping, schema validation, and indicator calculations.

## **Purpose**

* **Code Reusability:** Offer a central place for helper functions and classes that are used by multiple modules (e.g., data loaders, analysis engines).  
* **Encapsulation:** Isolate specific tasks like data validation or complex calculations into manageable units.  
* **Maintainability:** Make the codebase easier to understand and modify by centralizing common logic.

## **Key Modules**

* **schema\_validator.py (Version: 1.0)**  
  * **Responsibility:** Validates data dictionaries against the ZBAR™ schema requirements before ZBAR™ object instantiation.  
  * **Functionality:**  
    * validate\_zbar\_dict(): Checks for the presence of required fields (timestamp, open, high, low, close, volume), correct data types (datetime, numeric), and logical consistency (e.g., Low \<= High, Volume \>= 0).  
    * Configurable behavior, such as allow\_zero\_volume, through parameters passed from the main config.yaml (via the validator\_rules section).  
  * **Usage:** Called by zbar\_mapper.py to ensure data integrity before creating ZBar objects.  
* **zbar\_mapper.py (Version: 3.0 \- Policy Aligned)**  
  * **Responsibility:** Maps preprocessed pandas DataFrames or dictionaries (with standardized ZBAR™ field names) into ZBar objects.  
  * **Functionality:**  
    * map\_dict\_to\_zbar(): Takes a dictionary (representing a single bar with standardized column names), performs final type coercions, ensures the timestamp is timezone-aware (UTC), and then calls validate\_zbar\_dict() before instantiating a ZBar object.  
    * map\_dataframe\_to\_zbars(): Iterates through a DataFrame (expected to have standard ZBAR™ columns like timestamp, open, etc., as prepared by a loader module) and uses map\_dict\_to\_zbar() to convert each row into a ZBar object.  
  * **Usage:** Called by data loaders (e.g., csv\_loader.py or the future tick\_processor.py after its aggregation step) to transform tabular data into the system's core ZBar objects.  
* **indicators.py (Version: 4.0 \- Implemented DSS, OBV, CVD)**  
  * **Responsibility:** Provides functions for calculating a wide range of technical indicators.  
  * **Functionality:**  
    * Includes implementations for standard indicators: SMA, EMA, RSI (Wilder's), MACD, Bollinger Bands, VWAP (rolling/cumulative).  
    * Includes implementations for "Silent but Present" indicators: DSS Bressert, On-Balance Volume (OBV), and Cumulative Volume Delta (CVD). The CVD calculation relies on an input Delta series (typically bar\_delta from ZBAR™ objects).  
    * add\_indicators\_to\_df(): A master function that takes a DataFrame (typically OHLCV data, optionally with a 'Delta' column) and adds new columns for all indicators specified as active: true in the indicators section of config.yaml. Indicator parameters (e.g., window, span, period) are also sourced from this configuration.  
  * **Usage:** Called by the main pipeline (run\_test\_pipeline.py) after data loading to enrich the primary DataFrame with indicator values before it's (potentially) mapped to ZBARs or used directly by analysis engines.

## **General Principles**

* **Cohesion:** Each utility module should focus on a specific set of related tasks.  
* **Low Coupling:** Utilities should have minimal dependencies on other high-level application modules, primarily operating on standard data types like pandas DataFrames, Series, or basic Python dictionaries.  
* **Testability:** Utility functions should be designed to be easily unit-testable with clear inputs and expected outputs.