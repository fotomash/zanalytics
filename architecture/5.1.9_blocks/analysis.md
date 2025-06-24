# **/zanalytics/analysis/ \- Core Analytical Engines**

This directory houses the primary analytical engines of the Zanzibar Analytics system. These modules are responsible for interpreting market data (primarily ZBAR™ objects and enriched DataFrames) to identify patterns, structures, and generate insights based on various methodologies like Wyckoff, VSA, SMC, and indicator analysis.

## **Purpose**

* **Market Interpretation:** Apply sophisticated analytical techniques to understand market behavior, identify potential trading opportunities, and determine market context.  
* **Signal Generation Primitives:** While not necessarily generating final trade signals, these engines produce the core analytical outputs (e.g., detected Wyckoff events, market phase, POIs, liquidity events) that downstream modules (like ML validation or execution logic) will consume.  
* **Methodology Implementation:** Encapsulate the specific rules, algorithms, and logic for each analytical approach (Wyckoff, SMC, etc.) in a modular fashion.

## **Key Sub-Modules**

* **wyckoff/**:  
  * **Responsibility:** Implements the full Wyckoff method analysis.  
  * **Key Components:**  
    * event\_detector.py: (Contains ZBar and MarketOrderData models). Detects individual Wyckoff events (PS, SC, AR, ST, Spring, UT, etc.) using detailed Price, Spread, Volume, and Bar Delta analysis (VSA). Event detection is driven by configurable thresholds.  
    * state\_machine.py: The WyckoffStateMachine tracks the sequence of detected events and uses contextual ZBAR™ data (like Trading Range boundaries) to determine the current Wyckoff market phase (Accumulation/Distribution A-E). Transition rules are configurable.  
  * **Further Details:** See /zanzibar/analysis/wyckoff/README.md.  
* **(Future) liquidity/**:  
  * **Responsibility:** Focus on advanced liquidity analysis beyond simple bar delta.  
  * **Potential Components:** Liquidity sweep detectors (building on liquidity\_sweep\_detector.py from zanalytics\_1.txt), order flow imbalance analyzers (using price\_ladder from ZBARs), absorption detectors.  
* **(Future) structure/**:  
  * **Responsibility:** Broader market structure analysis, potentially incorporating elements of Smart Money Concepts (SMC) that are not directly covered by the Wyckoff engine.  
  * **Potential Components:** Automated CHoCH/BOS detection, Fair Value Gap (FVG) identification, Order Block (OB) tagging, Point of Interest (POI) manager (refining poi\_manager\_smc.py concepts from zanalytics\_1.txt).  
* **(Future) features/**:  
  * **Responsibility:** Engineering features from raw data, indicator outputs, and structural analysis for consumption by Machine Learning models.  
  * **Potential Components:** Factor calculation library, feature scaling and selection utilities.  
* **(Future) profile\_engine/**:  
  * **Responsibility:** Advanced Volume Profile and Market Profile analysis beyond the basic POC/POI in ZBARs.  
  * **Potential Components:** Composite profile generation, Value Area (VAH/VAL) calculation over dynamic ranges, TPO chart logic.

## **Data Flow**

Modules in this directory typically consume:

* Lists of ZBar objects (from zanzibar.utils.zbar\_mapper).  
* Enriched pandas DataFrames (from zanzibar.utils.indicators).

They output:

* Dictionaries of detected events and their locations.  
* Market phase classifications.  
* Identified Points of Interest (POIs) with associated scores/context.  
* Other structured analytical metadata.

This output is then consumed by the main orchestrator, ML validation modules, or future execution logic.