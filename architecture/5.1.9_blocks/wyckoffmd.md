# **/zanatytics/analysis/wyckoff/ \- Wyckoff Method Analysis Engine**

This module is dedicated to implementing the Wyckoff method for market analysis. It aims to identify key Wyckoff events, define trading ranges, and determine the current market phase (Accumulation, Distribution, Markup, Markdown) based on price action, volume, spread, and intra-bar delta dynamics.

## **Purpose**

* **Structural Analysis:** Provide a foundational understanding of the market's structural position according to Wyckoff principles.  
* **Event Identification:** Pinpoint significant Wyckoff events that signal potential shifts in supply and demand.  
* **Phase Determination:** Classify the market into one of the Wyckoff phases (A, B, C, D, E for both Accumulation and Distribution schematics).  
* **Context Generation:** Supply crucial context (e.g., Trading Range boundaries, current phase, key support/resistance areas derived from Wyckoff events) to other analytical modules and downstream decision-making processes.

## **Key Components**

* **event\_detector.py (Version: 5.0 \- ZBar with Heuristic Delta)**  
  * **Defines Data Models:** Contains the core ZBar and MarketOrderData dataclass definitions, which are fundamental to the entire Zanzibar system. The ZBar includes OHLCV, a price\_ladder (for tick-derived microstructure), and methods to calculate bar\_delta (either heuristically from OHLCV or accurately from the price\_ladder), Point of Control (poc\_price), and Point of Interest (poi\_price).  
  * **Event Detection Logic:** Implements functions to detect a comprehensive suite of Wyckoff events:  
    * **Accumulation:** Preliminary Support (PS), Selling Climax (SC), Automatic Rally (AR\_acc), Secondary Test (ST\_Acc, ST\_Acc\_Weak), Spring (Spring, Spring\_Weak).  
    * **Distribution (Partially Implemented):** Upthrust (UT, UT\_Weak). (PSY, BC, UTAD, SOW, LPSY are planned).  
  * **VSA & Delta Integration:** Event detection heavily relies on Volume Spread Analysis (VSA) principles, comparing current bar's price, volume, and spread characteristics against historical averages and prior significant bars. Crucially, it incorporates ZBar.bar\_delta to confirm the aggressive buying or selling pressure associated with events.  
  * **Configuration:** All detection thresholds and lookback periods are configurable via config/config.yaml.  
* **state\_machine.py (Version: 3.1 \- Accumulation D & E Logic)**  
  * **Purpose:** Implements the WyckoffStateMachine class, which tracks the sequence of detected Wyckoff events and uses this sequence along with contextual data from ZBar objects to determine the current market phase.  
  * **Context-Awareness:** The state machine receives the full list of ZBar objects (zbars\_context) and uses it to:  
    * Determine the overall schematic type (Accumulation or Distribution).  
    * Define and update Trading Range (TR) boundaries (support and resistance).  
    * Validate if events occur in logically consistent locations relative to the TR and prior significant highs/lows.  
  * **Phase Transition Logic:** Contains rules (configurable in config.yaml) for transitioning between Wyckoff phases (UNKNOWN, ACCUMULATION\_A through E, DISTRIBUTION\_A through E). Current implementation has detailed contextual logic for Accumulation phases A-E, with Distribution phase logic being the next major refinement.  
  * **Output:** Provides the current determined Wyckoff phase and a log of phase transitions.

## **Data Flow**

* **Input:**  
  * A list of ZBar objects, fully populated (either from M1 bar data with heuristic delta or, ideally, from tick data processed by tick\_processor.py with accurate price\_ladder and bar\_delta).  
  * Configuration parameters from config.yaml (via wyckoff\_detector and wyckoff\_state\_machine sections).  
* **Output (from event\_detector.py fed into state\_machine.py):**  
  * A dictionary mapping detected event types to a list of bar indices where they occurred.  
* **Output (from state\_machine.py):**  
  * The current classified Wyckoff phase (e.g., "ACCUMULATION\_C").  
  * A log of all phase transitions.  
  * Contextual information like the current schematic type and TR boundaries.

## **Future Enhancements**

* Complete implementation of all Distribution phase events and transition logic.  
* Integrate Point and Figure (PnF) analysis for target projection based on TRs.  
* Add logic for Re-accumulation and Re-distribution schematics.  
* Potentially incorporate Machine Learning models to validate detected events or phases, or to provide confidence scores.