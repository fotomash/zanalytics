# **/zanzibar/data\_management/ \- Data Models & Storage Abstraction**

This directory is planned to house modules related to the definition of core data structures (like the ZBAR™ protocol) and abstractions for data storage and retrieval within the Zanzibar Analytics system.

## **Purpose**

* **Centralized Data Definitions:** Provide a single source of truth for the structure and schema of key data objects used throughout the application, most notably the ZBar and MarketOrderData.  
* **Storage Agnosticism:** Abstract the underlying data storage mechanisms (e.g., local JSON files, SQL databases, NoSQL databases, time-series databases, vector databases) from the core analytical logic. This allows the storage backend to be changed or extended with minimal impact on other modules.  
* **Data Integrity & Access Patterns:** Define clear interfaces (Repositories) for creating, reading, updating, and deleting (CRUD) persisted data, ensuring consistency and optimized access.

## **Key Components (Planned & Current Location)**

* **models.py (Currently, ZBar & MarketOrderData are in zanzibar/analysis/wyckoff/event\_detector.py)**  
  * **Responsibility:** Define the core Python dataclasses for the system's primary data entities.  
  * **Key Entities:**  
    * **ZBar**: The proprietary enriched bar object (OHLCV, price\_ladder, bar\_delta, poc\_price, poi\_price, etc.). Includes methods like calculate\_heuristic\_delta() and calculate\_derived\_metrics\_from\_ladder().  
    * **MarketOrderData**: Represents aggregated order flow data at a specific price level within a ZBar's price\_ladder.  
    * (Future) PMSEvent, GMNEvent: Dataclasses for structured narrative inputs.  
  * **Strategic Importance:** These models embody the ZBAR™ protocol and are fundamental to all data processing and analysis. Centralizing them here will improve clarity and maintainability.  
* **(Future) repository\_interface.py or storage\_interface.py**  
  * **Responsibility:** Define abstract base classes or interfaces for data repositories (e.g., IZBarRepository).  
  * **Functionality:** Specify methods like save\_zbar(zbar: ZBar), get\_zbar(timestamp: datetime) \-\> Optional\[ZBar\], get\_zbars\_range(start\_ts: datetime, end\_ts: datetime) \-\> List\[ZBar\], check\_timestamp\_exists(timestamp: datetime) \-\> bool, update\_zbar\_index(...).  
* **(Future) json\_repository.py**  
  * **Responsibility:** An implementation of the IZBarRepository interface for storing and retrieving ZBAR™ objects and their index using local JSON files (as currently outlined in config.yaml under storage).  
* **(Future) db\_repository.py (e.g., timescaledb\_repository.py)**  
  * **Responsibility:** An implementation of the IZBarRepository for a specific database backend (e.g., TimescaleDB, PostgreSQL).

## **Data Flow**

1. **Data Ingestion (zanzibar/loader/ modules):** After parsing and mapping raw data, the zbar\_mapper.py utility instantiates ZBar objects (defined in models.py).  
2. **Storage (via Repository):** The ingestion pipeline (or a dedicated storage service) would use a ZBarRepository implementation to save these ZBar objects and update the timestamp index.  
3. **Retrieval (via Repository):** Analytical engines (zanzibar/analysis/) would use the ZBarRepository to fetch ZBar data for processing, without needing to know the specifics of the underlying storage.

## **Migration Note**

Currently, the ZBar and MarketOrderData dataclasses are defined within zanzibar/analysis/wyckoff/event\_detector.py. A planned refactoring step is to move these core model definitions to zanzibar/data\_management/models.py to better reflect their system-wide importance and decouple them from a specific analysis module.

The `UnifiedAnalyticsBar` Pydantic model in `core/schema.py` exposes a JSON-friendly view of each `ZBar`. Keep these structures synchronized—they are the canonical analytics schema for both internal engines and API consumers.
