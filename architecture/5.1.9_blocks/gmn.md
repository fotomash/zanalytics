# **/zanzibar/gmn/ \- Global Macro News (GMN) Module**

This directory is dedicated to the Global Macro News (GMN) module, responsible for ingesting, parsing, and structuring factual news articles from various external sources (e.g., financial news websites, APIs, PDF reports) into a machine-readable format for the Zanzibar Analytics system.

## **Purpose**

* **Factual Context Layer:** To provide an objective layer of information based on reported news events, complementing the speculative insights from the PMS module.  
* **Event & Entity Extraction:** To identify and structure key information from news content, such as main events, entities involved (countries, organizations, people), affected regions/sectors, and potentially implied asset impacts.  
* **Sentiment Analysis:** To (optionally) assess the sentiment of news articles regarding specific entities or overall market implications.  
* **Augment System Awareness:** GMN data provides a factual backdrop against which quantitative signals and PMS speculations can be evaluated, allowing the system to check for alignment or divergence.

## **Key Components (Current & Planned)**

* **schema.py (Version: gmn\_schema\_v1 defined)**  
  * **Responsibility:** Defines or loads the standard JSON schema for GMN events. This schema structures extracted news information, including fields like event\_id, timestamps (recorded and publication), source\_name, news\_headline, news\_summary, key\_event\_type, entities\_mentioned, market\_impact\_assessment, and optional sentiment\_score.  
  * **Importance:** Ensures consistent data format for news-derived information.  
* **parser.py (Version: v1 Scaffold \- gmn\_module\_scaffold\_v1)**  
  * **Responsibility:** Ingests raw news content (from text, PDFs, URLs) and uses Natural Language Processing (NLP) techniques to extract and structure the information according to the GMN schema.  
  * **Functionality (Planned):**  
    * **Text Extraction:** Robustly extract clean text from various input formats (PDFs using libraries like PyPDF2 or pdfminer.six; URLs using requests and BeautifulSoup4).  
    * **Core NLP Processing (using spaCy or similar):**  
      * **Headline & Publication Date Extraction.**  
      * **Summarization:** Generate concise summaries of articles.  
      * **Named Entity Recognition (NER):** Identify countries, organizations, persons, financial instruments, etc.  
      * **Event Extraction/Classification:** Determine the main type of event described (e.g., "TariffAction", "Geopolitical", "EconomicDataRelease").  
      * **Sentiment Analysis:** Determine the sentiment of the article or towards specific entities.  
    * **Mapping to Schema:** Populate the GMN JSON object with extracted data.  
  * **Current Status:** Scaffolded with placeholder functions. Requires implementation of text extraction and NLP logic.

## **Data Flow**

1. **Input:** News articles (content from PDFs, web URLs, direct text, or future API feeds).  
2. **Processing:** parser.py ingests the content, performs text extraction, and applies a cascade of NLP models for information extraction and structuring.  
3. **Output:** Structured JSON objects conforming to gmn\_schema.json. These are intended for storage (e.g., in zanzibar.data\_management.metadata\_db) and consumption by the Orchestrator and other analytical modules.

## **Configuration**

* The GMN parser will likely require its own section in config/config.yaml for:  
  * NLP model paths and settings.  
  * Web scraping parameters (e.g., user agents, target HTML elements for common sources).  
  * API keys for news provider APIs (if used in the future).  
  * Keyword lists or rules for event type classification or entity mapping.

## **Integration with Zanzibar Core**

* **Contextual Enrichment:** GMN events provide factual context that can be correlated with market movements, Wyckoff phases, or PMS speculations.  
* **Narrative Alignment:** The system can compare GMN-reported events with PMS views to identify consensus or divergence.  
* **Feature Engineering:** Extracted entities, event types, and sentiment scores from GMN can serve as features for future Machine Learning models.