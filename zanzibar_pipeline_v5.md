# 📡 Zanzibar Trader v3: Data Flow Pipeline

**Description**:  
This document outlines the full data pipeline used in Zanzibar Trader v3, covering all system components, technologies, and data transformations from raw tick intake to broker execution and logging.

---

## 🔁 Stage 1: Raw Data Intake
- **Component**: External Sources (Brokers / Feeds)
- **Action**: Provides raw market data (vendor-specific tick data). This is the entry point for all market data.
- **Technology**: Broker APIs, CSV Files
- **Data Flow**: `Vendor-specific tick data` → `Raw Ticks`
- **Target**: Ingestion Layer

---

## 📥 Stage 2: Ingestion & Initial Queuing
- **Component**: Ingestion Layer
- **Action**: Batches raw ticks and publishes them, creating a durable, high-throughput buffer.
- **Technology**: Kafka
- **Data Flow**: `Raw Ticks` → `Raw Ticks (Batched, Serialized)`
- **Source**: Ingestion Layer
- **Target**: Kafka Topic(s)

---

## 🧠 Stage 3: Timeframing & Enrichment
- **Component**: Enrichment Layer
- **Action**: Resamples into different timeframes, calculates BOS/CHoCH, Liquidity, FVG, Wyckoff, etc.
- **Technology**: Kafka Consumer, Pandas/Numpy (Vectorized Ops)
- **Data Flow**: `Raw Ticks (Batched, Deserialized)` → `EnrichedBar (Protobuf)`
- **Source**: Kafka Topic(s)
- **Target**: Redis Stream

---

## 📡 Stage 4: Broadcasting Enriched Data
- **Component**: Enrichment Layer
- **Action**: Publishes processed EnrichedBar messages to Redis stream topics, acting like a data Routing Coordinator.
- **Technology**: Redis Stream
- **Data Flow**: `EnrichedBar (Protobuf)` → `EnrichedBar (Protobuf, Streamed)`
- **Source**: Enrichment Layer
- **Target**: Redis Stream Topics (e.g., `bar.enriched.<pair>.<tf>`)

---

## 🧬 Stage 5: Strategy Pod Consumption & Reasoning
- **Component**: Variant Pods (ISP, IOF, WYCKOFF, BBC)
- **Action**: Each pod consumes relevant EnrichedBar streams and applies profile-driven logic.
- **Technology**: Redis Stream Consumer Groups, Asyncio, Python
- **Data Flow**: `EnrichedBar (Protobuf, Deserialized)` → `Strategy Signal/Decision (JSON)`
- **Source**: Redis Stream Topics
- **Target**: Orchestrator

---

## 🧑‍✈️ Stage 6: Orchestration & Decision
- **Component**: Orchestrator (ZANZIBAR Copilot)
- **Action**: Central decision-making hub; merges signals, applies risk filters, finalizes trade decisions.
- **Technology**: Python, optional LLM component
- **Data Flow**: `Strategy Signal/Decision (JSON)` → `Orders / Commands (JSON or specific format)`
- **Source**: Variant Pods
- **Target**: Execution & Brokerage Layer

---

## 💼 Stage 7: Execution & Logging
- **Component**: Execution & Brokerage Layer
- **Action**: Manages broker API interaction, order lifecycle, and logs all trade/journal events.
- **Technology**: Broker API Client (e.g., PineConnector REST), Database/Logging Framework
- **Data Flow**: `Orders / Commands` → `Trade Execution Logs, Journal Entries`
- **Source**: Orchestrator
- **Target**: Broker + Notion/Grafana/Database

---