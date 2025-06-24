
# 🟫 Zanzibar Trader v2.0 – Zone Annotator Overview

## 🎯 Purpose
Automatically detect and tag significant **Smart Money Concept (SMC)** zones based on OHLCV candle data (generated from tick or broker feed). Designed to precompute zones *before* LLM agent execution, reducing overhead and increasing precision.

---

## ⚙️ Step-by-Step Logic

### 1. **Input**
- Timeframe data (M1, M15, H1 etc.), from tick-to-OHLCV conversion
- Optional: RSI, Delta, ATR, Volume
- Configuration parameters: swing sensitivity, POI types to detect

---

### 2. **Swing Detection**
- Identify local **swing highs and lows** using rolling window pivots
- These mark potential BOS/CHoCH and POI anchor zones

---

### 3. **POI Detection Rules**

#### 🔹 Demand Zone
- Price declines, then strong impulse upward
- POI = candle body before impulse
- Tag example:
```json
{
  "type": "Demand",
  "source": "M15",
  "low": 3210.2,
  "high": 3213.5
}
```

#### 🔹 Fair Value Gap (FVG)
- Price gap between candles (e.g. `C[i+1] < O[i+2]`)
- Define FVG zone between gap candles

#### 🔹 Mitigation Block
- Retest of demand/supply after BOS
- Must occur in discount or premium (Fibonacci validated)

---

### 4. **Optional Filters**
- RSI divergence near POI
- EMA crossover
- Session window alignment (Asia/London/NY)

---

### 5. **Output**
- JSON `pois[]` array like:
```json
[
  {
    "type": "Demand",
    "start_time": "2025-04-15T13:45:00",
    "end_time": "2025-04-15T14:00:00",
    "low": 3210.25,
    "high": 3213.50,
    "source": "M15"
  }
]
```

---

## 🧠 Why This Matters
- Reduces load on LLMs
- Standardizes zone visibility across all variants (Mentfx, Inversion, Maz2, TMC)
- Connects with SierraChart logic, enabling hybrid SMC footprint workflows

---

## 🔁 Ready to Connect
This system can be used in two ways:
- 🖥️ Run locally (heavy data)
- 🧠 Or injected into agent orchestrator for final-stage POI refinement

