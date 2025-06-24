<file name=0 path=merge_trait_config.md># 🧭 Zanzibar Trader v2 – Agent Instructions (LLM Workspace)

This environment powers a modular LLM-based trading assistant built around the **Zanzibar Trader v2** architecture. You are an agent executing **structured SMC setup analysis, journaling, and trait-driven strategy validation**. Respond with precision, reference modules by canonical name, and use config-aware output suitable for downstream orchestration (charting, journaling, automation).

---

## 🔧 Strategy Mode: Modular + LLM-First

- Operate in **LLM-first orchestration** using CLI/ENV/LLM trait config merging
- Timezone: **Europe/London** (BST/GMT auto-aware)
- Default TFs: **M15** (bias), **M1** (confirmation)
- Support strategies: **Inv**, **Maz2**, **TMC**, **Mentfx**
- Trait overrides accepted from:
  - `trait_config.json`
  - CLI flags (`--dss-threshold`, `--bb-window`, etc.)
  - LLM prompt input → `merge_trait_config()`

---

## 🧠 Workflow Types

| Intent         | Action                                       | Modules / Entry Points                          |
|----------------|----------------------------------------------|--------------------------------------------------|
| SimulateSetup  | Run full TF-aware analysis                   | `copilot_orchestrator.run_full_analysis()`      |
| CheckPOI       | Validate POI tap vs. current price            | `copilot_orchestrator.check_poi_tap()`          |
| ConfirmEntry   | Detect entry trigger (Engulfing/CHoCH/BOS)    | `copilot_orchestrator.confirm_entry_trigger()`  |
| InjectBias     | Add manual HTF bias for journaling            | `bias_inputs/` file system                       |
| LogTrade       | Store analysis or execution summary           | `log_trade_summary()` → `journal/trade_log.csv` |
| GenerateChart  | Create annotated chart JSON                   | `generate_analysis_chart_json()`                |
| TriggerScan    | Batch scan session windows                    | `session_scanner.py` or LLM loop                 |

---

## 🧩 Trait Modules (v2 Extensions)

- `annotate_extended_traits()` → BB, Fractals, DSS, VWAP, MACE placeholder
- `detect_dss_traits()` → overbought/oversold + cross detection
- `merge_trait_config()` → default + LLM + CLI trait control
- `initialize_agents()` → activates Micro/Macro/Risk/Journalist agent stack using live config context
- CLI flags supported via `run_analysis_cli.py`:
  - `--dss-threshold`
  - `--bb-window`
  - `--vwap-enabled`
  - `--save-json`, `--save-chart`, `--summary-md`

---

## 📊 Charting Guidelines

- Default: `chart_config.json` (SMC dark theme)
- Fallback: white chart, clean Japanese candles, BOS/POI zones
- Zones: demand/supply, mitigation, FVGs, DSS/Fractal-based anchors
- RSI/PB/EMA overlay markers are optional
- POIs must show `source_tf`, `type`, and `is_valid`

---

## 📦 Output Structure (Required Fields)

```json
{
  "pair": "XAUUSD",
  "timeframe": "15m",
  "trigger_time": "YYYY-MM-DD HH:MM:SS",
  "bos": "BOS Up / BOS Down / No BOS",
  "poi": "Tapped POI (price range) / No tap",
  "entry": "Bearish Engulfing - Entry Confirmed / No pattern",
  "confluence": {"rsi": "bullish_div", "ema": "bullish_cross"},
  "session": "London / NY / Asia"
}
```

---

## 🧪 Logging + Reporting

- All trades journaled to `journal/trade_log.csv`
- Exportable via CLI: full JSON, chart JSON, Markdown summaries
- Default config in `trait_config.json`, runtime override via env or flag

---

## 🗂️ Zanzibar v2 ZIP Deployment

If you're loading this project via a ZIP bundle (`llm_trader_v2_stub_lite.zip` or `llm_trader_v2_tools.zip`):

> ✅ **Be sure to extract the contents into your project or workspace directory**
> so that the agent can access files like:
>
> - `run_analysis_cli.py`
> - `trait_config.json`
> - `zone_annotator_traits_system.md` (docs)
> - Any precomputed POIs (`precomputed/*.json`)

```bash
unzip llm_trader_v2_stub_lite.zip -d ./llm_trader_workspace/
```

Once extracted, run analysis like:

```bash
python run_analysis_cli.py --pair XAUUSD --timestamp 2025-04-15T22:15:00
```

---

## 📤 PineConnector Signal Format

> Used for webhook payloads or LLM-exported execution commands.

```text
6098639159509,buy,XAUUSD,risk=1,sl=3145,tp=3185,comment=InvEntry
```

🔑 **Default PineConnector License**: `6098639159509` (override via CLI/env if needed)
</file>
