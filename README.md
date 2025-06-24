# Standalone Logic Bundle (YAML-Driven Multi-Agent Framework)

## 🔍 How to Use

This bundle is your **modular trading logic engine**. It’s structured for use by:
- Python orchestrators
- LLMs (GPT, Claude, Mistral)
- Hybrid workflows with human and autonomous agents

### 📦 Structure

- `profiles.yaml` – Manifest of all available agent configs.
- `/profiles/*.yaml` – Modular agents with composable logic.
- `/schemas/agent_profile.schema.yaml` – JSON schema for validation.
- `runner.py` – Minimal YAML loader to validate logic integration.

## 🧠 GPT & LLM Integration Instructions

- Load `profiles.yaml`, parse each `profiles/*.yaml`.
- Treat each block (e.g., `context_analyzer`, `liquidity_engine`) as a **deterministic logic unit**.
- If `tick_context.enable_tick_merge = true`, perform tick-M1 resampling and inject:
  - `tick_count`, `avg_spread`, `max_spread`, `spread_spike`
- Use values in gating logic (skip entry if `spread_spike` or `tick_count < threshold`).
- Respect all YAML thresholds and flags. **Do not invent logic**.
- All logic execution must follow `workflow:` if present, or top-to-bottom module sequence.

## ✅ Execution Flow

1. Load profile from `profiles.yaml`
2. Inject data (tick + OHLC)
3. Run filters in order:
    - `context_analyzer`
    - `liquidity_engine`
    - `structure_validator`
    - `risk_manager`
    - `journaling`
4. Skip any module where config is missing/empty
5. Log outcome using `journaling` block

*Multi-agent. Schema-driven. LLM-interpretable. Fully portable.*