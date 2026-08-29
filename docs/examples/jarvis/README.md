# Jarvis two-router example

Tracked copy of the Strix Halo llama.cpp split used under `output/Jarvis/` (gitignored deploy tree).

| Port | INI | llama.cpp behavior (not gateway) |
|---|---|---|
| `11535` | `models.ini` | one model, `load-on-startup` (stays warm) |
| `11537` | `models-lab.ini` | swap pool, `--models-max 1`, idle sleep after 15 min |

Gateway setup (only addresses + aliases — no swap/sticky concepts):

1. Source `chat` → `host:11535`
2. Source `chat2` (or any name) → `host:11537`
3. Alias slot `qwen3.6` → active `Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL-VL`, preferred source `chat`,
   candidates = all `Qwen3.6-*` (Suggest family), **hide others** on
4. Alias slot `smart` / `qwen3.8` → active `Qwen3.8-27B-UD-Q4_K_XL-MTP` (or Q8), preferred `chat2`,
   candidates = all `Qwen3.8-*`, hide others on

Clients see `qwen3.6` / `smart` in `GET /v1/models`; raw quant variants stay in the admin catalog
for switching the active candidate. Grant keys the alias id (e.g. `chat:qwen3.6`).

Each model/alias entry may also include live capacity from the preferred upstream:

```json
{ "id": "qwen3.6", "slots_total": 3, "slots_idle": 2, "slots_busy": 1, "load_state": "ok" }
```

(`slots_*` from engine `/slots` or Services `max_concurrency`; cached ~3s. Fleet-wide, not per-key.)

Swap/load-on-startup is entirely llama.cpp router config. The gateway only proxies to the address you configured.
