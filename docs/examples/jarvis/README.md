# Jarvis two-router example

Tracked copy of the Strix Halo llama.cpp split used under `output/Jarvis/` (gitignored deploy tree).

| Port | INI | llama.cpp behavior (not gateway) |
|---|---|---|
| `11535` | `models.ini` | one model, `load-on-startup` (stays warm) |
| `11537` | `models-lab.ini` | swap pool, `--models-max 1`, idle sleep after 15 min |

Gateway setup (only addresses + aliases — no swap/sticky concepts):

1. Source `chat` → `host:11535`
2. Source `chat2` (or any name) → `host:11537`
3. Alias `qwen3.6` → `Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL-VL`, preferred source `chat`
4. Alias `smart` → `Qwen3.8-27B-UD-Q4_K_XL-MTP` (or `Qwen3.8-27B-UD-Q8_K_XL-MTP`), preferred source `chat2`

Swap/load-on-startup is entirely llama.cpp router config. The gateway only proxies to the address you configured.
