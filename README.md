# OnPrem AI Gateway

**Version:** [0.1.0](CHANGELOG.md) · early production (`0.x` — APIs and UX may still change)

One front door for your **local** AI servers: chat, embeddings, speech-to-text, and text-to-speech — with login, API keys, access grants, and usage tracking.

You keep the models on your LAN. Clients (VS Code extensions, scripts, apps, friends) talk to **one OpenAI-compatible base URL** with an API key. The gateway checks who may call what, then proxies to the right backend.

```
Your laptop / app  ──►  OnPrem AI Gateway  ──►  llama.cpp / Ollama / Whisper / Piper / …
                         (login + API keys)         (on your machines)
```

<p align="center">
  <img src="docs/screenshots/login.png" alt="Sign-in screen" width="720">
</p>

<p align="center">
  <img src="docs/screenshots/dashboard.png" alt="Ops overview with fleet health" width="720">
</p>

| API keys | Backends |
|:---:|:---:|
| <img src="docs/screenshots/keys.png" alt="API Keys list" width="420"> | <img src="docs/screenshots/services.png" alt="Services and live status" width="420"> |

---

## Who is this for?

| You… | Then this fits |
|------|----------------|
| Run AI on a home server, gaming PC, or small lab | Yes — expose models safely without giving everyone SSH |
| Want friends/family/colleagues to use *your* models with their own keys | Yes — create users, grant sources/models, hand out keys |
| Need OpenAI-shaped APIs (`/v1/chat/completions`, …) for existing tools | Yes — point `baseURL` at the gateway |
| Want a public SaaS model marketplace | No — this is **on-prem / self-hosted** |
| Expect zero ops (hosted cloud only) | No — you run Docker + your backends |

**Roles in short**

- **Platform admin** — backends, users, grants, settings, SMTP, catalog  
- **User** — own API keys, usage, profile (only what you granted them)

---

## What it does

- **Single API base** — e.g. `https://ai.example.com/v1` (or `http://localhost:9081/v1` locally)
- **API keys** — create in the web UI; secret shown **once**; use `X-Api-Key` or `Authorization: Bearer …`
- **Access control** — admin grants users sources and models; keys can only use that grant
- **Model catalog** — sync from backends, enable/disable models, notes show up in `/v1/models`
- **Routing by `model`** — request says which model; gateway picks the matching enabled source (no silent dump onto another box)
- **Chat · embed · STT · TTS** — path selects the kind of service; backends stay on your network
- **Optional teams**, rate/concurrency limits, daily quotas, usage views
- **Optional GPU thermal guard** — pause traffic when a source host runs too hot (sidecar)
- **Welcome mail / invites** — optional SMTP for new users (keys still shared out of band — copy once)

---

## How it works

Two containers (Compose):

1. **`onprem-auth`** — web UI + auth + catalog + “may this key call this model?”  
2. **`onprem-api`** — nginx front for `/v1/…`; asks auth, then proxies to `host:port` backends

```
Browser  →  Web UI (login cookie)     →  manage keys, users, services
Client   →  /v1/... + API key         →  check grant → upstream AI server
```

You register **sources** in the UI (or seed them once via `.env` on an empty DB): kind (`chat` / `embed` / `stt` / `tts`) + address. Clients always speak OpenAI-style `/v1`; each source can use a dialect suited to the backend.

**Typical admin path**

1. Start gateway → log in as bootstrap admin  
2. Add services (backends) → sync models → enable what you want  
3. Create a user → set their grant (which sources/models)  
4. Create an API key for them → send them base URL + key (WhatsApp / password manager; not in welcome mail by default)

**Typical end-user path**

1. Log in → change password if asked  
2. Open **API Keys** (or use the key admin created)  
3. In the client: base URL = gateway `/v1`, header = API key  
4. Call models as usual; only granted models appear in `/v1/models`

---

## Quick start (local, no reverse proxy)

**Requirements:** Docker + Compose, and at least one AI backend reachable from the gateway host.

```bash
cp .env.example .env
# Set SESSION_SECRET, ADMIN_BOOTSTRAP_PASSWORD, DOMAIN
# Optional first-boot seeds: CHAT_SOURCE=192.168.x.x:11535  (etc.)

docker compose up -d --build
```

| What | URL |
|------|-----|
| Web UI | http://localhost:9080 |
| API base | http://localhost:9081/v1 |

Log in with `ADMIN_BOOTSTRAP_USER` / `ADMIN_BOOTSTRAP_PASSWORD` from `.env`, finish setup, create a key, then:

```bash
curl -s -H "X-Api-Key: YOUR_KEY" http://localhost:9081/v1/models
```

Example chat (shape depends on your models):

```bash
curl -s http://localhost:9081/v1/chat/completions \
  -H "X-Api-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"YOUR_MODEL_ID","messages":[{"role":"user","content":"Hello"}]}'
```

In OpenAI-compatible apps (Continue, OpenCode, custom SDKs, …):

- **Base URL:** `http://localhost:9081/v1` (or your public `https://…/v1`)  
- **API key:** the key from the UI  

---

## Deploy behind Traefik (optional)

If you already run Traefik on Docker network `proxy`:

```bash
# In .env: PUBLIC_HOST=ai.example.com  PUBLIC_SUBDOMAIN=ai  DOMAIN=example.com
docker compose -f compose.traefik.yaml up -d --build
```

- Web UI: `https://ai.example.com/`  
- API: `https://ai.example.com/v1/...`  

Same product; only how you reach it changes. See `.env.example` for cookie/HTTPS notes (`PUBLIC_HOST` / `SESSION_COOKIE_SECURE`).

---

## API paths (what clients call)

| Path | Purpose |
|------|---------|
| `GET /v1/models` | Chat (+ embed) this key may see. STT/TTS use `/v1/audio/*` (not listed by default; `?kinds=all` opt-in) |
| `/v1/chat/completions` (and related chat paths) | Chat — routed by `model` |
| `/v1/embeddings` | Embeddings — routed by `model` |
| `/v1/audio/transcriptions` | Speech → text (STT) |
| `/v1/audio/speech` | Text → speech (TTS) |
| `/s/{name}/v1/…` | Optional: pin a named source; same catalog rules |

Unknown, disabled, or missing `model` → clear error (`unknown_model` / `missing_model`), not a random other server.

Aliases such as **`auto`**, **`auto-quality`**, **`auto-long`** can be configured under **Settings → Routing**.

Health / version: `GET /healthz` → `{"status":"ok","version":"0.1.0"}`.

---

## Configuration: `.env` vs web UI

| Set in `.env` | Manage in the web UI |
|---------------|----------------------|
| `DOMAIN`, `SESSION_SECRET`, bootstrap admin | API keys, users, grants |
| `PUBLIC_HOST` / ports / Traefik labels | Services (sources), model catalog |
| Optional `CHAT_SOURCE` … on **empty** DB only | SMTP, teams, operator/Impressum if env empty |
| Thermal sidecar thresholds | Limits on keys / grants |

Do not commit `.env`. After the first source exists, env seed vars are ignored — use **Services** in the UI.

---

## Optional: thermal / power sidecar

On the **GPU / source host**, you can run `services/source-sidecar/` so the gateway can read temperature/power and refuse traffic when too hot (`TEMP_MAX_C`, fail-open/closed in `.env`). Useful for a gaming PC that also serves models.

---

## Project layout

```
app/                      # FastAPI: web UI + API auth + data
services/source-sidecar/  # optional power/thermal helper on source hosts
compose.yaml              # local ports 9080 / 9081
compose.traefik.yaml      # Traefik example
docs/                     # architecture & layout notes
tests/                    # pytest
```

Deeper design notes: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · UI conventions: [docs/LAYOUT.md](docs/LAYOUT.md) · history: [CHANGELOG.md](CHANGELOG.md)

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q -m "not integration"          # default / CI — no LAN required

# Optional: hit real backends from .env (never commit .env)
INTEGRATION=1 pytest -m integration -q
```

---

## License

See [LICENSE](LICENSE).

---

**Bottom line:** OnPrem AI Gateway turns your home/lab AI boxes into a shared, keyed, OpenAI-compatible service — without putting the models in someone else’s cloud.
