# Changelog

All notable changes to OnPrem AI Gateway are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Anthropic Messages client dialect: `POST /v1/messages` (Claude Code compatible — mid-conversation `role: system`, tools, vision, stream; Anthropic-only fields like `thinking` dropped for local upstreams)

## [0.1.0] — 2026-08-20

First tagged release. Early production: API and UI are usable; contracts may still change in 0.x.

### Added

- OpenAI-compatible gateway (`/v1/*`) with API keys, grants, and model catalog
- Web UI for users, keys, usage, services, models, settings, SMTP, legal pages
- Optional teams mode, thermal/temp guard, Traefik compose example
- Admin user create with optional username; email login when username is pending
- Welcome mail flow and key creation for granted users

### Fixed

- Owner/user labels show email when username is still `pending-*`
- Large JSON body model extraction for clients that send big chat payloads

[0.1.0]: https://github.com/fr4iser90/LocalAI-GateWay/releases/tag/v0.1.0
