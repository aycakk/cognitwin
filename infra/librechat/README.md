# LibreChat Setup

This folder contains the team-shareable LibreChat configuration for connecting to the CogniTwin API.

## What To Commit

- `infra/librechat/.env.example`
- `infra/librechat/librechat.yaml`
- `infra/librechat/docker-compose.override.yml.example`
- `infra/librechat/README.md`

Do not commit real `.env` files, logs, uploads, databases, or LibreChat secrets.

## Team Setup

1. In this folder, copy `.env.example` to `.env`.
2. (Windows optional) copy `docker-compose.override.yml.example` to `docker-compose.override.yml`.
3. Start the CogniTwin API locally on port `8011`.
4. Start LibreChat with `docker compose up -d`.
5. Open `http://localhost:3900`.

## CogniTwin API

LibreChat is configured to call:

`http://host.docker.internal:8011/v1/`

If the CogniTwin API uses another port, update `baseURL` in `librechat.yaml`.

## Notes

- `PORT=3900` avoids the blocked `3080/3081` ports we hit locally.
- If LibreChat fails with log permission errors on Windows, use the override file that runs the relevant containers as `0:0`.

## Multi-Workspace Ports

- Student workspace: `http://localhost:3900`
- Agile workspace: `http://localhost:3901`
- HR workspace: `http://localhost:3902`
- Workspace portal: `http://localhost:8080`
- n8n UI (automation): `http://localhost:5678`
