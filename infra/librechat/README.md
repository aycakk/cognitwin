# LibreChat Setup

This folder contains the team-shareable LibreChat configuration for connecting to the CogniTwin API.

## What To Commit

- `infra/librechat/.env.example`
- `infra/librechat/librechat.yaml`
- `infra/librechat/docker-compose.override.yml.example`
- `infra/librechat/README.md`

Do not commit real `.env` files, logs, uploads, databases, or LibreChat secrets.

## Team Setup

1. Clone the official LibreChat repository into any local folder.
2. Copy `infra/librechat/.env.example` to the LibreChat root as `.env`.
3. Copy `infra/librechat/librechat.yaml` to the LibreChat root.
4. On Windows, copy `infra/librechat/docker-compose.override.yml.example` to the LibreChat root as `docker-compose.override.yml`.
5. Start the CogniTwin API locally on port `8000`.
6. Start LibreChat with `docker compose up -d`.
7. Open `http://localhost:3900`.

## CogniTwin API

LibreChat is configured to call:

`http://host.docker.internal:8000/v1/`

If the CogniTwin API uses another port, update `baseURL` in `librechat.yaml`.

## Notes

- `PORT=3900` avoids the blocked `3080/3081` ports we hit locally.
- If LibreChat fails with log permission errors on Windows, use the override file that runs the relevant containers as `0:0`.
