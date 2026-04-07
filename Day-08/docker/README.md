# Day-08 Docker Assets

This directory contains Day-08-ready Docker Compose assets based on the
official upstream GitHub examples.

## Sources

- LiteLLM official repo:
  - `docker-compose.yml`
  - `prometheus.yml`
  - `dev_config.yaml`
  - Repo: https://github.com/BerriAI/litellm
- Langfuse official repo:
  - `docker-compose.yml`
  - Repo: https://github.com/langfuse/langfuse

## Layout

- `litellm/`
  - `docker-compose.yml`
  - `litellm.config.yaml`
  - `prometheus.yml`
  - `.env.example`
  - `SETUP.md`
- `langfuse/`
  - `docker-compose.yml`
  - `.env.example`
  - `SETUP.md`

## Usage

### LiteLLM

```bash
cd /Users/jhj/Desktop/2026_1_sds_ax_advanced/Day-08/docker/litellm
cp .env.example .env
docker compose up -d
```

LiteLLM will listen on `http://localhost:4000`.

### Langfuse

```bash
cd /Users/jhj/Desktop/2026_1_sds_ax_advanced/Day-08/docker/langfuse
cp .env.example .env
docker compose up -d
```

Langfuse UI will listen on `http://localhost:3000`.

## Day-08 notes

- LiteLLM is configured to sit in front of an instructor-provided `vLLM`
  endpoint via `VLLM_API_BASE`.
- LiteLLM uses a workshop model alias so the notebooks do not need to know
  the raw upstream model name.
- LiteLLM does not publish its Postgres port to the host to avoid collisions
  with the Langfuse stack.
- Prometheus is published on `9092` to avoid colliding with Langfuse MinIO on
  `9090`.
