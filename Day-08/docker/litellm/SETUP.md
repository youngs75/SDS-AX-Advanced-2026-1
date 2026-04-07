# LiteLLM Setup

## Included config

Day-08 LiteLLM 실행에 필요한 설정 파일은 이미 포함되어 있습니다.

- Compose: [docker-compose.yml](/Users/jhj/Desktop/2026_1_sds_ax_advanced/Day-08/docker/litellm/docker-compose.yml)
- Proxy config: [litellm.config.yaml](/Users/jhj/Desktop/2026_1_sds_ax_advanced/Day-08/docker/litellm/litellm.config.yaml)
- Env template: [.env.example](/Users/jhj/Desktop/2026_1_sds_ax_advanced/Day-08/docker/litellm/.env.example)
- Prometheus config: [prometheus.yml](/Users/jhj/Desktop/2026_1_sds_ax_advanced/Day-08/docker/litellm/prometheus.yml)

## What to fill in `.env`

At minimum:

- `LITELLM_MASTER_KEY`
- `VLLM_API_BASE`
- `VLLM_API_KEY`
- `VLLM_LITELLM_MODEL`
- `LITELLM_MODEL_ALIAS`

## Day-08 runtime contract

- Students call LiteLLM through `http://localhost:4000/v1`
- Students use `LITELLM_MASTER_KEY` as the OpenAI-compatible bearer key
- Students use `LITELLM_MODEL_ALIAS` as the only model name
- LiteLLM forwards that alias to the instructor-provided vLLM upstream

## Run

```bash
cd /Users/jhj/Desktop/2026_1_sds_ax_advanced/Day-08/docker/litellm
cp .env.example .env
docker compose up -d
```

## Verify

```bash
curl http://localhost:4000/health/liveliness
```

```bash
curl http://localhost:4000/v1/models \
  -H "Authorization: Bearer sk-day08-litellm"
```

