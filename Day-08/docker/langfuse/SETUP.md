# Langfuse Setup

## Included config

Day-08 Langfuse 실행에 필요한 설정 파일은 이미 포함되어 있습니다.

- Compose: [docker-compose.yml](/Users/jhj/Desktop/2026_1_sds_ax_advanced/Day-08/docker/langfuse/docker-compose.yml)
- Env template: [.env.example](/Users/jhj/Desktop/2026_1_sds_ax_advanced/Day-08/docker/langfuse/.env.example)

## Important note

Langfuse official Docker setup is `.env`-driven.
There is no separate Day-08 YAML config file to maintain for Langfuse.
The effective runtime config is the combination of:

- `docker-compose.yml`
- `.env`

## Minimum values to fill in `.env`

At minimum:

- `NEXTAUTH_SECRET`
- `DATABASE_URL`
- `SALT`
- `ENCRYPTION_KEY`
- `CLICKHOUSE_PASSWORD`
- `MINIO_ROOT_PASSWORD`
- `REDIS_AUTH`
- `POSTGRES_PASSWORD`

If you want the Day-08 notebooks to connect immediately after startup, also prepare:

- `LANGFUSE_INIT_PROJECT_PUBLIC_KEY`
- `LANGFUSE_INIT_PROJECT_SECRET_KEY`
- `LANGFUSE_INIT_USER_EMAIL`
- `LANGFUSE_INIT_USER_PASSWORD`

## Generate secure values

```bash
openssl rand -hex 32
```

Use one output for `ENCRYPTION_KEY`.

```bash
openssl rand -base64 32
```

Use outputs like this for `NEXTAUTH_SECRET` and `SALT`.

## Run

```bash
cd /Users/jhj/Desktop/2026_1_sds_ax_advanced/Day-08/docker/langfuse
cp .env.example .env
docker compose up -d
```

## Verify

Open:

- [http://localhost:3000](http://localhost:3000)

For Day-08 notebooks, the learner-facing values will usually be:

- `LANGFUSE_HOST=http://localhost:3000`
- `LANGFUSE_PUBLIC_KEY=<project public key>`
- `LANGFUSE_SECRET_KEY=<project secret key>`

