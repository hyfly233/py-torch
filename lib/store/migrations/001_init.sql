-- 001_init.sql Carrot AI Infra 初始表结构（对齐 ARCHITECTURE-V2 §4）
-- 表：models / model_versions / deployments / deployment_events / tenant_quotas / api_keys / audit_logs

-- 模型
CREATE TABLE IF NOT EXISTS models (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 模型版本
CREATE TABLE IF NOT EXISTS model_versions (
    id             TEXT PRIMARY KEY,
    model_id       TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    version        TEXT NOT NULL,
    artifact_uri   TEXT NOT NULL,
    runtime        TEXT NOT NULL DEFAULT 'vLLM',
    gpu_type       TEXT NOT NULL,
    gpu_count      INT  NOT NULL DEFAULT 1,
    memory_mb      BIGINT NOT NULL DEFAULT 0,
    context_length INT  NOT NULL DEFAULT 0,
    status         TEXT NOT NULL DEFAULT 'REGISTERED',
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (model_id, version)
);
CREATE INDEX IF NOT EXISTS idx_model_versions_model ON model_versions(model_id);

-- 部署
CREATE TABLE IF NOT EXISTS deployments (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL UNIQUE,
    model_id         TEXT NOT NULL,
    model_version_id TEXT NOT NULL,
    model_name       TEXT NOT NULL DEFAULT '',
    model_version    TEXT NOT NULL DEFAULT '',
    tenant_id        TEXT NOT NULL DEFAULT 'default',
    namespace        TEXT NOT NULL DEFAULT '',
    replicas         INT  NOT NULL DEFAULT 1,
    gpu_type         TEXT NOT NULL DEFAULT '',
    gpu_count        INT  NOT NULL DEFAULT 1,
    memory_mb        BIGINT NOT NULL DEFAULT 0,
    runtime          TEXT NOT NULL DEFAULT 'vLLM',
    startup_args     TEXT NOT NULL DEFAULT '[]',  -- JSON 数组
    endpoint         TEXT NOT NULL DEFAULT '',
    status           TEXT NOT NULL DEFAULT 'NEW',
    generation       BIGINT NOT NULL DEFAULT 0,
    diagnostics      TEXT NOT NULL DEFAULT '',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_deployments_tenant ON deployments(tenant_id);
CREATE INDEX IF NOT EXISTS idx_deployments_status ON deployments(status);

-- 部署状态事件（审计）
CREATE TABLE IF NOT EXISTS deployment_events (
    id            BIGSERIAL PRIMARY KEY,
    deployment_id TEXT NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    from_status   TEXT NOT NULL DEFAULT '',
    to_status     TEXT NOT NULL DEFAULT '',
    reason        TEXT NOT NULL DEFAULT '',
    request_id    TEXT NOT NULL DEFAULT '',
    diagnostics   TEXT NOT NULL DEFAULT '',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_deployment_events_deploy ON deployment_events(deployment_id);

-- 租户配额
CREATE TABLE IF NOT EXISTS tenant_quotas (
    tenant_id TEXT NOT NULL,
    gpu_type  TEXT NOT NULL,
    quota     INT NOT NULL DEFAULT 0,
    used      INT NOT NULL DEFAULT 0,
    PRIMARY KEY (tenant_id, gpu_type)
);

-- API Key（哈希存储，不存明文）
CREATE TABLE IF NOT EXISTS api_keys (
    id          TEXT PRIMARY KEY,
    key_hash    TEXT NOT NULL UNIQUE,
    tenant_id   TEXT NOT NULL DEFAULT 'default',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at TIMESTAMPTZ,
    disabled    BOOLEAN NOT NULL DEFAULT false
);

-- 审计日志
CREATE TABLE IF NOT EXISTS audit_logs (
    id         BIGSERIAL PRIMARY KEY,
    action     TEXT NOT NULL,
    actor      TEXT NOT NULL DEFAULT '',
    tenant_id  TEXT NOT NULL DEFAULT '',
    resource   TEXT NOT NULL DEFAULT '',
    request_id TEXT NOT NULL DEFAULT '',
    detail     TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created ON audit_logs(created_at DESC);
