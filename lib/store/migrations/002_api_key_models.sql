-- 002_api_key_models.sql R2-4：API Key 模型授权白名单
ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS models TEXT NOT NULL DEFAULT '[]';  -- JSON 数组
