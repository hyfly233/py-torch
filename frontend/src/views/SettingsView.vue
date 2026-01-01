<script setup lang="ts">
// 系统设置（管理员占位）
</script>

<template>
  <div class="page">
    <div class="page-header">
      <h2>系统设置</h2>
    </div>
    <div class="panel">
      <div class="panel-title">租户与配额</div>
      <p class="dim">
        MVP 为单租户（default），多租户与配额管理将在后续版本提供。
      </p>
    </div>
    <div class="panel mt-16">
      <div class="panel-title">关于</div>
      <p class="dim">
        Carrot AI Infra Platform · 版本 0.1.0 (MVP)<br />
        基于 Kubernetes + vLLM 的 AI 推理服务平台
      </p>
    </div>
  </div>
</template>

<style scoped>
.dim {
  color: var(--text-dim);
  line-height: 1.8;
}
</style>
