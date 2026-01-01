<script setup lang="ts">
// 登录页：MVP 本地模拟角色选择
import { ref } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAuth } from '../composables/useAuth'
import type { Role } from '../composables/useAuth'

const auth = useAuth()
const router = useRouter()
const route = useRoute()

const username = ref('')
const role = ref<Role>('admin')
const error = ref('')

function login() {
  if (!username.value.trim()) {
    error.value = '请输入用户名'
    return
  }
  auth.login(role.value, username.value.trim())
  const redirect = (route.query.redirect as string) || '/'
  router.push(redirect)
}
</script>

<template>
  <div class="login-wrap">
    <div class="login-card">
      <div class="login-logo">
        <span class="logo-dot"></span>
        <h1>Carrot AI Infra</h1>
        <p>AI 推理服务平台</p>
      </div>

      <div v-if="error" class="error-box">{{ error }}</div>

      <div class="form-row">
        <label>用户名</label>
        <input v-model="username" placeholder="请输入用户名" @keyup.enter="login" />
      </div>
      <div class="form-row">
        <label>角色</label>
        <div class="role-select">
          <button
            class="role-btn"
            :class="{ active: role === 'admin' }"
            @click="role = 'admin'"
          >
            <span class="role-icon">👑</span>
            <span>
              <strong>管理员</strong>
              <small>模型/服务/API Key 全量管理</small>
            </span>
          </button>
          <button
            class="role-btn"
            :class="{ active: role === 'user' }"
            @click="role = 'user'"
          >
            <span class="role-icon">👤</span>
            <span>
              <strong>普通用户</strong>
              <small>查看模型与服务，调用 API</small>
            </span>
          </button>
        </div>
      </div>

      <button class="primary login-btn" @click="login">登 录</button>
    </div>
  </div>
</template>

<style scoped>
.login-wrap {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: radial-gradient(ellipse at top, #1a2440 0%, var(--bg) 60%);
}
.login-card {
  width: 400px;
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 36px;
}
.login-logo {
  text-align: center;
  margin-bottom: 28px;
}
.logo-dot {
  display: inline-block;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: var(--primary);
  box-shadow: 0 0 12px var(--primary);
  margin-bottom: 12px;
}
.login-logo h1 {
  font-size: 22px;
  margin-bottom: 6px;
}
.login-logo p {
  color: var(--text-dim);
  font-size: 13px;
}
.form-row {
  margin-bottom: 18px;
}
.role-select {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}
.role-btn {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px;
  display: flex;
  gap: 10px;
  align-items: center;
  text-align: left;
}
.role-btn.active {
  border-color: var(--primary);
  background: rgba(79, 140, 255, 0.1);
}
.role-btn strong {
  display: block;
  font-size: 14px;
  margin-bottom: 4px;
}
.role-btn small {
  color: var(--text-dim);
  font-size: 11px;
  line-height: 1.4;
}
.role-icon {
  font-size: 22px;
}
.login-btn {
  width: 100%;
  padding: 12px;
  font-size: 15px;
  margin-top: 6px;
}
</style>
