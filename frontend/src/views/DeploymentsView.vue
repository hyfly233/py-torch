<script setup lang="ts">
// 模型服务列表：状态/副本/操作（管理员可扩容/删除）
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { deleteDeployment, fetchDeployments } from '../api'
import { useAuth } from '../composables/useAuth'
import type { Deployment } from '../types'
import { statusBadge } from '../utils/status'

const router = useRouter()
const { isAdmin } = useAuth()

const loading = ref(true)
const error = ref('')
const deployments = ref<Deployment[]>([])
const search = ref('')

const filtered = () => {
  const q = search.value.trim().toLowerCase()
  if (!q) return deployments.value
  return deployments.value.filter(
    (d) =>
      d.name.toLowerCase().includes(q) ||
      d.modelName.toLowerCase().includes(q) ||
      d.status.toLowerCase().includes(q),
  )
}

async function load() {
  loading.value = true
  error.value = ''
  try {
    deployments.value = await fetchDeployments()
  } catch (e) {
    error.value = (e as Error).message
  } finally {
    loading.value = false
  }
}

async function remove(d: Deployment) {
  if (!confirm(`确认删除服务「${d.name}」？将释放 ${d.resource.gpuCount * d.replicas} 张 GPU。`)) return
  try {
    await deleteDeployment(d.id)
    await load()
  } catch (e) {
    alert(`删除失败: ${(e as Error).message}`)
  }
}

onMounted(load)
</script>

<template>
  <div class="page">
    <div class="page-header">
      <h2>模型服务</h2>
      <div class="flex">
        <input v-model="search" placeholder="搜索服务..." class="search-input" />
        <button v-if="isAdmin" class="primary" @click="router.push('/deployments/new')">部署服务</button>
      </div>
    </div>

    <div v-if="error" class="error-box">{{ error }}</div>
    <div v-if="loading" class="empty">加载中...</div>
    <div v-else-if="!filtered().length" class="empty">暂无模型服务</div>

    <div class="panel" v-else>
      <table>
        <thead>
          <tr><th>服务名称</th><th>模型版本</th><th>状态</th><th>副本</th><th>GPU</th><th>Endpoint</th><th>操作</th></tr>
        </thead>
        <tbody>
          <tr v-for="d in filtered()" :key="d.id">
            <td class="svc-name" @click="router.push(`/deployments/${d.id}`)">{{ d.name }}</td>
            <td>{{ d.modelName }}:{{ d.modelVersion }}</td>
            <td><span class="badge" :class="statusBadge(d.status)">{{ d.status }}</span></td>
            <td>{{ d.replicas }}</td>
            <td>{{ d.resource.gpuType }} × {{ d.resource.gpuCount }}</td>
            <td class="dim mono">{{ d.endpoint || '-' }}</td>
            <td>
              <div class="flex">
                <button class="ghost" @click="router.push(`/deployments/${d.id}`)">详情</button>
                <button v-if="isAdmin && d.status === 'FAILED'" class="success" @click="router.push(`/deployments/${d.id}`)">重试</button>
                <button v-if="isAdmin && !['DELETED', 'DELETING'].includes(d.status)" class="danger" @click="remove(d)">删除</button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.search-input {
  width: 220px;
}
.svc-name {
  font-weight: 600;
  color: var(--primary);
  cursor: pointer;
}
.dim {
  color: var(--text-dim);
}
.mono {
  font-family: 'SF Mono', Menlo, monospace;
  font-size: 12px;
}
</style>
