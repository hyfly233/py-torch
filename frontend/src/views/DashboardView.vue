<script setup lang="ts">
// 总览 Dashboard：GPU 资源汇总 + 在线/异常服务 + 最近部署
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { fetchDeployments, fetchGPUs } from '../api'
import { useAuth } from '../composables/useAuth'
import type { Deployment, GPUResourcesView } from '../types'
import { statusBadge } from '../utils/status'

const router = useRouter()
const { isAdmin } = useAuth()
const loading = ref(true)
const error = ref('')

const gpus = ref<GPUResourcesView | null>(null)
const deployments = ref<Deployment[]>([])

// 统计卡片
const statCards = computed(() => {
  const s = gpus.value?.summary
  const running = deployments.value.filter((d) => d.status === 'RUNNING').length
  const failed = deployments.value.filter((d) => d.status === 'FAILED').length
  return [
    { label: 'GPU 总量', value: s?.totalGpu ?? 0, color: 'var(--primary)' },
    { label: '可用 GPU', value: s?.availableGpu ?? 0, color: 'var(--success)' },
    { label: '在线服务', value: running, color: 'var(--info)' },
    // 异常为 0 时用中性色，>0 才标红
    { label: '异常服务', value: failed, color: failed > 0 ? 'var(--danger)' : 'var(--text-dim)' },
  ]
})

const recentDeployments = computed(() => deployments.value.slice(0, 6))

async function load() {
  loading.value = true
  error.value = ''
  try {
    const [g, d] = await Promise.all([fetchGPUs(), fetchDeployments()])
    gpus.value = g
    deployments.value = d
  } catch (e) {
    error.value = (e as Error).message
  } finally {
    loading.value = false
  }
}

onMounted(load)
</script>

<template>
  <div class="page">
    <div class="page-header">
      <h2>集群总览</h2>
      <div class="flex">
        <button class="ghost" @click="load">刷新</button>
        <RouterLink v-if="isAdmin" to="/deployments/new">
          <button class="primary">部署服务</button>
        </RouterLink>
      </div>
    </div>

    <div v-if="error" class="error-box">{{ error }}</div>
    <div v-if="loading" class="empty">加载中...</div>

    <template v-else>
      <div class="grid-4">
        <div v-for="c in statCards" :key="c.label" class="stat-card">
          <div class="label">{{ c.label }}</div>
          <div class="value" :style="{ color: c.color }">{{ c.value }}</div>
        </div>
      </div>

      <div class="grid-2">
        <div class="panel">
          <div class="panel-title">GPU 节点利用率</div>
          <div v-if="!gpus?.nodes.length" class="empty">暂无 GPU 节点</div>
          <table v-else>
            <thead>
              <tr><th>节点</th><th>型号</th><th>利用率</th><th>状态</th></tr>
            </thead>
            <tbody>
              <tr v-for="n in gpus?.nodes" :key="n.nodeName">
                <td>{{ n.nodeName }}</td>
                <td>{{ n.gpuType }}</td>
                <td>
                  <div class="util-bar">
                    <div class="util-fill" :style="{ width: n.utilization + '%' }"></div>
                  </div>
                  <span class="util-text">{{ n.utilization }}%</span>
                </td>
                <td><span class="badge" :class="n.health === 'Healthy' ? 'green' : 'red'">{{ n.health }}</span></td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="panel">
          <div class="panel-title">最近部署</div>
          <div v-if="!recentDeployments.length" class="empty">暂无部署记录</div>
          <table v-else>
            <thead>
              <tr><th>服务</th><th>模型</th><th>状态</th></tr>
            </thead>
            <tbody>
              <tr
                v-for="d in recentDeployments"
                :key="d.id"
                class="clickable"
                @click="router.push(`/deployments/${d.id}`)"
              >
                <td>{{ d.name }}</td>
                <td>{{ d.modelName }}:{{ d.modelVersion }}</td>
                <td><span class="badge" :class="statusBadge(d.status)">{{ d.status }}</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.clickable {
  cursor: pointer;
}
.util-bar {
  display: inline-block;
  width: 90px;
  height: 6px;
  border-radius: 3px;
  background: var(--bg);
  overflow: hidden;
  vertical-align: middle;
  margin-right: 8px;
}
.util-fill {
  height: 100%;
  background: var(--primary);
  border-radius: 3px;
}
.util-text {
  font-size: 12px;
  color: var(--text-dim);
}
</style>
