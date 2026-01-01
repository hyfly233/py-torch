<script setup lang="ts">
// GPU 资源页：汇总卡片 + 节点表格 + 型号筛选
import { computed, onMounted, ref } from 'vue'
import { fetchGPUs } from '../api'
import type { GPUResourcesView } from '../types'

const loading = ref(true)
const error = ref('')
const gpus = ref<GPUResourcesView | null>(null)
const filterType = ref('')
const search = ref('')

const summaryCards = computed(() => {
  const s = gpus.value?.summary
  return [
    { label: 'GPU 总量', value: s?.totalGpu ?? 0, color: 'var(--primary)' },
    { label: '已用', value: s?.usedGpu ?? 0, color: 'var(--warning)' },
    { label: '可用', value: s?.availableGpu ?? 0, color: 'var(--success)' },
    { label: '异常', value: s?.errorGpu ?? 0, color: 'var(--danger)' },
  ]
})

const gpuTypes = computed(() => {
  const s = gpus.value?.summary
  return s ? Object.keys(s.byType ?? {}) : []
})

const filteredNodes = computed(() => {
  let nodes = gpus.value?.nodes ?? []
  if (filterType.value) {
    nodes = nodes.filter((n) => n.gpuType === filterType.value)
  }
  const q = search.value.trim().toLowerCase()
  if (q) {
    nodes = nodes.filter(
      (n) => n.nodeName.toLowerCase().includes(q) || n.gpuType.toLowerCase().includes(q),
    )
  }
  return nodes
})

async function load() {
  loading.value = true
  error.value = ''
  try {
    gpus.value = await fetchGPUs()
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
      <h2>GPU 资源</h2>
      <button class="ghost" @click="load">刷新</button>
    </div>

    <div v-if="error" class="error-box">{{ error }}</div>
    <div v-if="loading" class="empty">加载中...</div>

    <template v-else>
      <div class="grid-4">
        <div v-for="c in summaryCards" :key="c.label" class="stat-card">
          <div class="label">{{ c.label }}</div>
          <div class="value" :style="{ color: c.color }">{{ c.value }}</div>
        </div>
      </div>

      <div class="panel">
        <div class="flex-between mb-16">
          <div class="flex">
            <select v-model="filterType" style="width: 160px">
              <option value="">全部型号</option>
              <option v-for="t in gpuTypes" :key="t" :value="t">{{ t }}</option>
            </select>
            <input v-model="search" placeholder="搜索节点..." style="width: 200px" />
          </div>
          <span class="dim">节点数：{{ filteredNodes.length }}</span>
        </div>

        <div v-if="!filteredNodes.length" class="empty">暂无 GPU 节点</div>
        <table v-else>
          <thead>
            <tr><th>节点</th><th>型号</th><th>GPU 数</th><th>已用</th><th>可用</th><th>显存</th><th>利用率</th><th>状态</th></tr>
          </thead>
          <tbody>
            <tr v-for="n in filteredNodes" :key="n.nodeName">
              <td><strong>{{ n.nodeName }}</strong></td>
              <td>{{ n.gpuType }}</td>
              <td>{{ n.total }}</td>
              <td>{{ n.used }}</td>
              <td>{{ n.allocatable - n.used }}</td>
              <td>{{ (n.memoryMB / 1024).toFixed(0) }} GB</td>
              <td>
                <div class="util-wrap">
                  <div class="util-bar"><div class="util-fill" :style="{ width: n.utilization + '%' }"></div></div>
                  <span>{{ n.utilization }}%</span>
                </div>
              </td>
              <td>
                <span class="badge" :class="n.health === 'Healthy' ? 'green' : 'red'">{{ n.health }}</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>
  </div>
</template>

<style scoped>
.dim {
  color: var(--text-dim);
}
.ok {
  color: var(--success);
  font-weight: 600;
}
.util-wrap {
  display: flex;
  align-items: center;
  gap: 8px;
}
.util-bar {
  width: 80px;
  height: 6px;
  border-radius: 3px;
  background: var(--bg);
  overflow: hidden;
}
.util-fill {
  height: 100%;
  background: var(--primary);
  border-radius: 3px;
}
</style>
