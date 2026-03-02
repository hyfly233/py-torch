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

// 三级展开：Pool（型号）→ 节点 → GPU 卡
interface PoolGroup {
  gpuType: string
  nodes: typeof filteredNodes.value
  expanded: boolean
}
const expandedPools = ref<Record<string, boolean>>({})
const expandedNodes = ref<Record<string, boolean>>({})

const pools = computed<PoolGroup[]>(() => {
  const byType = new Map<string, typeof filteredNodes.value>()
  for (const n of filteredNodes.value) {
    const list = byType.get(n.gpuType) ?? []
    list.push(n)
    byType.set(n.gpuType, list)
  }
  return Array.from(byType.entries()).map(([gpuType, nodes]) => ({
    gpuType,
    nodes,
    expanded: expandedPools.value[gpuType] ?? true, // 默认展开 Pool
  }))
})

function togglePool(gpuType: string) {
  expandedPools.value[gpuType] = !(expandedPools.value[gpuType] ?? true)
}
function toggleNode(nodeName: string) {
  expandedNodes.value[nodeName] = !expandedNodes.value[nodeName]
}
function isNodeExpanded(nodeName: string) {
  return expandedNodes.value[nodeName] ?? false // 节点默认折叠，点击展开 GPU 卡
}

// 模拟单节点内 GPU 卡列表（按 Total 生成）
function gpuCards(node: { total: number; memoryMB: number; used: number; utilization: number }) {
  const cards = []
  for (let i = 0; i < node.total; i++) {
    // 简单模拟：前 used 张已分配
    const used = i < node.used
    cards.push({
      id: i + 1,
      used,
      memoryMB: node.memoryMB,
      utilization: used ? node.utilization : 0,
    })
  }
  return cards
}

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

        <div v-if="!pools.length" class="empty">暂无 GPU 节点</div>
        <div v-else class="pool-tree">
          <!-- 一级：Pool（型号） -->
          <div v-for="p in pools" :key="p.gpuType" class="pool-group">
            <div class="pool-header" @click="togglePool(p.gpuType)">
              <span class="chevron">{{ p.expanded ? '▼' : '▶' }}</span>
              <strong>{{ p.gpuType }} Pool</strong>
              <span class="dim">· {{ p.nodes.length }} 节点 · {{ p.nodes.reduce((s, n) => s + n.total, 0) }} GPU</span>
            </div>

            <!-- 二级：节点 -->
            <div v-if="p.expanded" class="pool-body">
              <div v-for="n in p.nodes" :key="n.nodeName" class="node-group">
                <div class="node-header" @click="toggleNode(n.nodeName)">
                  <span class="chevron">{{ isNodeExpanded(n.nodeName) ? '▼' : '▶' }}</span>
                  <strong>{{ n.nodeName }}</strong>
                  <span class="badge" :class="n.health === 'Healthy' ? 'green' : 'red'">{{ n.health }}</span>
                  <span class="dim node-stats">
                    {{ n.total }} GPU · 已用 {{ n.used }} · 可用 {{ n.allocatable - n.used }}
                    · {{ (n.memoryMB / 1024).toFixed(0) }} GB/卡
                    · 利用率 {{ n.utilization }}%
                  </span>
                </div>

                <!-- 三级：GPU 卡 -->
                <div v-if="isNodeExpanded(n.nodeName)" class="node-body">
                  <div
                    v-for="card in gpuCards(n)"
                    :key="card.id"
                    class="gpu-card"
                    :class="{ used: card.used }"
                    :title="`GPU-${card.id} · ${card.used ? '已分配' : '空闲'} · ${(card.memoryMB / 1024).toFixed(0)}GB · 利用率 ${card.utilization}%`"
                  >
                    <span class="gpu-id">GPU-{{ card.id }}</span>
                    <span class="gpu-state">{{ card.used ? '已分配' : '空闲' }}</span>
                    <span class="gpu-util">{{ card.used ? card.utilization + '%' : '-' }}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
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
/* 三级展开树 */
.pool-tree {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.pool-group {
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}
.pool-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 16px;
  background: var(--bg-hover);
  cursor: pointer;
  font-size: 14px;
}
.pool-body {
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.node-group {
  border: 1px solid rgba(42, 53, 80, 0.6);
  border-radius: 6px;
  overflow: hidden;
}
.node-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  background: var(--bg);
  cursor: pointer;
  font-size: 13px;
}
.node-stats {
  font-size: 12px;
}
.node-body {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 10px 14px;
}
.gpu-card {
  width: 96px;
  padding: 10px 12px;
  border-radius: 6px;
  background: rgba(34, 197, 94, 0.08);
  border: 1px solid rgba(34, 197, 94, 0.3);
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 11px;
  cursor: default;
}
.gpu-card.used {
  background: rgba(239, 68, 68, 0.08);
  border-color: rgba(239, 68, 68, 0.3);
}
.gpu-id {
  font-weight: 600;
  color: var(--text);
}
.gpu-state {
  color: var(--success);
}
.gpu-card.used .gpu-state {
  color: var(--danger);
}
.gpu-util {
  color: var(--text-dim);
}
.chevron {
  display: inline-block;
  width: 14px;
  color: var(--text-dim);
}
</style>
