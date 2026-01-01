<script setup lang="ts">
// 服务详情：状态/指标/事件/API 示例 + 扩容/重启/删除（管理员）
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { deleteDeployment, fetchDeployment, restartDeployment, scaleDeployment } from '../api'
import { useAuth } from '../composables/useAuth'
import type { DeploymentView } from '../types'
import { fmtTime, statusBadge } from '../utils/status'

const route = useRoute()
const router = useRouter()
const { isAdmin } = useAuth()

const id = route.params.id as string
const loading = ref(true)
const error = ref('')
const dep = ref<DeploymentView | null>(null)
const actionError = ref('')

// 扩容弹窗
const showScale = ref(false)
const scaleReplicas = ref(1)
const scaling = ref(false)

const apiExample = computed(() => {
  if (!dep.value) return ''
  const model = dep.value.name
  return `curl https://api.example.com/v1/chat/completions \\
  -H "Authorization: Bearer sk-carrot-***" \\
  -H "Content-Type: application/json" \\
  -d '{"model":"${model}","messages":[{"role":"user","content":"你好"}]}'`
})

async function load() {
  loading.value = true
  error.value = ''
  try {
    dep.value = await fetchDeployment(id)
  } catch (e) {
    error.value = (e as Error).message
  } finally {
    loading.value = false
  }
}

function openScale() {
  if (dep.value) {
    scaleReplicas.value = dep.value.replicas
    showScale.value = true
  }
}

async function doScale() {
  actionError.value = ''
  scaling.value = true
  try {
    dep.value = await scaleDeployment(id, scaleReplicas.value)
    showScale.value = false
  } catch (e) {
    actionError.value = (e as Error).message
  } finally {
    scaling.value = false
  }
}

async function doRestart() {
  actionError.value = ''
  if (!confirm(`确认重启服务「${dep.value?.name}」？`)) return
  try {
    dep.value = await restartDeployment(id)
  } catch (e) {
    actionError.value = (e as Error).message
  }
}

async function doDelete() {
  const d = dep.value
  if (!d) return
  const gpuCount = d.resource.gpuCount * d.replicas
  if (!confirm(`确认删除服务「${d.name}」？将释放 ${gpuCount} 张 GPU，此操作不可恢复。`)) return
  actionError.value = ''
  try {
    await deleteDeployment(id)
    router.push('/deployments')
  } catch (e) {
    actionError.value = (e as Error).message
  }
}

const canManage = computed(() => isAdmin.value && dep.value && !['DELETED', 'DELETING'].includes(dep.value.status))

onMounted(load)
</script>

<template>
  <div class="page">
    <div class="page-header">
      <div>
        <button class="ghost mb-16" @click="router.push('/deployments')">← 返回服务列表</button>
        <h2>{{ dep?.name ?? '...' }}</h2>
      </div>
      <div v-if="canManage" class="flex">
        <button class="ghost" @click="openScale">扩容</button>
        <button class="ghost" @click="doRestart">重启</button>
        <button class="danger" @click="doDelete">删除</button>
      </div>
    </div>

    <div v-if="error" class="error-box">{{ error }}</div>
    <div v-if="actionError" class="error-box">{{ actionError }}</div>
    <div v-if="loading" class="empty">加载中...</div>

    <template v-else-if="dep">
      <!-- 基本信息 -->
      <div class="panel mb-16">
        <div class="flex-between">
          <div class="flex">
            <span class="badge" :class="statusBadge(dep.status)">{{ dep.status }}</span>
            <span class="dim">Endpoint: <span class="mono">{{ dep.endpoint || '-' }}</span></span>
          </div>
        </div>
        <div class="info-grid mt-16">
          <div><label>模型</label><strong>{{ dep.modelName }}:{{ dep.modelVersion }}</strong></div>
          <div><label>运行时</label><strong>{{ dep.runtime }}</strong></div>
          <div><label>副本数</label><strong>{{ dep.replicas }}</strong></div>
          <div><label>GPU</label><strong>{{ dep.resource.gpuType }} × {{ dep.resource.gpuCount }}</strong></div>
          <div><label>内存</label><strong>{{ (dep.resource.memoryMB / 1024).toFixed(0) }} GB</strong></div>
          <div><label>租户</label><strong>{{ dep.tenantId }}</strong></div>
          <div><label>命名空间</label><strong class="mono">{{ dep.namespace }}</strong></div>
          <div><label>创建时间</label><strong>{{ fmtTime(dep.createdAt) }}</strong></div>
        </div>
        <div v-if="dep.diagnostics" class="error-box mt-16">诊断：{{ dep.diagnostics }}</div>
      </div>

      <!-- API 示例 -->
      <div class="panel mb-16">
        <div class="panel-title">API 示例</div>
        <pre>{{ apiExample }}</pre>
      </div>

      <!-- 状态事件 -->
      <div class="panel">
        <div class="panel-title">状态事件</div>
        <div v-if="!dep.events?.length" class="empty">暂无事件</div>
        <table v-else>
          <thead>
            <tr><th>时间</th><th>变更</th><th>原因</th></tr>
          </thead>
          <tbody>
            <tr v-for="(e, i) in dep.events" :key="i">
              <td class="dim">{{ fmtTime(e.at) }}</td>
              <td>
                <span v-if="e.reason" class="badge blue">{{ e.reason }}</span>
              </td>
              <td>{{ e.message }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>

    <!-- 扩容弹窗 -->
    <div v-if="showScale" class="modal-mask" @click.self="showScale = false">
      <div class="modal">
        <h3>扩缩容</h3>
        <div class="form-row">
          <label>副本数（当前 {{ dep?.replicas }}）</label>
          <input v-model.number="scaleReplicas" type="number" min="1" />
        </div>
        <div v-if="dep" class="dim">
          变更后 GPU 占用：{{ dep.resource.gpuCount * scaleReplicas }} 张
        </div>
        <div class="flex" style="justify-content: flex-end; margin-top: 16px">
          <button @click="showScale = false">取消</button>
          <button class="primary" :disabled="scaling" @click="doScale">
            {{ scaling ? '提交中...' : '确认' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.dim {
  color: var(--text-dim);
}
.mono {
  font-family: 'SF Mono', Menlo, monospace;
  font-size: 12px;
}
.info-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}
.info-grid label {
  margin-bottom: 4px;
}
</style>
