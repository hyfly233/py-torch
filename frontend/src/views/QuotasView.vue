<script setup lang="ts">
// 租户与配额页（管理员）：配额/已用/调整配额
import { onMounted, ref } from 'vue'
import { fetchQuotas, setQuota, type TenantQuota } from '../api'

const loading = ref(true)
const error = ref('')
const quotas = ref<TenantQuota[]>([])

// 调整配额弹窗
const showEdit = ref(false)
const editTarget = ref<TenantQuota | null>(null)
const editQuota = ref(0)
const editing = ref(false)
const editError = ref('')

async function load() {
  loading.value = true
  error.value = ''
  try {
    quotas.value = await fetchQuotas()
  } catch (e) {
    error.value = (e as Error).message
  } finally {
    loading.value = false
  }
}

function openEdit(q: TenantQuota) {
  editTarget.value = q
  editQuota.value = q.quota
  editError.value = ''
  showEdit.value = true
}

async function doSet() {
  if (!editTarget.value) return
  editError.value = ''
  if (editQuota.value < 0) {
    editError.value = '配额必须 >= 0'
    return
  }
  editing.value = true
  try {
    await setQuota(editTarget.value.tenantId, editTarget.value.gpuType, editQuota.value)
    showEdit.value = false
    await load()
  } catch (e) {
    editError.value = (e as Error).message
  } finally {
    editing.value = false
  }
}

onMounted(load)
</script>

<template>
  <div class="page">
    <div class="page-header">
      <h2>租户与配额</h2>
      <button class="ghost" @click="load">刷新</button>
    </div>

    <div v-if="error" class="error-box">{{ error }}</div>
    <div v-if="loading" class="empty">加载中...</div>

    <div class="panel" v-else>
      <div v-if="!quotas.length" class="empty">
        暂无配额配置。配额在创建部署时校验：超过配额上限的请求将被拒绝。
      </div>
      <table v-else>
        <thead>
          <tr><th>租户</th><th>GPU 型号</th><th>配额上限</th><th>已用</th><th>可用</th><th>利用率</th><th>操作</th></tr>
        </thead>
        <tbody>
          <tr v-for="q in quotas" :key="q.tenantId + q.gpuType">
            <td><strong>{{ q.tenantId }}</strong></td>
            <td>{{ q.gpuType }}</td>
            <td>{{ q.quota }}</td>
            <td>{{ q.used }}</td>
            <td class="ok">{{ Math.max(0, q.quota - q.used) }}</td>
            <td>
              <div class="util-wrap">
                <div class="util-bar">
                  <div class="util-fill" :style="{ width: (q.quota > 0 ? (q.used / q.quota) * 100 : 0) + '%' }"></div>
                </div>
                <span>{{ q.quota > 0 ? ((q.used / q.quota) * 100).toFixed(0) : 0 }}%</span>
              </div>
            </td>
            <td><button class="ghost" @click="openEdit(q)">调整</button></td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- 调整配额弹窗 -->
    <div v-if="showEdit" class="modal-mask" @click.self="showEdit = false">
      <div class="modal">
        <h3>调整配额：{{ editTarget?.tenantId }} / {{ editTarget?.gpuType }}</h3>
        <div v-if="editError" class="error-box">{{ editError }}</div>
        <div class="form-row">
          <label>配额上限（GPU 数）</label>
          <input v-model.number="editQuota" type="number" min="0" />
        </div>
        <div v-if="editTarget" class="dim">
          当前已用：{{ editTarget.used }} GPU
        </div>
        <div class="flex" style="justify-content: flex-end; margin-top: 16px">
          <button @click="showEdit = false">取消</button>
          <button class="primary" :disabled="editing" @click="doSet">
            {{ editing ? '提交中...' : '确认' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.ok {
  color: var(--success);
  font-weight: 600;
}
.dim {
  color: var(--text-dim);
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
