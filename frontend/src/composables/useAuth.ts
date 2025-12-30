// 角色认证状态（MVP 本地模拟，不接真实认证服务）
import { computed, ref } from 'vue'

export type Role = 'admin' | 'user'

interface AuthState {
  role: Role
  username: string
}

const state = ref<AuthState | null>(null)

const STORAGE_KEY = 'carrot-auth'

// 从 localStorage 恢复
function load() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) state.value = JSON.parse(raw)
  } catch {
    state.value = null
  }
}
load()

export function useAuth() {
  const isAdmin = computed(() => state.value?.role === 'admin')
  const isLoggedIn = computed(() => state.value !== null)
  const role = computed(() => state.value?.role ?? null)
  const username = computed(() => state.value?.username ?? '')

  function login(role_: Role, username_: string) {
    state.value = { role: role_, username: username_ }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state.value))
  }

  function logout() {
    state.value = null
    localStorage.removeItem(STORAGE_KEY)
  }

  return { isAdmin, isLoggedIn, role, username, login, logout }
}
