// 状态 → 徽章样式映射工具

// 部署状态 → badge class
export function statusBadge(status: string): string {
  switch (status) {
    case 'RUNNING':
      return 'green'
    case 'FAILED':
      return 'red'
    case 'STARTING':
    case 'SUBMITTING':
    case 'VALIDATING':
      return 'blue'
    case 'SCALING':
    case 'RESTARTING':
    case 'DELETING':
      return 'orange'
    case 'NEW':
      return 'blue'
    case 'DELETED':
    default:
      return 'gray'
  }
}

// 模型版本状态 → badge class
export function versionBadge(status: string): string {
  switch (status) {
    case 'VALIDATED':
      return 'green'
    case 'VALIDATING':
      return 'blue'
    case 'REGISTERED':
      return 'orange'
    default:
      return 'gray'
  }
}

// 时间格式化
export function fmtTime(iso: string): string {
  if (!iso) return '-'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString('zh-CN', { hour12: false })
}
