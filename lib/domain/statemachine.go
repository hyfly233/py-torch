package domain

import (
	"fmt"
	"sync"
)

// 部署状态机。
//
//	NEW
//	  ↓
//	VALIDATING ──→ FAILED
//	  ↓
//	SUBMITTING ──→ FAILED
//	  ↓
//	STARTING ────→ FAILED
//	  ↓
//	RUNNING
//	  ├── SCALING → RUNNING
//	  ├── RESTARTING → RUNNING
//	  └── DELETING → DELETED
//
// 任何状态（除 DELETED）均可被终止删除 → DELETING。
// FAILED 可重试 → SUBMITTING（重新提交）。

// 合法状态转换表：from → 允许的 to 集合
var deploymentTransitions = map[string]map[string]bool{
	DeploymentStatusNew: {
		DeploymentStatusValidating: true,
		DeploymentStatusFailed:     true,
		DeploymentStatusDeleting:   true,
	},
	DeploymentStatusValidating: {
		DeploymentStatusSubmitting: true,
		DeploymentStatusFailed:     true,
		DeploymentStatusDeleting:   true,
	},
	DeploymentStatusSubmitting: {
		DeploymentStatusStarting: true,
		DeploymentStatusFailed:   true,
		DeploymentStatusDeleting: true,
	},
	DeploymentStatusStarting: {
		DeploymentStatusRunning: true,
		DeploymentStatusFailed:  true,
		DeploymentStatusDeleting: true,
	},
	DeploymentStatusRunning: {
		DeploymentStatusScaling:    true,
		DeploymentStatusRestarting: true,
		DeploymentStatusDeleting:   true,
		DeploymentStatusFailed:     true,
	},
	DeploymentStatusScaling: {
		DeploymentStatusRunning: true,
		DeploymentStatusFailed:  true,
		DeploymentStatusDeleting: true,
	},
	DeploymentStatusRestarting: {
		DeploymentStatusRunning:  true,
		DeploymentStatusFailed:   true,
		DeploymentStatusDeleting: true,
	},
	DeploymentStatusDeleting: {
		DeploymentStatusDeleted: true,
		DeploymentStatusFailed:  true,
	},
	DeploymentStatusFailed: {
		DeploymentStatusSubmitting: true, // 重试
		DeploymentStatusDeleting:   true,
	},
	DeploymentStatusDeleted: {},
}

// DeploymentStateMachine 部署状态机（线程安全）
type DeploymentStateMachine struct {
	mu sync.RWMutex
}

// NewDeploymentStateMachine 创建状态机
func NewDeploymentStateMachine() *DeploymentStateMachine {
	return &DeploymentStateMachine{}
}

// CanTransition 判断 from → to 是否合法
func (sm *DeploymentStateMachine) CanTransition(from, to string) bool {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	if tos, ok := deploymentTransitions[from]; ok {
		return tos[to]
	}
	return false
}

// Transition 执行状态转换，非法转换返回错误
func (sm *DeploymentStateMachine) Transition(from, to string) error {
	if !sm.CanTransition(from, to) {
		return fmt.Errorf("非法状态转换: %s → %s", from, to)
	}
	return nil
}
