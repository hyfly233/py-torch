package domain

import "testing"

// 状态机合法/非法转换测试
func TestStateMachine_ValidTransitions(t *testing.T) {
	sm := NewDeploymentStateMachine()

	cases := []struct {
		from string
		to   string
		ok   bool
	}{
		{DeploymentStatusNew, DeploymentStatusValidating, true},
		{DeploymentStatusValidating, DeploymentStatusSubmitting, true},
		{DeploymentStatusSubmitting, DeploymentStatusStarting, true},
		{DeploymentStatusStarting, DeploymentStatusRunning, true},
		{DeploymentStatusRunning, DeploymentStatusScaling, true},
		{DeploymentStatusScaling, DeploymentStatusRunning, true},
		{DeploymentStatusRunning, DeploymentStatusRestarting, true},
		{DeploymentStatusRestarting, DeploymentStatusRunning, true},
		{DeploymentStatusRunning, DeploymentStatusDeleting, true},
		{DeploymentStatusDeleting, DeploymentStatusDeleted, true},
		// 非法转换
		{DeploymentStatusNew, DeploymentStatusRunning, false},
		{DeploymentStatusNew, DeploymentStatusDeleted, false},
		{DeploymentStatusRunning, DeploymentStatusValidating, false},
		{DeploymentStatusStarting, DeploymentStatusSubmitting, false},
		{DeploymentStatusDeleted, DeploymentStatusRunning, false},
		{DeploymentStatusDeleted, DeploymentStatusDeleting, false},
		// FAILED 重试路径
		{DeploymentStatusStarting, DeploymentStatusFailed, true},
		{DeploymentStatusFailed, DeploymentStatusSubmitting, true},
		{DeploymentStatusFailed, DeploymentStatusDeleting, true},
		// 任意状态可终止删除
		{DeploymentStatusValidating, DeploymentStatusDeleting, true},
		{DeploymentStatusFailed, DeploymentStatusDeleting, true},
	}

	for _, c := range cases {
		got := sm.CanTransition(c.from, c.to)
		if got != c.ok {
			t.Errorf("CanTransition(%s→%s) = %v, want %v", c.from, c.to, got, c.ok)
		}
	}
}

// 状态机错误信息
func TestStateMachine_TransitionError(t *testing.T) {
	sm := NewDeploymentStateMachine()
	if err := sm.Transition(DeploymentStatusNew, DeploymentStatusRunning); err == nil {
		t.Fatal("期望非法转换报错")
	}
	if err := sm.Transition(DeploymentStatusRunning, DeploymentStatusScaling); err != nil {
		t.Fatalf("合法转换不应报错: %v", err)
	}
}

// 模型版本可部署性
func TestModelVersion_Deployable(t *testing.T) {
	ok := &ModelVersion{
		Status:     ModelStatusValidated,
		Runtime:    RuntimeVLLM,
		ArtifactURI: "s3://bucket/qwen",
		GPUType:    "A100",
		GPUCount:   1,
	}
	if !ok.Deployable() {
		t.Fatal("有效版本应可部署")
	}
	cases := []*ModelVersion{
		{Status: ModelStatusRegistered, Runtime: RuntimeVLLM, ArtifactURI: "s3://x", GPUType: "A100", GPUCount: 1},
		{Status: ModelStatusValidated, Runtime: "Triton", ArtifactURI: "s3://x", GPUType: "A100", GPUCount: 1},
		{Status: ModelStatusValidated, Runtime: RuntimeVLLM, ArtifactURI: "", GPUType: "A100", GPUCount: 1},
		{Status: ModelStatusValidated, Runtime: RuntimeVLLM, ArtifactURI: "s3://x", GPUType: "", GPUCount: 1},
		{Status: ModelStatusValidated, Runtime: RuntimeVLLM, ArtifactURI: "s3://x", GPUType: "A100", GPUCount: 0},
	}
	for i, c := range cases {
		if c.Deployable() {
			t.Errorf("case %d 不应可部署: %+v", i, c)
		}
	}
}

// GPU 资源可用数
func TestGPUResource_Available(t *testing.T) {
	g := &GPUResource{Allocatable: 8, Used: 3}
	if g.Available() != 5 {
		t.Fatalf("Available = %d, want 5", g.Available())
	}
	g = &GPUResource{Allocatable: 2, Used: 5}
	if g.Available() != 0 {
		t.Fatalf("超卖时应返回 0")
	}
}

// 租户配额剩余
func TestTenantQuota_Available(t *testing.T) {
	q := &TenantQuota{Quota: 8, Used: 3}
	if q.Available() != 5 {
		t.Fatalf("Available = %d, want 5", q.Available())
	}
}
