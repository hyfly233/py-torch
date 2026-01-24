package renderer

import (
	"testing"

	"kk-infra/lib/domain"
)

func TestRender_VLLMDeployment(t *testing.T) {
	in := &RenderInput{
		DeploymentID: "d1",
		Name:         "qwen-demo",
		Namespace:    "tenant-default",
		TenantID:     "default",
		ModelID:      "m1",
		ModelVersion: "7b",
		ModelPath:    "/models/qwen-7b",
		Replicas:     1,
		Resource: domain.Resource{
			GPUType:  "A100",
			GPUCount: 1,
			MemoryMB: 32768,
		},
		StartupArgs: []string{"--gpu-memory-utilization", "0.9"},
	}
	out, err := Render(in)
	if err != nil {
		t.Fatalf("渲染失败: %v", err)
	}

	// Deployment 断言
	d := out.Deployment
	if d.Metadata.Name != "qwen-demo" {
		t.Errorf("名称错误: %s", d.Metadata.Name)
	}
	if d.Spec.Replicas != 1 {
		t.Errorf("副本数错误: %d", d.Spec.Replicas)
	}
	// 标签
	for k, want := range map[string]string{
		LabelDeploymentID: "d1",
		LabelManagedBy:    ManagedByValue,
		LabelTenantID:     "default",
	} {
		if got := d.Metadata.Labels[k]; got != want {
			t.Errorf("标签 %s = %s, want %s", k, got, want)
		}
	}
	// GPU 限制
	c := d.Spec.Template.Spec.Containers[0]
	if got := c.Resources.Limits["nvidia.com/gpu"]; got != "1" {
		t.Errorf("GPU limit = %s, want 1", got)
	}
	if got := c.Resources.Limits["memory"]; got != "32768Mi" {
		t.Errorf("内存 limit = %s, want 32768Mi", got)
	}
	// 节点选择器
	ns := d.Spec.Template.Spec.NodeSelector
	if ns["nvidia.com/gpu.type"] != "A100" {
		t.Errorf("节点选择器 GPU 类型错误: %v", ns)
	}
	// 启动参数注入
	found := false
	for _, a := range c.Args {
		if a == "--gpu-memory-utilization" {
			found = true
		}
	}
	if !found {
		t.Errorf("自定义启动参数未注入: %v", c.Args)
	}
	// 探针
	if c.ReadinessProbe == nil || c.ReadinessProbe.HTTPGet == nil {
		t.Fatal("缺少就绪探针")
	}
	if c.ReadinessProbe.HTTPGet.Path != "/health" {
		t.Errorf("就绪探针路径错误: %s", c.ReadinessProbe.HTTPGet.Path)
	}

	// Service 断言
	svc := out.Service
	if svc.Spec.Selector["app"] != "qwen-demo" {
		t.Errorf("Service selector 错误: %v", svc.Spec.Selector)
	}
	if len(svc.Spec.Ports) != 1 || svc.Spec.Ports[0].Port != 80 || svc.Spec.Ports[0].TargetPort != 8000 {
		t.Errorf("Service 端口错误: %+v", svc.Spec.Ports)
	}

	// Secret/ConfigMap
	if out.Secret.Metadata.Name != "qwen-demo-model" {
		t.Errorf("Secret 名称错误: %s", out.Secret.Metadata.Name)
	}
	if out.ConfigMap.Data["vllm-args.yaml"] == "" {
		t.Error("ConfigMap 缺少启动参数")
	}
}

func TestRender_InvalidInput(t *testing.T) {
	_, err := Render(&RenderInput{Name: "", Namespace: "ns"})
	if err == nil {
		t.Fatal("空名称应报错")
	}
}

func TestToClientSpec(t *testing.T) {
	d := &domain.ModelDeployment{
		ID:       "d1",
		Name:     "demo",
		ModelID:  "m1",
		TenantID: "t1",
		Replicas: 2,
		Resource: domain.Resource{GPUType: "A100", GPUCount: 1},
	}
	spec := ToClientSpec(d, "/models/qwen")
	if spec.DeploymentID != "d1" || spec.Replicas != 2 || spec.ModelPath != "/models/qwen" {
		t.Errorf("spec 转换错误: %+v", spec)
	}
	if spec.Labels[LabelDeploymentID] != "d1" {
		t.Errorf("标签丢失: %+v", spec.Labels)
	}
}
