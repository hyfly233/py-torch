// Package renderer 将平台部署请求渲染为 Kubernetes 资源清单。
package renderer

import (
	"fmt"

	"kk-infra/lib/domain"
	"kk-infra/services/k8sadapter/internal/client"
	"kk-infra/services/k8sadapter/internal/k8s"
)

// Carrot 资源标签约定
const (
	LabelDeploymentID = "carrot.ai/deployment-id"
	LabelModelID      = "carrot.ai/model-id"
	LabelModelVersion = "carrot.ai/model-version"
	LabelTenantID     = "carrot.ai/tenant-id"
	LabelManagedBy    = "carrot.ai/managed-by"
	ManagedByValue    = "carrot"
)

// VLLMImage vLLM 镜像（生产可配置，MVP 固定）
const VLLMImage = "vllm/vllm-openai:latest"

// RenderInput 渲染输入
type RenderInput struct {
	DeploymentID  string
	Name          string
	Namespace     string
	TenantID      string
	ModelID       string
	ModelVersion  string
	ModelPath     string // 模型权重路径（挂载到容器）
	Replicas      int32
	Resource      domain.Resource
	StartupArgs   []string
}

// RenderOutput 渲染产物
type RenderOutput struct {
	Deployment *k8s.Deployment
	Service    *k8s.Service
	Secret     *k8s.Secret
	ConfigMap  *k8s.ConfigMap
}

// Render 渲染一组 K8s 资源
func Render(in *RenderInput) (*RenderOutput, error) {
	if in.Name == "" || in.Namespace == "" || in.Resource.GPUCount <= 0 {
		return nil, fmt.Errorf("渲染参数不完整: name=%s ns=%s gpu=%d", in.Name, in.Namespace, in.Resource.GPUCount)
	}
	labels := map[string]string{
		LabelDeploymentID: in.DeploymentID,
		LabelModelID:      in.ModelID,
		LabelModelVersion: in.ModelVersion,
		LabelTenantID:     in.TenantID,
		LabelManagedBy:    ManagedByValue,
		"app":             in.Name,
	}

	// 组装 vLLM 启动参数：--model + 资源相关默认参数
	args := []string{
		"--model", in.ModelPath,
		"--host", "0.0.0.0",
		"--port", "8000",
		"--max-model-len", "8192",
	}
	args = append(args, in.StartupArgs...)

	deployment := &k8s.Deployment{
		APIVersion: "apps/v1",
		Kind:       "Deployment",
		Metadata: k8s.ObjectMeta{
			Name:        in.Name,
			Namespace:   in.Namespace,
			Labels:      labels,
		},
		Spec: k8s.DeploymentSpec{
			Replicas: in.Replicas,
			Selector: &k8s.LabelSelector{MatchLabels: map[string]string{"app": in.Name}},
			Template: k8s.PodTemplateSpec{
				Metadata: k8s.ObjectMeta{Labels: labels},
				Spec: k8s.PodSpec{
					NodeSelector: map[string]string{
						"nvidia.com/gpu.type": in.Resource.GPUType,
					},
					RestartPolicy: "Always",
					Containers: []k8s.Container{
						{
							Name:  "vllm",
							Image: VLLMImage,
							Args:  args,
							Ports: []k8s.ContainerPort{{Name: "http", ContainerPort: 8000}},
							Resources: k8s.ResourceRequirements{
								Limits: map[string]string{
									"nvidia.com/gpu": fmt.Sprintf("%d", in.Resource.GPUCount),
									"memory":         fmt.Sprintf("%dMi", in.Resource.MemoryMB),
								},
								Requests: map[string]string{
									"nvidia.com/gpu": fmt.Sprintf("%d", in.Resource.GPUCount),
									"memory":         fmt.Sprintf("%dMi", in.Resource.MemoryMB),
								},
							},
							ReadinessProbe: &k8s.Probe{
								HTTPGet:    &k8s.HTTPGetAction{Path: "/health", Port: 8000},
								InitialDelaySeconds: 30,
								PeriodSeconds:      10,
							},
							LivenessProbe: &k8s.Probe{
								HTTPGet:    &k8s.HTTPGetAction{Path: "/health", Port: 8000},
								InitialDelaySeconds: 60,
								PeriodSeconds:      15,
							},
						},
					},
				},
			},
		},
	}

	service := &k8s.Service{
		APIVersion: "v1",
		Kind:       "Service",
		Metadata: k8s.ObjectMeta{
			Name:        in.Name,
			Namespace:   in.Namespace,
			Labels:      labels,
		},
		Spec: k8s.ServiceSpec{
			Selector: map[string]string{"app": in.Name},
			Ports:    []k8s.ServicePort{{Name: "http", Port: 80, TargetPort: 8000}},
			Type:     "ClusterIP",
		},
	}

	secret := &k8s.Secret{
		APIVersion: "v1",
		Kind:       "Secret",
		Metadata:   k8s.ObjectMeta{Name: in.Name + "-model", Namespace: in.Namespace, Labels: labels},
		Type:       "Opaque",
		StringData: map[string]string{
			// MVP 不存真实凭证，仅占位；真实凭据由 Secret 管理注入
			"model-path": in.ModelPath,
		},
	}

	configMap := &k8s.ConfigMap{
		APIVersion: "v1",
		Kind:       "ConfigMap",
		Metadata:   k8s.ObjectMeta{Name: in.Name + "-config", Namespace: in.Namespace, Labels: labels},
		Data: map[string]string{
			"vllm-args.yaml": renderArgsYAML(args),
		},
	}

	return &RenderOutput{
		Deployment: deployment,
		Service:    service,
		Secret:     secret,
		ConfigMap:  configMap,
	}, nil
}

// ToClientSpec 由控制面部署对象生成 K8s 客户端入参
func ToClientSpec(d *domain.ModelDeployment, modelPath string) *client.DeploymentSpec {
	return &client.DeploymentSpec{
		DeploymentID: d.ID,
		Name:         d.Name,
		Namespace:    d.Namespace,
		Replicas:     d.Replicas,
		Resource:     d.Resource,
		Image:        VLLMImage,
		Args:         d.StartupArgs,
		Labels: map[string]string{
			LabelDeploymentID: d.ID,
			LabelModelID:      d.ModelID,
			LabelModelVersion: d.ModelVersion,
			LabelTenantID:     d.TenantID,
			LabelManagedBy:    ManagedByValue,
		},
		ModelPath: modelPath,
	}
}

// renderArgsYAML 简单序列化 args（ConfigMap 数据用）
func renderArgsYAML(args []string) string {
	out := "args:\n"
	for _, a := range args {
		out += fmt.Sprintf("  - %q\n", a)
	}
	return out
}
