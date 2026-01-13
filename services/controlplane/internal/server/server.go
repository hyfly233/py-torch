// Package server controlplane HTTP 服务
package server

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"time"

	"kk-infra/lib/apitypes"
	"kk-infra/lib/domain"
	"kk-infra/lib/errcode"
	"kk-infra/lib/middleware"
	"kk-infra/services/controlplane/internal/biz"
	"kk-infra/services/controlplane/internal/clients"
	"kk-infra/services/controlplane/internal/data"
)

// Server 控制面 HTTP 服务
type Server struct {
	deployments   *biz.DeploymentUseCase
	resources     *biz.ResourceUseCase
	repo          data.DeploymentRepository
	logger        *slog.Logger
	observability *clients.ObservabilityClient // 可为 nil（未配置时返回占位）
}

// NewServer 创建服务
func NewServer(deployments *biz.DeploymentUseCase, resources *biz.ResourceUseCase, repo data.DeploymentRepository, logger *slog.Logger) *Server {
	return &Server{deployments: deployments, resources: resources, repo: repo, logger: logger}
}

// SetObservabilityClient 注入可观测性客户端（未配置时指标返回占位）。
func (s *Server) SetObservabilityClient(c *clients.ObservabilityClient) {
	s.observability = c
}

// Handler 路由
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	// 部署
	mux.HandleFunc("POST /api/v1/deployments", s.handleCreateDeployment)
	mux.HandleFunc("GET /api/v1/deployments", s.handleListDeployments)
	mux.HandleFunc("GET /api/v1/deployments/{id}", s.handleGetDeployment)
	mux.HandleFunc("POST /api/v1/deployments/{id}/scale", s.handleScaleDeployment)
	mux.HandleFunc("POST /api/v1/deployments/{id}/restart", s.handleRestartDeployment)
	mux.HandleFunc("DELETE /api/v1/deployments/{id}", s.handleDeleteDeployment)
	mux.HandleFunc("GET /api/v1/deployments/{id}/metrics", s.handleDeploymentMetrics)

	// 资源
	mux.HandleFunc("GET /api/v1/resources/gpus", s.handleListGPUs)

	return middleware.WithRequestID(
		middleware.Recover(s.logger,
			middleware.AccessLog(s.logger, mux),
		),
	)
}

// ---- 部署 ----

func (s *Server) handleCreateDeployment(w http.ResponseWriter, r *http.Request) {
	var req apitypes.CreateDeploymentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		apitypes.WriteResult(w, r, nil, errcode.New(errcode.ErrBadRequest, "请求体解析失败: "+err.Error()))
		return
	}
	d, err := s.deployments.CreateDeployment(r.Context(), &req)
	if err != nil {
		apitypes.WriteResult(w, r, nil, err)
		return
	}
	apitypes.WriteResult(w, r, d, nil)
}

func (s *Server) handleListDeployments(w http.ResponseWriter, r *http.Request) {
	tenant := r.URL.Query().Get("tenant")
	list, err := s.deployments.ListDeployments(tenant)
	if err != nil {
		apitypes.WriteResult(w, r, nil, err)
		return
	}
	apitypes.WriteResult(w, r, list, nil)
}

func (s *Server) handleGetDeployment(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	d, err := s.deployments.GetDeployment(id)
	if err != nil {
		apitypes.WriteResult(w, r, nil, err)
		return
	}
	// 附加事件
	view := apitypes.DeploymentView{ModelDeployment: *d}
	events := s.repo.Events(id)
	for _, e := range events {
		view.Events = append(view.Events, apitypes.EventView{
			Type:    "Status",
			Reason:  e.To,
			Message: e.Reason,
			At:      e.At,
		})
	}
	apitypes.WriteResult(w, r, view, nil)
}

func (s *Server) handleScaleDeployment(w http.ResponseWriter, r *http.Request) {
	var req apitypes.ScaleDeploymentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		apitypes.WriteResult(w, r, nil, errcode.New(errcode.ErrBadRequest, "请求体解析失败: "+err.Error()))
		return
	}
	d, err := s.deployments.ScaleDeployment(r.Context(), r.PathValue("id"), req.Replicas)
	if err != nil {
		apitypes.WriteResult(w, r, nil, err)
		return
	}
	apitypes.WriteResult(w, r, d, nil)
}

func (s *Server) handleRestartDeployment(w http.ResponseWriter, r *http.Request) {
	d, err := s.deployments.RestartDeployment(r.Context(), r.PathValue("id"))
	if err != nil {
		apitypes.WriteResult(w, r, nil, err)
		return
	}
	apitypes.WriteResult(w, r, d, nil)
}

func (s *Server) handleDeleteDeployment(w http.ResponseWriter, r *http.Request) {
	if err := s.deployments.DeleteDeployment(r.Context(), r.PathValue("id")); err != nil {
		apitypes.WriteResult(w, r, nil, err)
		return
	}
	apitypes.WriteResult(w, r, map[string]bool{"deleted": true}, nil)
}

// handleDeploymentMetrics 指标查询：转发到 observability；未配置时返回占位。
func (s *Server) handleDeploymentMetrics(w http.ResponseWriter, r *http.Request) {
	_, err := s.deployments.GetDeployment(r.PathValue("id"))
	if err != nil {
		apitypes.WriteResult(w, r, nil, err)
		return
	}
	// 已配置 observability：转发真实指标
	if s.observability != nil {
		view, err := s.observability.DeploymentMetrics(r.Context(), r.PathValue("id"), r.URL.Query().Get("range"))
		if err != nil {
			s.logger.Warn("查询 observability 指标失败", "deploymentId", r.PathValue("id"), "err", err)
			apitypes.WriteResult(w, r, nil, err)
			return
		}
		apitypes.WriteResult(w, r, *view, nil)
		return
	}
	// 占位（observability 未启动）
	view := apitypes.MetricsView{
		DeploymentID: r.PathValue("id"),
		Range:        r.URL.Query().Get("range"),
		Series: []apitypes.MetricSeries{
			{Name: "requests", Points: []apitypes.MetricPoint{{Ts: time.Now().Unix(), Val: 0}}},
		},
	}
	apitypes.WriteResult(w, r, view, nil)
}

// ---- 资源 ----

func (s *Server) handleListGPUs(w http.ResponseWriter, r *http.Request) {
	gpuType := r.URL.Query().Get("gpuType")
	summary, nodes, err := s.resources.ListGPUResources(r.Context(), gpuType)
	if err != nil {
		apitypes.WriteResult(w, r, nil, err)
		return
	}
	apitypes.WriteResult(w, r, apitypes.GPUResourcesView{Summary: *summary, Nodes: nodes}, nil)
}

// 辅助：确保 domain 引用（类型断言场景预留）
var _ = domain.RuntimeVLLM
