// Package server observability HTTP API。
// 提供指标上报与查询接口：
//
//	POST /api/v1/metrics/requests   上报请求指标（gateway 调用）
//	POST /api/v1/metrics/gpu        上报 GPU 利用率（采集器调用）
//	GET  /api/v1/deployments/{id}/metrics?range=1h  查询部署请求指标
//	GET  /api/v1/gpus/metrics?range=1h              查询 GPU 利用率
package server

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"time"

	"kk-infra/lib/apitypes"
	"kk-infra/lib/errcode"
	"kk-infra/lib/middleware"
	"kk-infra/services/observability/internal/metrics"
)

// Server observability HTTP 服务
type Server struct {
	store  *metrics.Store
	logger *slog.Logger
}

// NewServer 创建 HTTP 服务。
func NewServer(store *metrics.Store, logger *slog.Logger) *Server {
	return &Server{store: store, logger: logger}
}

// Handler 返回带中间件的路由。
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	// 上报
	mux.HandleFunc("POST /api/v1/metrics/requests", s.handleRecordRequest)
	mux.HandleFunc("POST /api/v1/metrics/gpu", s.handleRecordGPU)

	// 查询
	mux.HandleFunc("GET /api/v1/deployments/{id}/metrics", s.handleDeploymentMetrics)
	mux.HandleFunc("GET /api/v1/gpus/metrics", s.handleGPUMetrics)

	return middleware.WithRequestID(
		middleware.Recover(s.logger,
			middleware.AccessLog(s.logger, mux),
		),
	)
}

// ---- 上报 ----

type recordRequestReq struct {
	DeploymentID string `json:"deploymentId"`
	Model        string `json:"model"`
	LatencyMs    int64  `json:"latencyMs"`
	TTFTMs       int64  `json:"ttftMs"`
	Tokens       int    `json:"tokens"`
	Err          bool   `json:"err"`
}

func (s *Server) handleRecordRequest(w http.ResponseWriter, r *http.Request) {
	var req recordRequestReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		apitypes.WriteResult(w, r, nil, errcode.New(errcode.ErrBadRequest, "请求体解析失败: "+err.Error()))
		return
	}
	if req.DeploymentID == "" {
		apitypes.WriteResult(w, r, nil, errcode.New(errcode.ErrBadRequest, "deploymentId 必填"))
		return
	}
	s.store.RecordRequest(metrics.Sample{
		Ts:           time.Now(),
		DeploymentID: req.DeploymentID,
		Model:        req.Model,
		LatencyMs:    req.LatencyMs,
		TTFTMs:       req.TTFTMs,
		Tokens:       req.Tokens,
		Err:          req.Err,
	})
	apitypes.WriteResult(w, r, map[string]string{"status": "ok"}, nil)
}

type recordGPUReq struct {
	NodeName    string  `json:"nodeName"`
	GPUType     string  `json:"gpuType"`
	Utilization float64 `json:"utilization"` // 0-100
	Used        int32   `json:"used"`
	Total       int32   `json:"total"`
}

func (s *Server) handleRecordGPU(w http.ResponseWriter, r *http.Request) {
	var req recordGPUReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		apitypes.WriteResult(w, r, nil, errcode.New(errcode.ErrBadRequest, "请求体解析失败: "+err.Error()))
		return
	}
	s.store.RecordGPU(metrics.GPUSample{
		Ts:          time.Now(),
		NodeName:    req.NodeName,
		GPUType:     req.GPUType,
		Utilization: req.Utilization,
		Used:        req.Used,
		Total:       req.Total,
	})
	apitypes.WriteResult(w, r, map[string]string{"status": "ok"}, nil)
}

// ---- 查询 ----

// handleDeploymentMetrics 查询某部署的请求指标序列。
func (s *Server) handleDeploymentMetrics(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	rg := metrics.ParseRange(r.URL.Query().Get("range"), time.Now())
	series := s.store.DeploymentMetrics(id, rg)

	view := apitypes.MetricsView{
		DeploymentID: id,
		Range:        r.URL.Query().Get("range"),
	}
	if r.URL.Query().Get("range") == "" {
		view.Range = "1h"
	}
	if len(series.Buckets) == 0 {
		view.Series = []apitypes.MetricSeries{}
		apitypes.WriteResult(w, r, view, nil)
		return
	}

	view.Series = []apitypes.MetricSeries{
		{Name: "requests", Points: seriesPoints(series.Buckets, func(b metrics.Bucket) float64 { return float64(b.Requests) })},
		{Name: "errorRate", Points: seriesPoints(series.Buckets, func(b metrics.Bucket) float64 { return b.ErrorRate() })},
		{Name: "ttftMs", Points: seriesPoints(series.Buckets, func(b metrics.Bucket) float64 { return b.AvgTTFT() })},
		{Name: "tokensPerSec", Points: seriesPoints(series.Buckets, func(b metrics.Bucket) float64 { return b.TokensPerSec(series.BucketSecs) })},
	}
	apitypes.WriteResult(w, r, view, nil)
}

// handleGPUMetrics 查询 GPU 利用率序列（按节点）。
func (s *Server) handleGPUMetrics(w http.ResponseWriter, r *http.Request) {
	rg := metrics.ParseRange(r.URL.Query().Get("range"), time.Now())
	gs := s.store.GPUMetrics(rg)
	buckets := gs.AvgGPUUtil()

	view := apitypes.MetricsView{
		DeploymentID: "gpu",
		Range:        r.URL.Query().Get("range"),
	}
	if r.URL.Query().Get("range") == "" {
		view.Range = "1h"
	}
	if len(buckets) == 0 {
		view.Series = []apitypes.MetricSeries{}
		apitypes.WriteResult(w, r, view, nil)
		return
	}
	view.Series = []apitypes.MetricSeries{
		{Name: "gpuUtil", Points: seriesPoints(buckets, func(b metrics.Bucket) float64 { return float64(b.LatencySum) })},
	}
	apitypes.WriteResult(w, r, view, nil)
}

func seriesPoints(buckets []metrics.Bucket, f func(metrics.Bucket) float64) []apitypes.MetricPoint {
	pts := make([]apitypes.MetricPoint, 0, len(buckets))
	for _, b := range buckets {
		pts = append(pts, apitypes.MetricPoint{Ts: b.Ts, Val: f(b)})
	}
	return pts
}
