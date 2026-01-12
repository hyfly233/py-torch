// Package middleware 提供 HTTP 通用中间件：RequestID、访问日志、panic 恢复。
package middleware

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"log/slog"
	"net/http"
	"time"
)

// RequestIDKey 请求 ID 在 context 中的键
type requestIDKey struct{}

// RequestIDHeader 请求 ID 响应头
const RequestIDHeader = "X-Request-Id"

// GetRequestID 从 context 取请求 ID
func GetRequestID(ctx context.Context) string {
	if v, ok := ctx.Value(requestIDKey{}).(string); ok {
		return v
	}
	return ""
}

// WithRequestID 生成并注入请求 ID（支持透传客户端提供的 ID）
func WithRequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get(RequestIDHeader)
		if id == "" {
			id = newRequestID()
		}
		ctx := context.WithValue(r.Context(), requestIDKey{}, id)
		w.Header().Set(RequestIDHeader, id)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// AccessLog 访问日志中间件：记录方法、路径、状态码、耗时、请求 ID。
// 不记录请求体，避免泄露 API Key / 权重地址。
func AccessLog(logger *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		sw := &statusWriter{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(sw, r)
		logger.Info("http",
			"requestId", GetRequestID(r.Context()),
			"method", r.Method,
			"path", r.URL.Path,
			"status", sw.status,
			"durationMs", time.Since(start).Milliseconds(),
			"remote", r.RemoteAddr,
		)
	})
}

// Recover panic 恢复中间件
func Recover(logger *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				logger.Error("panic recovered",
					"requestId", GetRequestID(r.Context()),
					"panic", rec,
				)
				http.Error(w, `{"code":1000,"message":"内部错误"}`, http.StatusInternalServerError)
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// Chain 串联中间件（外层先执行）
func Chain(logger *slog.Logger, handlers ...func(*slog.Logger, http.Handler) http.Handler) func(http.Handler) http.Handler {
	return func(final http.Handler) http.Handler {
		h := final
		for i := len(handlers) - 1; i >= 0; i-- {
			h = handlers[i](logger, h)
		}
		return h
	}
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}

// Flush 透传流式刷新（支持 SSE 流式响应）
func (w *statusWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func newRequestID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
