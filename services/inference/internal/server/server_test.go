package server

import (
	"bufio"
	"bytes"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func newTestSrv(t *testing.T) http.Handler {
	t.Helper()
	logger := slog.New(slog.NewTextHandler(&bytes.Buffer{}, nil))
	s := NewServer(logger, "qwen-demo")
	s.ttftDelay = 5 * time.Millisecond
	s.extraDelay = 1 * time.Millisecond
	return s.Handler()
}

// 非流式响应
func TestChatNonStream(t *testing.T) {
	h := newTestSrv(t)
	body := `{"model":"qwen-demo","messages":[{"role":"user","content":"你好"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("状态码 = %d", rec.Code)
	}
	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("解析失败: %v", err)
	}
	if len(resp.Choices) == 0 || resp.Choices[0].Message.Content == "" {
		t.Fatal("缺少回复内容")
	}
	if resp.Usage.TotalTokens <= 0 {
		t.Fatal("缺少 token 统计")
	}
}

// 流式响应（SSE）
func TestChatStream(t *testing.T) {
	h := newTestSrv(t)
	body := `{"model":"qwen-demo","messages":[{"role":"user","content":"讲个故事"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("状态码 = %d", rec.Code)
	}
	ct := rec.Header().Get("Content-Type")
	if !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("流式响应 Content-Type 错误: %s", ct)
	}
	var chunks []Chunk
	scanner := bufio.NewScanner(bytes.NewReader(rec.Body.Bytes()))
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			continue
		}
		var ch Chunk
		if err := json.Unmarshal([]byte(data), &ch); err == nil {
			chunks = append(chunks, ch)
		}
	}
	if len(chunks) == 0 {
		t.Fatal("流式响应无数据块")
	}
	// 首块有 delta 内容
	if chunks[0].Choices[0].Delta.Content == "" && chunks[0].Choices[0].Delta.Role == "" {
		t.Fatal("首块应包含内容")
	}
	// 末块 finish_reason = stop
	last := chunks[len(chunks)-1]
	if last.Choices[0].FinishReason == nil || *last.Choices[0].FinishReason != "stop" {
		t.Fatal("末块应带 finish_reason=stop")
	}
}

// 模型不存在
func TestModelNotFound(t *testing.T) {
	h := newTestSrv(t)
	body := `{"model":"unknown","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("未知模型应返回 404: %d", rec.Code)
	}
}

// /v1/models
func TestListModels(t *testing.T) {
	h := newTestSrv(t)
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("状态码 = %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "qwen-demo") {
		t.Fatalf("模型列表缺少 qwen-demo: %s", rec.Body.String())
	}
}

// 健康检查
func TestHealth(t *testing.T) {
	h := newTestSrv(t)
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("健康检查失败: %d", rec.Code)
	}
}
