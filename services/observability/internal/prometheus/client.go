// Package prometheus Prometheus HTTP API 查询客户端。
// 通过 promql 查询 DCGM 与推理指标；无 Prometheus 时 observability 降级内存存储。
package prometheus

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

// Client Prometheus HTTP API 客户端
type Client struct {
	baseURL string
	http    *http.Client
}

// NewClient 创建客户端
func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: baseURL,
		http:    &http.Client{Timeout: 15 * time.Second},
	}
}

// QueryResult promql 查询结果
type QueryResult struct {
	Status string `json:"status"`
	Data   struct {
		ResultType string `json:"resultType"`
		Result     []struct {
			Metric map[string]string `json:"metric"`
			Value  []interface{}     `json:"value"` // [ts, "value"]
		} `json:"result"`
	} `json:"data"`
}

// QueryRangeResult 范围查询结果
type QueryRangeResult struct {
	Status string `json:"status"`
	Data   struct {
		ResultType string `json:"resultType"`
		Result     []struct {
			Metric map[string]string `json:"metric"`
			Values [][]interface{}   `json:"values"` // [[ts, "value"], ...]
		} `json:"result"`
	} `json:"data"`
}

// Query 即时查询
func (c *Client) Query(ctx context.Context, q string) (*QueryResult, error) {
	u := c.baseURL + "/api/v1/query?query=" + url.QueryEscape(q)
	body, err := c.get(ctx, u)
	if err != nil {
		return nil, err
	}
	var res QueryResult
	if err := json.Unmarshal(body, &res); err != nil {
		return nil, fmt.Errorf("解析 Prometheus 响应失败: %w", err)
	}
	if res.Status != "success" {
		return nil, fmt.Errorf("Prometheus 查询失败: %s", res.Status)
	}
	return &res, nil
}

// QueryRange 范围查询
func (c *Client) QueryRange(ctx context.Context, q string, from, to time.Time, step time.Duration) (*QueryRangeResult, error) {
	params := url.Values{}
	params.Set("query", q)
	params.Set("start", fmt.Sprintf("%d", from.Unix()))
	params.Set("end", fmt.Sprintf("%d", to.Unix()))
	params.Set("step", fmt.Sprintf("%ds", int(step.Seconds())))
	u := c.baseURL + "/api/v1/query_range?" + params.Encode()
	body, err := c.get(ctx, u)
	if err != nil {
		return nil, err
	}
	var res QueryRangeResult
	if err := json.Unmarshal(body, &res); err != nil {
		return nil, fmt.Errorf("解析 Prometheus 响应失败: %w", err)
	}
	if res.Status != "success" {
		return nil, fmt.Errorf("Prometheus 查询失败: %s", res.Status)
	}
	return &res, nil
}

func (c *Client) get(ctx context.Context, u string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("调用 Prometheus 失败: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Prometheus 返回 %d", resp.StatusCode)
	}
	return io.ReadAll(io.LimitReader(resp.Body, 10<<20))
}
