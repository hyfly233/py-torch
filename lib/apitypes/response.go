package apitypes

import (
	"encoding/json"
	"net/http"

	"kk-infra/lib/errcode"
	"kk-infra/lib/middleware"
)

// WriteResult 统一响应写出：成功返回 data，失败返回错误码/信息。
// 所有服务复用，保证响应格式一致。
func WriteResult(w http.ResponseWriter, r *http.Request, data interface{}, err error) {
	resp := Response{
		RequestID: middleware.GetRequestID(r.Context()),
	}
	status := http.StatusOK
	if err != nil {
		be := errcode.FromErr(err)
		resp.Code = int(be.Code)
		resp.Message = be.Message
		status = HTTPStatusFor(be.Code)
	} else {
		resp.Data = data
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(resp)
}

// HTTPStatusFor 业务错误码 → HTTP 状态码
func HTTPStatusFor(code errcode.Code) int {
	switch code {
	case errcode.ErrBadRequest:
		return http.StatusBadRequest
	case errcode.ErrNotFound, errcode.ErrModelNotFound, errcode.ErrModelVersionFound:
		return http.StatusNotFound
	case errcode.ErrConflict, errcode.ErrQuotaExceeded, errcode.ErrInsufficientGPU,
		errcode.ErrIllegalState, errcode.ErrDeploymentRunning:
		return http.StatusConflict
	case errcode.ErrUnauthorized:
		return http.StatusUnauthorized
	case errcode.ErrRateLimited:
		return http.StatusTooManyRequests
	default:
		return http.StatusInternalServerError
	}
}
