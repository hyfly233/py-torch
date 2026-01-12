// Package errcode 定义统一错误码，贯穿所有服务。
package errcode

import "fmt"

// Code 业务错误码
type Code int

// 错误码定义（P0 统一）
const (
	OK                   Code = 0
	ErrInternal          Code = 1000 // 内部错误
	ErrBadRequest        Code = 1001 // 请求参数错误
	ErrNotFound          Code = 1002 // 资源不存在
	ErrConflict          Code = 1003 // 资源冲突（如重名）
	ErrModelNotFound     Code = 1004 // 模型不存在
	ErrModelVersionFound Code = 1005 // 模型版本不存在
	ErrModelNotDeployable Code = 1006 // 模型版本不可部署
	ErrQuotaExceeded     Code = 1007 // 租户配额不足
	ErrInsufficientGPU   Code = 1008 // GPU 资源不足
	ErrIllegalState      Code = 1009 // 非法状态转换
	ErrDeploymentRunning Code = 1010 // 部署正在运行，禁止操作
	ErrUnauthorized      Code = 1011 // 未授权 / API Key 无效
	ErrRateLimited       Code = 1012 // 限流
	ErrTimeout           Code = 1013 // 超时
	ErrUpstream          Code = 1014 // 上游（模型服务）错误
)

// Error 统一业务错误
type Error struct {
	Code    Code   `json:"code"`
	Message string `json:"message"`
	RequestID string `json:"requestId,omitempty"`
	Err     error  `json:"-"`
}

func (e *Error) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("[%d] %s: %v", e.Code, e.Message, e.Err)
	}
	return fmt.Sprintf("[%d] %s", e.Code, e.Message)
}

func (e *Error) Unwrap() error { return e.Err }

// New 创建业务错误
func New(code Code, msg string) *Error {
	return &Error{Code: code, Message: msg}
}

// Wrap 创建带底层错误的业务错误
func Wrap(code Code, msg string, err error) *Error {
	return &Error{Code: code, Message: msg, Err: err}
}

// FromErr 判断错误是否为业务错误，是则返回；否则包装为内部错误
func FromErr(err error) *Error {
	if err == nil {
		return nil
	}
	if e, ok := err.(*Error); ok {
		return e
	}
	return Wrap(ErrInternal, "内部错误", err)
}
