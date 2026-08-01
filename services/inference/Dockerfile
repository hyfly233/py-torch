# inference 服务（Mock vLLM）镜像，本机 Docker Desktop K8s 验证用。
# 生产环境替换为真实 vLLM 镜像。
# 构建上下文必须是仓库根目录：docker build -f services/inference/Dockerfile .
FROM golang:1.26-alpine AS build
WORKDIR /src
# 复制整个仓库（go.work 多模块）
COPY go.work go.work.sum* ./
COPY lib/ lib/
COPY services/ services/
WORKDIR /src/services/inference
# 关闭 workspace 模式，用本地 replace 构建
ENV GOWORK=off
RUN go mod download
RUN CGO_ENABLED=0 go build -o /bin/inference ./cmd

FROM alpine:3.20
# 使用清华 alpine 镜像源（国内网络）
RUN sed -i 's|dl-cdn.alpinelinux.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apk/repositories \
    && apk add --no-cache ca-certificates tzdata
COPY --from=build /bin/inference /usr/local/bin/inference
EXPOSE 8000
# "*" 通配模式：mock 响应任意模型名（验证链路用；生产替换为真实 vLLM）
ENTRYPOINT ["/usr/local/bin/inference", "--addr", ":8000", "--model", "*"]
