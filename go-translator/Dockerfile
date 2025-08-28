FROM golang:1.22-alpine AS builder
ENV GOTOOLCHAIN=auto
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o /go-translator ./

FROM gcr.io/distroless/base-debian12
WORKDIR /
COPY --from=builder /go-translator /go-translator
EXPOSE 6565
USER 65532:65532
ENTRYPOINT ["/go-translator"]


