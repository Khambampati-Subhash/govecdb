# Optimized GoVecDB Docker Configuration

# Multi-stage build for optimal production image
FROM golang:1.23-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git make gcc musl-dev

# Set working directory
WORKDIR /app

# Copy go mod files first for better layer caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build optimized binary
RUN CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build \
    -ldflags='-w -s -extldflags "-static"' \
    -a -installsuffix cgo \
    -tags netgo \
    -o govecdb \
    ./cmd/server

# Production stage
FROM alpine:latest

# Install runtime dependencies
RUN apk --no-cache add ca-certificates tzdata

# Create non-root user
RUN addgroup -g 1001 govecdb && \
    adduser -D -s /bin/sh -u 1001 -G govecdb govecdb

# Set working directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/govecdb .

# Create data directory with proper permissions
RUN mkdir -p /data /logs && \
    chown -R govecdb:govecdb /app /data /logs

# Set resource limits
ENV GOGC=100 \
    GOMAXPROCS=0 \
    GOMEMLIMIT=1GiB

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ./govecdb health || exit 1

# Switch to non-root user
USER govecdb

# Expose ports
EXPOSE 8080 8081 8082

# Default command
CMD ["./govecdb", "serve", "--config=/app/config.yaml"]

# Labels for metadata
LABEL maintainer="GoVecDB Team" \
      version="2.0.0" \
      description="High-performance vector database" \
      org.opencontainers.image.source="https://github.com/Khambampati-Subhash/govecdb"