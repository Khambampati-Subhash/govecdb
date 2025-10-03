# 🚀 GoVecDB Makefile
# Professional build automation for GoVecDB

# 📋 Variables
BINARY_NAME=govecdb
DEMO_BINARY=govecdb-demo
VERSION?=$(shell git describe --tags --always --dirty)
BUILD_TIME=$(shell date +%Y-%m-%dT%H:%M:%S%z)
LDFLAGS=-ldflags "-X main.Version=${VERSION} -X main.BuildTime=${BUILD_TIME}"

# 🎯 Go configuration
GOBASE=$(shell pwd)
GOBIN=$(GOBASE)/bin
GOOS?=$(shell go env GOOS)
GOARCH?=$(shell go env GOARCH)

# 📊 Test configuration
COVERAGE_OUT=coverage.out
COVERAGE_HTML=coverage.html
BENCH_OUTPUT=bench.out

# 🏗️ Build targets
.PHONY: help build clean test bench fmt lint vet demo examples install deps
.DEFAULT_GOAL := help

## 📖 help: Show this help message
help:
	@echo "🚀 GoVecDB Build System"
	@echo "======================="
	@echo ""
	@echo "📋 Available targets:"
	@sed -n 's/^##//p' $(MAKEFILE_LIST) | column -t -s ':' | sed -e 's/^/ /'
	@echo ""
	@echo "🎯 Quick commands:"
	@echo "  make test    - Run all tests"
	@echo "  make demo    - Run demo application"
	@echo "  make bench   - Run benchmarks"
	@echo "  make fmt     - Format code"

## 🧹 clean: Remove build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf $(GOBIN)
	@rm -f $(BINARY_NAME) $(DEMO_BINARY)
	@rm -f $(COVERAGE_OUT) $(COVERAGE_HTML) $(BENCH_OUTPUT)
	@rm -rf dist/
	@go clean -cache
	@echo "✅ Clean complete"

## 📦 deps: Download and tidy dependencies
deps:
	@echo "📦 Managing dependencies..."
	@go mod download
	@go mod tidy
	@go mod verify
	@echo "✅ Dependencies updated"

## 🏗️ build: Build the library (validates compilation)
build: deps
	@echo "🏗️ Building GoVecDB library..."
	@go build -v ./...
	@echo "✅ Build successful"

## 🎮 demo: Run all demo examples
demo: build
	@echo "🎮 Running HNSW index demo..."
	@go run examples/basic-usage/hnsw_demo.go
	@echo ""
	@echo "🚀 Running collection demo..."
	@go run examples/advanced-features/collection_demo.go

## 📝 examples: Build all examples
examples: deps
	@echo "📝 Building examples..."
	@mkdir -p $(GOBIN)
	@go build -o $(GOBIN)/hnsw-demo $(LDFLAGS) examples/basic-usage/hnsw_demo.go
	@go build -o $(GOBIN)/collection-demo $(LDFLAGS) examples/advanced-features/collection_demo.go
	@echo "✅ Examples built in $(GOBIN)/"

## 🧪 test: Run all tests
test: deps
	@echo "🧪 Running tests..."
	@go test -v -race ./...
	@echo "✅ All tests passed"

## 🚀 test-fast: Run tests without race detection (faster)
test-fast: deps
	@echo "🚀 Running fast tests..."
	@go test ./... -v -cover
	@echo "✅ Fast tests passed"

## 🔬 test-integration: Run integration tests only
test-integration: deps
	@echo "🔬 Running integration tests..."
	@go test -v ./tests/...
	@echo "✅ Integration tests passed"

## 📊 coverage: Generate test coverage report
coverage: deps
	@echo "📊 Generating coverage report..."
	@go test -coverprofile=$(COVERAGE_OUT) ./...
	@go tool cover -html=$(COVERAGE_OUT) -o $(COVERAGE_HTML)
	@go tool cover -func=$(COVERAGE_OUT)
	@echo "📈 Coverage report generated: $(COVERAGE_HTML)"

## ⚡ bench: Run all benchmarks
bench: deps
	@echo "⚡ Running benchmarks..."
	@go test -bench=. -benchmem -run=^Benchmark ./... | tee $(BENCH_OUTPUT)
	@echo "📊 Benchmark results saved to $(BENCH_OUTPUT)"

## 🎯 bench-index: Run index benchmarks only
bench-index: deps
	@echo "🎯 Running index benchmarks..."
	@go test -bench=BenchmarkHNSW -benchmem ./index/

## 🔥 bench-collection: Run collection benchmarks only
bench-collection: deps
	@echo "🔥 Running collection benchmarks..."
	@go test -bench=BenchmarkCollection -benchmem ./collection/

## 🌪️ bench-cluster: Run cluster benchmarks only
bench-cluster: deps
	@echo "🌪️ Running cluster benchmarks..."
	@go test -bench=. -benchmem ./cluster/

## 🎨 fmt: Format all Go code
fmt:
	@echo "🎨 Formatting code..."
	@gofmt -s -w .
	@go mod tidy
	@echo "✅ Code formatted"

## 🔍 lint: Run linting tools
lint: deps
	@echo "🔍 Running linters..."
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run; \
	else \
		echo "⚠️  golangci-lint not installed, skipping..."; \
		echo "💡 Install with: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest"; \
	fi
	@echo "✅ Linting complete"

## ✅ vet: Run go vet
vet: deps
	@echo "✅ Running go vet..."
	@go vet ./...
	@echo "✅ Vet complete"

## 🔧 tools: Install development tools
tools:
	@echo "🔧 Installing development tools..."
	@go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	@go install golang.org/x/tools/cmd/goimports@latest
	@go install github.com/securecodewarrior/sast-scan@latest
	@echo "✅ Development tools installed"

## 🎪 demo-verbose: Run demo with verbose output
demo-verbose: demo

## 📈 perf: Run performance profiling
perf: examples
	@echo "📈 Running performance profiling..."
	@mkdir -p profiles/
	@go test -cpuprofile=profiles/cpu.prof -memprofile=profiles/mem.prof -bench=. ./index/
	@echo "📊 Profiles saved to profiles/"

## 🔬 profile-cpu: Analyze CPU profile
profile-cpu:
	@echo "🔬 Analyzing CPU profile..."
	@go tool pprof profiles/cpu.prof

## 🧠 profile-mem: Analyze memory profile
profile-mem:
	@echo "🧠 Analyzing memory profile..."
	@go tool pprof profiles/mem.prof

## 🏭 build-all: Cross-compile for all platforms
build-all: deps
	@echo "🏭 Cross-compiling for all platforms..."
	@mkdir -p dist/
	@GOOS=linux GOARCH=amd64 go build -o dist/$(DEMO_BINARY)-linux-amd64 $(LDFLAGS) ./cmd/demo/
	@GOOS=linux GOARCH=arm64 go build -o dist/$(DEMO_BINARY)-linux-arm64 $(LDFLAGS) ./cmd/demo/
	@GOOS=darwin GOARCH=amd64 go build -o dist/$(DEMO_BINARY)-darwin-amd64 $(LDFLAGS) ./cmd/demo/
	@GOOS=darwin GOARCH=arm64 go build -o dist/$(DEMO_BINARY)-darwin-arm64 $(LDFLAGS) ./cmd/demo/
	@GOOS=windows GOARCH=amd64 go build -o dist/$(DEMO_BINARY)-windows-amd64.exe $(LDFLAGS) ./cmd/demo/
	@echo "✅ Cross-compilation complete, binaries in dist/"

## 📦 package: Create release packages
package: build-all
	@echo "📦 Creating release packages..."
	@cd dist && for file in $(DEMO_BINARY)-*; do \
		if [[ $$file == *.exe ]]; then \
			zip "$${file%.exe}.zip" "$$file"; \
		else \
			tar -czf "$$file.tar.gz" "$$file"; \
		fi; \
	done
	@echo "✅ Release packages created in dist/"

## 🚀 install: Install demo to GOPATH/bin
install: deps
	@echo "🚀 Installing to GOPATH/bin..."
	@go install $(LDFLAGS) ./cmd/demo/
	@echo "✅ Installed as $(shell go env GOPATH)/bin/demo"

## 🎯 check: Run all quality checks
check: fmt vet lint test
	@echo "🎯 All quality checks passed! ✅"

## 🔥 ci: Full CI pipeline
ci: deps check bench coverage
	@echo "🔥 CI pipeline completed successfully! 🎉"

## 🧪 test-chaos: Run chaos engineering tests
test-chaos: deps
	@echo "🧪 Running chaos engineering tests..."
	@go test -v -run=TestChaos ./cluster/
	@echo "✅ Chaos tests passed"

## 📚 docs: Generate documentation
docs:
	@echo "📚 Generating documentation..."
	@go doc -all > docs/API.md
	@echo "✅ Documentation generated in docs/"

## 🎉 release-check: Pre-release validation
release-check: clean ci
	@echo "🎉 Release validation completed!"
	@echo "📋 Checklist:"
	@echo "  ✅ Dependencies updated"
	@echo "  ✅ Code formatted"
	@echo "  ✅ All tests passed" 
	@echo "  ✅ Benchmarks completed"
	@echo "  ✅ Coverage generated"
	@echo "🚀 Ready for release!"

## 🔧 dev-setup: Setup development environment
dev-setup: deps tools
	@echo "🔧 Setting up development environment..."
	@git config --local core.hooksPath .githooks
	@chmod +x .githooks/*
	@echo "✅ Development environment ready!"

## 🎮 quick-demo: Quick demo run (limited output)
quick-demo: build
	@echo "🎮 Running quick demo..."
	@go run examples/basic-usage/hnsw_demo.go | head -50

# 📊 Utility targets for specific scenarios
watch-test:
	@echo "👀 Watching for changes and running tests..."
	@find . -name "*.go" | entr -r make test-fast

serve-coverage: coverage
	@echo "🌐 Serving coverage report at http://localhost:8080"
	@python3 -m http.server 8080 --directory .

# 🎯 Version information
version:
	@echo "GoVecDB Version: $(VERSION)"
	@echo "Build Time: $(BUILD_TIME)"
	@echo "Go Version: $(shell go version)"
	@echo "OS/Arch: $(GOOS)/$(GOARCH)"
