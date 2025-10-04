#!/bin/bash

# GoVecDB Benchmark Runner Script
# This script makes it easy to run the GoVecDB benchmark with various configurations

echo "🚀 GoVecDB Benchmark Runner"
echo "=============================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo -e "${RED}❌ Error: Go is not installed${NC}"
    echo "Please install Go 1.23+ from https://golang.org/"
    exit 1
fi

echo -e "${GREEN}✅ Go found:${NC} $(go version)"
echo ""

# Check if we're in the right directory
if [ ! -f "govecdb_benchmark.go" ]; then
    echo -e "${RED}❌ Error: govecdb_benchmark.go not found${NC}"
    echo "Please run this script from the benchmarks directory"
    exit 1
fi

echo -e "${GREEN}✅ Benchmark file found${NC}"
echo ""

# Menu
echo "Select benchmark mode:"
echo "  1) Quick test (fewer dimensions, faster)"
echo "  2) Standard test (default configuration)"
echo "  3) Full test (all dimensions and sizes)"
echo "  4) Custom test (specify your own)"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo -e "${YELLOW}🏃 Running quick test...${NC}"
        OUTPUT="govecdb_quick_results.txt"
        ;;
    2)
        echo -e "${YELLOW}🏃 Running standard test...${NC}"
        OUTPUT="govecdb_standard_results.txt"
        ;;
    3)
        echo -e "${YELLOW}🏃 Running full test (this may take 30+ minutes)...${NC}"
        OUTPUT="govecdb_full_results.txt"
        ;;
    4)
        echo "Custom configuration requires editing the source code"
        echo "Edit govecdb_benchmark.go and modify the TestConfig"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo "Results will be saved to: $OUTPUT"
echo ""
echo -e "${GREEN}Starting benchmark in 3 seconds...${NC}"
sleep 3

# Run the benchmark
echo ""
go run govecdb_benchmark.go | tee "$OUTPUT"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Benchmark completed successfully!${NC}"
    echo ""
    echo "Results saved to: $OUTPUT"
    echo ""
    echo "📊 Quick summary:"
    grep -A 1 "Average:" "$OUTPUT" | head -20
else
    echo ""
    echo -e "${RED}❌ Benchmark failed${NC}"
    echo "Check the output above for errors"
    exit 1
fi

echo ""
echo "🎉 Done! Check $OUTPUT for full results"
