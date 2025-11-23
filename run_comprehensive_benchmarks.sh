#!/bin/bash

# Build binaries
echo "Building binaries..."
go build -o vecdb cmd/server/main.go
go build -o bench cmd/benchmark_suite/main.go

# Configuration
# Spectrum of dimensions from low to high
dims="128 256 512 1024 2048 4096 8192 16384"
n="1000"
RESULTS_FILE="comprehensive_benchmark_results.txt"

echo "Comprehensive Benchmark Results (N=$n)" > $RESULTS_FILE
echo "| Dimension | Insertion Rate (ops/s) | Recall@10 | Search QPS | Avg Latency |" >> $RESULTS_FILE
echo "|-----------|------------------------|-----------|------------|-------------|" >> $RESULTS_FILE

for dim in $dims; do
    echo "Running Benchmark: Dim=$dim, N=$n"
    
    WAL_FILE="wal-comp-${dim}-${n}.log"
    SERVER_LOG="server-comp-${dim}.log"
    
    # Start Server
    # Using async WAL for fair comparison
    ./vecdb -port 8092 -host 127.0.0.1 -dim $dim -wal $WAL_FILE > $SERVER_LOG 2>&1 &
    SERVER_PID=$!
    
    # Wait for server to start
    sleep 3
    
    # Run Benchmark
    # Capture output to a temporary file to parse
    TEMP_OUT="bench_temp_${dim}.txt"
    ./bench -addr 127.0.0.1:8092 -dim $dim -n $n -duration 2s > $TEMP_OUT 2>&1
    
    # Parse results
    # Example output format:
    # Insertion Rate: 222.62 ops/s
    # Recall@10: 0.6240
    # Search QPS: 9193.12
    # Avg Latency: 108.776µs
    
    insert_rate=$(grep "Insertion Rate:" $TEMP_OUT | awk '{print $3}')
    recall=$(grep "Recall@10:" $TEMP_OUT | awk '{print $2}')
    search_qps=$(grep "Search QPS:" $TEMP_OUT | awk '{print $3}')
    latency=$(grep "Avg Latency:" $TEMP_OUT | awk '{print $3}')
    
    # Append to results table
    echo "| $dim | $insert_rate | $recall | $search_qps | $latency |" >> $RESULTS_FILE
    
    # Print to console
    cat $TEMP_OUT
    echo "------------------------------------------------"
    
    # Cleanup Server
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    
    # Ensure port is free
    while lsof -i:8092 >/dev/null; do
        echo "Waiting for port 8092 to be free..."
        PID=$(lsof -t -i:8092)
        if [ -n "$PID" ]; then
            kill -9 $PID 2>/dev/null
        fi
        sleep 1
    done
    
    # Cleanup files
    rm -f $WAL_FILE $TEMP_OUT $SERVER_LOG
done

echo "Done! Results in $RESULTS_FILE"
cat $RESULTS_FILE
