#!/bin/bash

# Build binaries
echo "Building binaries..."
go build -o vecdb cmd/server/main.go
go build -o bench cmd/benchmark_suite/main.go

# Configuration
dim=6000
n=1000
RESULTS_FILE="benchmark_6000_results.txt"

echo "Benchmark Results for Dim=$dim (N=$n)" > $RESULTS_FILE
echo "| Dimension | Insertion Rate (ops/s) | Recall@10 | Search QPS | Avg Latency |" >> $RESULTS_FILE
echo "|-----------|------------------------|-----------|------------|-------------|" >> $RESULTS_FILE

echo "Running Benchmark: Dim=$dim, N=$n"

WAL_FILE="wal-6000-${n}.log"
SERVER_LOG="server-6000.log"

# Start Server
./vecdb -port 8093 -host 127.0.0.1 -dim $dim -wal $WAL_FILE > $SERVER_LOG 2>&1 &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Run Benchmark
TEMP_OUT="bench_temp_6000.txt"
./bench -addr 127.0.0.1:8093 -dim $dim -n $n -duration 5s > $TEMP_OUT 2>&1

# Parse results
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
while lsof -i:8093 >/dev/null; do
    echo "Waiting for port 8093 to be free..."
    PID=$(lsof -t -i:8093)
    if [ -n "$PID" ]; then
        kill -9 $PID 2>/dev/null
    fi
    sleep 1
done

# Cleanup files
rm -f $WAL_FILE $TEMP_OUT $SERVER_LOG

echo "Done! Results in $RESULTS_FILE"
cat $RESULTS_FILE
