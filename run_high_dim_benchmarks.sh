#!/bin/bash

# Build binaries
echo "Building binaries..."
go build -o vecdb cmd/server/main.go
go build -o bench cmd/benchmark_suite/main.go

# Configuration
dims="4096 8192 16384"
n="1000"
RESULTS_FILE="high_dim_benchmark_results.txt"

echo "High Dimension Benchmark Results" > $RESULTS_FILE
echo "------------------------------------------------" >> $RESULTS_FILE

for dim in $dims; do
    echo "Running Benchmark: Dim=$dim, N=$n"
    echo "------------------------------------------------" | tee -a $RESULTS_FILE
    echo "Running Benchmark: Dim=$dim, N=$n" | tee -a $RESULTS_FILE
    
    WAL_FILE="wal-high-${dim}-${n}.log"
    
    # Start Server
    # Using async WAL for fair comparison with raw speed, but we can enable sync if needed.
    # Default is async in our code now.
    ./vecdb -port 8090 -host 127.0.0.1 -dim $dim -wal $WAL_FILE > server-high.log 2>&1 &    # rm server_${DIM}.log 2>&1 &
    SERVER_PID=$!
    
    # Wait for server to start
    sleep 5
    
    # Run Benchmark
    ./bench -addr 127.0.0.1:8090 -dim $dim -n $n >> $RESULTS_FILE
    
    # Cleanup Server
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    
    # Ensure port is free
    while lsof -i:8090 >/dev/null; do
        echo "Waiting for port 8090 to be free..."
        PID=$(lsof -t -i:8090)
        if [ -n "$PID" ]; then
            kill -9 $PID 2>/dev/null
        fi
        sleep 1
    done
    
    # Cleanup WAL
    rm -f $WAL_FILE
    
    echo "------------------------------------------------" | tee -a $RESULTS_FILE
done

echo "Done! Results in $RESULTS_FILE"
