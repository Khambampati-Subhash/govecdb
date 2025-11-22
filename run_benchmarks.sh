#!/bin/bash

# Build binaries
echo "Building binaries..."
go build -o vecdb cmd/server/main.go
go build -o bench cmd/benchmark_suite/main.go

# Define configurations
DIMS=(256 512 1024 2048 3072 4096)
SIZES=(1000 3000 5000 10000)

RESULTS_FILE="benchmark_results.txt"
echo "Benchmark Results" > $RESULTS_FILE

for dim in "${DIMS[@]}"; do
    for n in "${SIZES[@]}"; do
        echo "------------------------------------------------" | tee -a $RESULTS_FILE
        echo "Running Benchmark: Dim=$dim, N=$n" | tee -a $RESULTS_FILE
        
        WAL_FILE="wal-${dim}-${n}.log"
        
        # Start Server
        ./vecdb -port 8080 -host 127.0.0.1 -dim $dim -wal $WAL_FILE > server.log 2>&1 &
        SERVER_PID=$!
        
        # Wait for server to start
        sleep 2
        
        # Run Benchmark
        ./bench -addr 127.0.0.1:8080 -dim $dim -n $n >> $RESULTS_FILE
        
        # Stop Server
        kill $SERVER_PID
        wait $SERVER_PID 2>/dev/null
        
        # Ensure port is free
        while lsof -i:8080 >/dev/null; do
            echo "Waiting for port 8080 to be free..."
            # Find PID using port 8080 and kill it
            PID=$(lsof -t -i:8080)
            if [ -n "$PID" ]; then
                kill -9 $PID 2>/dev/null
            fi
            sleep 1
        done
        
        # Cleanup WAL
        rm -f $WAL_FILE
        
        echo "------------------------------------------------" | tee -a $RESULTS_FILE
    done
done

echo "Benchmarks Complete. Results saved to $RESULTS_FILE"
