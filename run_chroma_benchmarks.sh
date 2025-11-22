#!/bin/bash

# Build binary
echo "Building ChromaDB benchmark..."
go build -o bench_chroma cmd/benchmark_chroma/main.go

# Define configurations
DIMS=(256 512 1024 2048 3072 4096)
SIZES=(1000 3000 5000 10000)

RESULTS_FILE="chroma_benchmark_results.txt"
echo "ChromaDB Benchmark Results" > $RESULTS_FILE

echo "NOTE: Ensure ChromaDB is running on localhost:8000"
echo "Docker command: docker run -p 8000:8000 chromadb/chroma:0.4.24"

for dim in "${DIMS[@]}"; do
    for n in "${SIZES[@]}"; do
        echo "------------------------------------------------" | tee -a $RESULTS_FILE
        echo "Running Benchmark: Dim=$dim, N=$n" | tee -a $RESULTS_FILE
        
        ./bench_chroma -addr http://localhost:8000 -dim $dim -n $n >> $RESULTS_FILE
        
        echo "------------------------------------------------" | tee -a $RESULTS_FILE
    done
done

echo "Benchmarks Complete. Results saved to $RESULTS_FILE"
