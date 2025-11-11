#!/usr/bin/env python3
"""
ChromaDB Benchmark - Comprehensive Testing Suite
Tests: Single Insert, Batch Insert, Search (k=1,10,100), Delete, Update
Dimensions: 512, 1024, 2048, 3072, 4096
Vector Counts: 1000, 3000, 5000
"""

import chromadb
import numpy as np
import time
import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import statistics


@dataclass
class BenchmarkResult:
    operation: str
    dimension: int
    vector_count: int
    avg_time_ms: float
    throughput: float  # ops/sec or vec/sec
    recall: float = 0.0  # For search operations
    success_rate: float = 100.0  # For operations that can fail


class ChromaDBBenchmark:
    def __init__(self):
        self.client = chromadb.Client()
        self.results: List[BenchmarkResult] = []
        
    def generate_vectors(self, count: int, dim: int, seed: int = 42) -> tuple:
        """Generate normalized random vectors for testing"""
        np.random.seed(seed)
        
        vectors = []
        ids = []
        metadatas = []
        
        for i in range(count):
            # Generate random vector
            vec = np.random.randn(dim).astype(np.float32)
            
            # Normalize for cosine similarity
            norm = np.linalg.norm(vec)
            vec = vec / norm
            
            vectors.append(vec.tolist())
            ids.append(f"vec_{i}")
            metadatas.append({
                "index": i,
                "category": f"cat_{i % 5}",
                "value": float(i * 0.1)
            })
        
        return vectors, ids, metadatas
    
    def create_collection(self, name: str, dimension: int):
        """Create a new ChromaDB collection"""
        # Delete if exists
        try:
            self.client.delete_collection(name)
        except:
            pass
        
        collection = self.client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
        return collection
    
    def benchmark_single_insert(self, collection, vectors, ids, metadatas, count: int = 100):
        """Benchmark single vector insertion"""
        times = []
        
        for i in range(count):
            start = time.time()
            collection.add(
                embeddings=[vectors[i]],
                ids=[ids[i]],
                metadatas=[metadatas[i]]
            )
            elapsed = (time.time() - start) * 1000  # Convert to ms
            times.append(elapsed)
        
        avg_time = statistics.mean(times)
        throughput = 1000 / avg_time if avg_time > 0 else 0
        
        return BenchmarkResult(
            operation="single_insert",
            dimension=len(vectors[0]),
            vector_count=count,
            avg_time_ms=avg_time,
            throughput=throughput
        )
    
    def benchmark_batch_insert(self, collection, vectors, ids, metadatas):
        """Benchmark batch insertion"""
        start = time.time()
        collection.add(
            embeddings=vectors,
            ids=ids,
            metadatas=metadatas
        )
        elapsed = (time.time() - start) * 1000  # Convert to ms
        
        avg_time = elapsed / len(vectors)
        throughput = len(vectors) / (elapsed / 1000) if elapsed > 0 else 0
        
        return BenchmarkResult(
            operation="batch_insert",
            dimension=len(vectors[0]),
            vector_count=len(vectors),
            avg_time_ms=avg_time,
            throughput=throughput
        )
    
    def benchmark_exact_search(self, collection, vectors, ids, num_searches: int = 100):
        """Benchmark exact search (k=1) and measure recall"""
        np.random.seed(42)
        times = []
        recalls = []
        
        for _ in range(num_searches):
            # Pick a random vector that we inserted
            query_idx = np.random.randint(0, len(vectors))
            query_vec = vectors[query_idx]
            expected_id = ids[query_idx]
            
            start = time.time()
            results = collection.query(
                query_embeddings=[query_vec],
                n_results=1
            )
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            
            # Check if we found the correct vector
            if results['ids'] and len(results['ids'][0]) > 0:
                found_id = results['ids'][0][0]
                recalls.append(1.0 if found_id == expected_id else 0.0)
            else:
                recalls.append(0.0)
        
        avg_time = statistics.mean(times)
        avg_recall = statistics.mean(recalls) * 100
        throughput = 1000 / avg_time if avg_time > 0 else 0
        
        return BenchmarkResult(
            operation="exact_search_k1",
            dimension=len(vectors[0]),
            vector_count=len(vectors),
            avg_time_ms=avg_time,
            throughput=throughput,
            recall=avg_recall
        )
    
    def benchmark_knn_search(self, collection, vectors, ids, k: int, num_searches: int = 100):
        """Benchmark k-NN search"""
        np.random.seed(42)
        times = []
        quality_scores = []
        
        for _ in range(num_searches):
            # Pick a random vector
            query_idx = np.random.randint(0, len(vectors))
            query_vec = vectors[query_idx]
            expected_id = ids[query_idx]
            
            start = time.time()
            results = collection.query(
                query_embeddings=[query_vec],
                n_results=k
            )
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            
            # Calculate quality: position of query vector in results
            if results['ids'] and len(results['ids'][0]) > 0:
                found_ids = results['ids'][0]
                if expected_id in found_ids:
                    position = found_ids.index(expected_id)
                    # Score: 1.0 if first, decreasing with position
                    quality = 1.0 - (position / k)
                    quality_scores.append(quality)
                else:
                    quality_scores.append(0.0)
            else:
                quality_scores.append(0.0)
        
        avg_time = statistics.mean(times)
        avg_quality = statistics.mean(quality_scores) * 100
        throughput = 1000 / avg_time if avg_time > 0 else 0
        
        return BenchmarkResult(
            operation=f"knn_search_k{k}",
            dimension=len(vectors[0]),
            vector_count=len(vectors),
            avg_time_ms=avg_time,
            throughput=throughput,
            recall=avg_quality
        )
    
    def benchmark_delete(self, collection, ids, count: int = 100):
        """Benchmark delete operations"""
        times = []
        
        for i in range(count):
            delete_id = ids[i]
            
            start = time.time()
            collection.delete(ids=[delete_id])
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
        
        # Verify deletion
        deleted_count = 0
        for i in range(count):
            try:
                results = collection.get(ids=[ids[i]])
                if not results['ids']:
                    deleted_count += 1
            except:
                deleted_count += 1
        
        success_rate = (deleted_count / count) * 100
        avg_time = statistics.mean(times)
        throughput = 1000 / avg_time if avg_time > 0 else 0
        
        return BenchmarkResult(
            operation="delete",
            dimension=0,
            vector_count=count,
            avg_time_ms=avg_time,
            throughput=throughput,
            success_rate=success_rate
        )
    
    def benchmark_update(self, collection, vectors, ids, dim: int, count: int = 100):
        """Benchmark update operations"""
        np.random.seed(100)
        times = []
        
        for i in range(count):
            # Generate new vector
            new_vec = np.random.randn(dim).astype(np.float32)
            new_vec = new_vec / np.linalg.norm(new_vec)
            
            update_id = ids[i]
            
            start = time.time()
            collection.update(
                ids=[update_id],
                embeddings=[new_vec.tolist()],
                metadatas=[{"index": i, "updated": True}]
            )
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
        
        # Verify update
        updated_count = 0
        for i in range(count):
            try:
                results = collection.get(ids=[ids[i]], include=['metadatas'])
                if results['metadatas'] and len(results['metadatas']) > 0:
                    if results['metadatas'][0].get('updated'):
                        updated_count += 1
            except:
                pass
        
        success_rate = (updated_count / count) * 100
        avg_time = statistics.mean(times)
        throughput = 1000 / avg_time if avg_time > 0 else 0
        
        return BenchmarkResult(
            operation="update",
            dimension=dim,
            vector_count=count,
            avg_time_ms=avg_time,
            throughput=throughput,
            success_rate=success_rate
        )
    
    def run_benchmark_suite(self, dimension: int, vector_count: int):
        """Run complete benchmark suite for given dimension and vector count"""
        print(f"\n{'='*70}")
        print(f"  Dimension: {dimension}, Vector Count: {vector_count}")
        print(f"{'='*70}\n")
        
        # Generate test data
        print("  📊 Generating test vectors...")
        vectors, ids, metadatas = self.generate_vectors(vector_count, dimension)
        
        # Create collection
        collection_name = f"bench_{dimension}_{vector_count}"
        collection = self.create_collection(collection_name, dimension)
        
        # 1. Batch Insert
        print("  ⏱️  Batch insert...", end=" ", flush=True)
        result = self.benchmark_batch_insert(collection, vectors, ids, metadatas)
        self.results.append(result)
        print(f"✅ ({result.avg_time_ms:.3f}ms avg, {result.throughput:.0f} vec/sec)")
        
        # 2. Exact Search (k=1)
        print("  🔍 Exact search (k=1)...", end=" ", flush=True)
        result = self.benchmark_exact_search(collection, vectors, ids)
        self.results.append(result)
        print(f"✅ ({result.avg_time_ms:.3f}ms avg, recall: {result.recall:.2f}%)")
        
        # 3. KNN Search (k=10)
        print("  🔍 KNN search (k=10)...", end=" ", flush=True)
        result = self.benchmark_knn_search(collection, vectors, ids, k=10)
        self.results.append(result)
        print(f"✅ ({result.avg_time_ms:.3f}ms avg, quality: {result.recall:.2f}%)")
        
        # 4. KNN Search (k=100)
        if vector_count >= 1000:
            print("  🔍 KNN search (k=100)...", end=" ", flush=True)
            result = self.benchmark_knn_search(collection, vectors, ids, k=100)
            self.results.append(result)
            print(f"✅ ({result.avg_time_ms:.3f}ms avg, quality: {result.recall:.2f}%)")
        
        # 5. Update operations
        print("  📝 Update operations...", end=" ", flush=True)
        result = self.benchmark_update(collection, vectors, ids, dimension, count=100)
        self.results.append(result)
        print(f"✅ ({result.avg_time_ms:.3f}ms avg, success: {result.success_rate:.1f}%)")
        
        # 6. Delete operations
        print("  🗑️  Delete operations...", end=" ", flush=True)
        result = self.benchmark_delete(collection, ids, count=100)
        self.results.append(result)
        print(f"✅ ({result.avg_time_ms:.3f}ms avg, success: {result.success_rate:.1f}%)")
        
        # Cleanup
        self.client.delete_collection(collection_name)
    
    def print_summary(self):
        """Print summary of all results"""
        print("\n" + "="*70)
        print("  CHROMADB BENCHMARK RESULTS SUMMARY")
        print("="*70 + "\n")
        
        # Group by operation
        ops = {}
        for result in self.results:
            op = result.operation
            if op not in ops:
                ops[op] = []
            ops[op].append(result)
        
        for op_name, results in sorted(ops.items()):
            print(f"📊 {op_name}")
            print(f"{'─'*70}")
            
            times = [r.avg_time_ms for r in results]
            throughputs = [r.throughput for r in results if r.throughput > 0]
            recalls = [r.recall for r in results if r.recall > 0]
            success_rates = [r.success_rate for r in results if r.success_rate > 0]
            
            print(f"  Average Time: {statistics.mean(times):.3f}ms | "
                  f"Min: {min(times):.3f}ms | Max: {max(times):.3f}ms")
            
            if throughputs:
                print(f"  Throughput: {statistics.mean(throughputs):.0f} ops/sec (avg)")
            
            if recalls:
                print(f"  Recall/Quality: {statistics.mean(recalls):.2f}%")
            
            if success_rates and statistics.mean(success_rates) < 100:
                print(f"  Success Rate: {statistics.mean(success_rates):.2f}%")
            
            print()
    
    def save_results(self, filename: str = "chromadb_results.json"):
        """Save results to JSON file"""
        results_dict = [asdict(r) for r in self.results]
        with open(filename, 'w') as f:
            json.dump({
                "database": "ChromaDB",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": results_dict
            }, f, indent=2)
        print(f"💾 Results saved to: {filename}")


def main():
    print("🚀 ChromaDB Comprehensive Benchmark")
    print("="*70)
    
    # Test configurations
    dimensions = [512, 1024, 2048, 3072, 4096]
    vector_counts = [1000, 3000, 5000]
    
    benchmark = ChromaDBBenchmark()
    
    for dim in dimensions:
        for count in vector_counts:
            # Skip very large combinations
            if dim >= 4096 and count >= 5000:
                print(f"\n⏭️  Skipping: {dim}D x {count} vectors (too large)")
                continue
            
            try:
                benchmark.run_benchmark_suite(dim, count)
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
    
    # Print summary
    benchmark.print_summary()
    
    # Save results
    benchmark.save_results("chromadb_results.json")
    
    print("\n✅ Benchmark Complete!")


if __name__ == "__main__":
    main()
