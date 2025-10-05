"""
ChromaDB Comprehensive Performance Benchmark (Updated for Latest API)

This script benchmarks ChromaDB performance across various dimensions and operations.
Compatible with ChromaDB 0.4.x+ with updated API.

Install: pip install chromadb numpy pandas matplotlib
"""

import chromadb
from chromadb.config import Settings
import numpy as np
import time
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
from datetime import datetime
import gc
import os

@dataclass
class BenchmarkResult:
    dimension: int
    num_vectors: int
    operation: str
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    throughput: float
    recall: float = 0.0
    search_quality: float = 0.0

class ChromaDBBenchmark:
    def __init__(self):
        """Initialize ChromaDB client with latest API"""
        # Use PersistentClient for better control (replaces deprecated Settings)
        self.client = chromadb.PersistentClient(
            path="/tmp/chromadb_benchmark",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

    def generate_vectors(self, count: int, dim: int, seed: int = 42) -> Tuple[np.ndarray, List[str], List[Dict]]:
        """Generate normalized random vectors with metadata"""
        np.random.seed(seed)

        # Generate vectors
        vectors = np.random.randn(count, dim).astype(np.float32)

        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-8)

        # Generate IDs
        ids = [f"vec_{i}" for i in range(count)]

        # Generate metadata
        categories = ["tech", "science", "arts", "sports", "news"]
        metadatas = [
            {
                "index": i,
                "category": categories[i % len(categories)],
                "score": float(np.random.random()),
                "group": i % 10,
            }
            for i in range(count)
        ]

        return vectors, ids, metadatas

    def should_skip(self, dim: int, num_vectors: int) -> bool:
        """Skip very large combinations"""
        if dim >= 1536 and num_vectors > 25000:
            return True
        if dim >= 1024 and num_vectors > 50000:
            return True
        return False

    def run_benchmark(self, dim: int, num_vectors: int, search_k: int = 10, num_searches: int = 100) -> List[BenchmarkResult]:
        """Run complete benchmark suite for given configuration"""
        results = []

        try:
            # Create collection with updated API
            collection_name = f"bench_{dim}_{num_vectors}"

            # Delete if exists
            try:
                self.client.delete_collection(collection_name)
            except:
                pass

            # Create collection with metadata
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Cosine distance
            )

            # Generate test vectors
            print(f"    🎲 Generating vectors... ", end="", flush=True)
            vectors, ids, metadatas = self.generate_vectors(num_vectors, dim)
            print("✅")

            # Benchmark: Batch Insert
            print(f"    ⏱️  Batch insert... ", end="", flush=True)
            result = self.benchmark_batch_insert(collection, vectors, ids, metadatas)
            results.append(result)
            print(f"✅ ({result.avg_time*1000:.2f}ms, {result.throughput:.0f} vec/sec)")

            # Benchmark: Single Insert (subset)
            if num_vectors <= 10000:
                print(f"    ⏱️  Single insert... ", end="", flush=True)
                result = self.benchmark_single_insert(collection, dim, 100)
                results.append(result)
                print(f"✅ ({result.avg_time*1000:.2f}ms avg)")

            # Benchmark: Exact Search
            print(f"    🔍 Exact search (k=1)... ", end="", flush=True)
            result = self.benchmark_exact_search(collection, vectors, ids, 1, num_searches)
            results.append(result)
            print(f"✅ ({result.avg_time*1000:.2f}ms avg, recall: {result.recall*100:.2f}%)")

            # Benchmark: KNN Search
            print(f"    🔍 KNN search (k=10)... ", end="", flush=True)
            result = self.benchmark_search(collection, vectors, search_k, num_searches)
            results.append(result)
            print(f"✅ ({result.avg_time*1000:.2f}ms avg, quality: {result.search_quality*100:.2f}%)")

            # Benchmark: Large K Search
            if num_vectors >= 1000:
                print(f"    🔍 Large K search (k=100)... ", end="", flush=True)
                result = self.benchmark_search(collection, vectors, 100, 50)
                results.append(result)
                print(f"✅ ({result.avg_time*1000:.2f}ms avg)")

            # Benchmark: Filtered Search
            print(f"    🔍 Filtered search... ", end="", flush=True)
            result = self.benchmark_filtered_search(collection, vectors, metadatas, search_k, 50)
            results.append(result)
            print(f"✅ ({result.avg_time*1000:.2f}ms avg)")

            # Benchmark: Update
            if num_vectors <= 10000:
                print(f"    📝 Update operations... ", end="", flush=True)
                result = self.benchmark_update(collection, dim, 100)
                results.append(result)
                print(f"✅ ({result.avg_time*1000:.2f}ms avg)")

            # Benchmark: Upsert (new feature)
            if num_vectors <= 10000:
                print(f"    📝 Upsert operations... ", end="", flush=True)
                result = self.benchmark_upsert(collection, dim, 100)
                results.append(result)
                print(f"✅ ({result.avg_time*1000:.2f}ms avg)")

            # Benchmark: Delete
            if num_vectors <= 10000:
                print(f"    🗑️  Delete operations... ", end="", flush=True)
                result = self.benchmark_delete(collection, 100)
                results.append(result)
                print(f"✅ ({result.avg_time*1000:.2f}ms avg)")

            # Benchmark: Get by ID
            print(f"    📌 Get by ID... ", end="", flush=True)
            result = self.benchmark_get_by_id(collection, ids, 100)
            results.append(result)
            print(f"✅ ({result.avg_time*1000:.2f}ms avg)")

            # Benchmark: Count
            print(f"    🔢 Count operations... ", end="", flush=True)
            result = self.benchmark_count(collection, 50)
            results.append(result)
            print(f"✅ ({result.avg_time*1000:.2f}ms avg)")

            # Cleanup
            self.client.delete_collection(collection_name)
            gc.collect()

        except Exception as e:
            print(f"    ❌ Error: {e}")
            import traceback
            traceback.print_exc()

        return results

    def benchmark_batch_insert(self, collection, vectors, ids, metadatas) -> BenchmarkResult:
        """Benchmark batch insertion"""
        start = time.perf_counter()

        collection.add(
            embeddings=vectors.tolist(),
            ids=ids,
            metadatas=metadatas
        )

        elapsed = time.perf_counter() - start

        return BenchmarkResult(
            dimension=vectors.shape[1],
            num_vectors=len(vectors),
            operation="batch_insert",
            total_time=elapsed,
            avg_time=elapsed / len(vectors),
            min_time=elapsed / len(vectors),
            max_time=elapsed / len(vectors),
            throughput=len(vectors) / elapsed
        )

    def benchmark_single_insert(self, collection, dim: int, count: int) -> BenchmarkResult:
        """Benchmark single vector insertion"""
        times = []

        for i in range(count):
            vector = np.random.randn(dim).astype(np.float32)
            vector = vector / (np.linalg.norm(vector) + 1e-8)

            start = time.perf_counter()
            collection.add(
                embeddings=[vector.tolist()],
                ids=[f"single_{dim}_{i}"],
                metadatas=[{"type": "single"}]
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return self._calculate_stats("single_insert", dim, count, times)

    def benchmark_search(self, collection, vectors, k: int, num_searches: int) -> BenchmarkResult:
        """Benchmark KNN search"""
        times = []
        qualities = []

        indices = np.random.randint(0, len(vectors), num_searches)

        for idx in indices:
            query = vectors[idx]
            query_id = f"vec_{idx}"

            start = time.perf_counter()
            results = collection.query(
                query_embeddings=[query.tolist()],
                n_results=k
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            # Calculate quality
            if results and results['ids'] and len(results['ids'][0]) > 0:
                result_ids = results['ids'][0]
                quality = self._calculate_search_quality(result_ids, query_id)
                qualities.append(quality)

        result = self._calculate_stats(f"search_k{k}", vectors.shape[1], num_searches, times)
        result.search_quality = np.mean(qualities) if qualities else 0.0
        return result

    def benchmark_exact_search(self, collection, vectors, ids, k: int, num_searches: int) -> BenchmarkResult:
        """Benchmark exact match search"""
        times = []
        recalls = []

        indices = np.random.randint(0, len(vectors), num_searches)

        for idx in indices:
            query = vectors[idx]
            expected_id = ids[idx]

            start = time.perf_counter()
            results = collection.query(
                query_embeddings=[query.tolist()],
                n_results=k
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            # Check recall
            if results and results['ids'] and len(results['ids'][0]) > 0:
                if results['ids'][0][0] == expected_id:
                    recalls.append(1.0)
                else:
                    recalls.append(0.0)

        result = self._calculate_stats("exact_search", vectors.shape[1], num_searches, times)
        result.recall = np.mean(recalls) if recalls else 0.0
        return result

    def benchmark_filtered_search(self, collection, vectors, metadatas, k: int, num_searches: int) -> BenchmarkResult:
        """Benchmark filtered search with where clause"""
        times = []
        categories = ["tech", "science", "arts", "sports", "news"]

        for i in range(num_searches):
            idx = np.random.randint(0, len(vectors))
            query = vectors[idx]
            category = np.random.choice(categories)

            start = time.perf_counter()
            try:
                results = collection.query(
                    query_embeddings=[query.tolist()],
                    n_results=k,
                    where={"category": category}
                )
            except Exception as e:
                # If filtering not supported, just do regular search
                results = collection.query(
                    query_embeddings=[query.tolist()],
                    n_results=k
                )
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return self._calculate_stats("filtered_search", vectors.shape[1], num_searches, times)

    def benchmark_update(self, collection, dim: int, count: int) -> BenchmarkResult:
        """Benchmark update operations"""
        times = []

        for i in range(count):
            vector = np.random.randn(dim).astype(np.float32)
            vector = vector / (np.linalg.norm(vector) + 1e-8)

            start = time.perf_counter()
            try:
                collection.update(
                    embeddings=[vector.tolist()],
                    ids=[f"vec_{i}"],
                    metadatas=[{"updated": True, "timestamp": time.time()}]
                )
            except Exception as e:
                pass  # ID might not exist
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return self._calculate_stats("update", dim, count, times)

    def benchmark_upsert(self, collection, dim: int, count: int) -> BenchmarkResult:
        """Benchmark upsert operations (insert or update)"""
        times = []

        for i in range(count):
            vector = np.random.randn(dim).astype(np.float32)
            vector = vector / (np.linalg.norm(vector) + 1e-8)

            start = time.perf_counter()
            try:
                collection.upsert(
                    embeddings=[vector.tolist()],
                    ids=[f"upsert_{i}"],
                    metadatas=[{"upserted": True}]
                )
            except Exception as e:
                pass
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return self._calculate_stats("upsert", dim, count, times)

    def benchmark_delete(self, collection, count: int) -> BenchmarkResult:
        """Benchmark delete operations"""
        times = []

        for i in range(count):
            start = time.perf_counter()
            try:
                collection.delete(ids=[f"vec_{i}"])
            except:
                pass  # ID might not exist
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return self._calculate_stats("delete", 0, count, times)

    def benchmark_get_by_id(self, collection, ids: List[str], count: int) -> BenchmarkResult:
        """Benchmark get by ID operations"""
        times = []

        indices = np.random.randint(0, len(ids), count)

        for idx in indices:
            target_id = ids[idx]

            start = time.perf_counter()
            try:
                results = collection.get(ids=[target_id])
            except:
                pass
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return self._calculate_stats("get_by_id", 0, count, times)

    def benchmark_count(self, collection, iterations: int) -> BenchmarkResult:
        """Benchmark count operations"""
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            try:
                count = collection.count()
            except:
                pass
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return self._calculate_stats("count", 0, iterations, times)

    def _calculate_stats(self, operation: str, dim: int, count: int, times: List[float]) -> BenchmarkResult:
        """Calculate statistics from timing measurements"""
        if not times:
            return BenchmarkResult(
                dimension=dim,
                num_vectors=count,
                operation=operation,
                total_time=0,
                avg_time=0,
                min_time=0,
                max_time=0,
                throughput=0
            )

        times_array = np.array(times)
        total = np.sum(times_array)

        return BenchmarkResult(
            dimension=dim,
            num_vectors=count,
            operation=operation,
            total_time=total,
            avg_time=np.mean(times_array),
            min_time=np.min(times_array),
            max_time=np.max(times_array),
            throughput=len(times) / total if total > 0 else 0
        )

    def _calculate_search_quality(self, result_ids: List[str], query_id: str) -> float:
        """Calculate search quality score"""
        if not result_ids:
            return 0.0

        for i, rid in enumerate(result_ids):
            if rid == query_id:
                return 1.0 - (i / len(result_ids))

        return 0.0

def print_summary(results: List[BenchmarkResult]):
    """Print comprehensive summary of results"""
    print("\n\n🏆 FINAL PERFORMANCE SUMMARY")
    print("=" * 80)
    print()

    # Group by operation
    df = pd.DataFrame([asdict(r) for r in results])

    if len(df) == 0:
        print("No results to display")
        return

    for operation in df['operation'].unique():
        op_data = df[df['operation'] == operation]

        print(f"\n📊 {operation}")
        print("─" * 80)

        avg_ms = op_data['avg_time'].mean() * 1000
        min_ms = op_data['avg_time'].min() * 1000
        max_ms = op_data['avg_time'].max() * 1000

        print(f"  Average: {avg_ms:.3f}ms | Min: {min_ms:.3f}ms | Max: {max_ms:.3f}ms")

        if op_data['throughput'].mean() > 0:
            print(f"  Throughput: {op_data['throughput'].mean():.0f} ops/sec (avg)")

        if operation == 'exact_search' and op_data['recall'].mean() > 0:
            print(f"  Recall: {op_data['recall'].mean()*100:.2f}%")

        if 'search_k' in operation and op_data['search_quality'].mean() > 0:
            print(f"  Search Quality: {op_data['search_quality'].mean()*100:.2f}%")

    print("\n✅ Benchmark Complete!")
    print()

def main():
    """Main benchmark execution"""
    print("🚀 ChromaDB Comprehensive Performance Benchmark")
    print("=" * 60)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Test configuration
    dimensions = [2, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6120]
    vector_counts = [1000, 2000, 3000, 4000, 5000]
    search_k = 10
    num_searches = 100

    benchmark = ChromaDBBenchmark()
    all_results = []

    # Run benchmarks
    for dim in dimensions:
        print(f"\n📊 Testing Dimension: {dim}")
        print("─" * 60)

        for num_vectors in vector_counts:
            if benchmark.should_skip(dim, num_vectors):
                print(f"  ⏭️  Skipping {num_vectors} vectors (too large)")
                continue

            print(f"\n  📦 Vector Count: {num_vectors}")

            results = benchmark.run_benchmark(dim, num_vectors, search_k, num_searches)
            all_results.extend(results)

    # Print summary
    print_summary(all_results)

    # Save results to JSON
    results_dict = [asdict(r) for r in all_results]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    json_filename = f'chromadb_benchmark_results_{timestamp}.json'
    with open(json_filename, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'dimensions': dimensions,
                'vector_counts': vector_counts,
                'total_tests': len(all_results)
            },
            'results': results_dict
        }, f, indent=2)

    print(f"\n💾 Results saved to: {json_filename}")

    # Create comparison DataFrame
    df = pd.DataFrame(results_dict)
    csv_filename = f'chromadb_benchmark_results_{timestamp}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"💾 Results saved to: {csv_filename}")
    
    print(f"\n⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()