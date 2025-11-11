"""
Comparison Analysis Script
Analyzes and compares GoVecDB and ChromaDB benchmark results

Usage:
  python compare_results.py govecdb_results.json chromadb_benchmark_results.json
"""

import json
import pandas as pd
import sys
from typing import Dict, List
import numpy as np

def load_results(filepath: str) -> pd.DataFrame:
    """Load benchmark results from JSON"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def compare_databases(govecdb_df: pd.DataFrame, chromadb_df: pd.DataFrame):
    """Compare performance between databases"""
    
    print("🔬 GoVecDB vs ChromaDB Performance Comparison")
    print("=" * 80)
    print()
    
    # Get common operations
    common_ops = set(govecdb_df['operation'].unique()) & set(chromadb_df['operation'].unique())
    
    comparison_data = []
    
    for op in sorted(common_ops):
        gov_data = govecdb_df[govecdb_df['operation'] == op]
        chroma_data = chromadb_df[chromadb_df['operation'] == op]
        
        gov_avg = gov_data['avg_time'].mean() * 1000  # Convert to ms
        chroma_avg = chroma_data['avg_time'].mean() * 1000
        
        speedup = chroma_avg / gov_avg if gov_avg > 0 else 0
        winner = "GoVecDB" if speedup > 1 else "ChromaDB"
        
        comparison_data.append({
            'operation': op,
            'govecdb_ms': gov_avg,
            'chromadb_ms': chroma_avg,
            'speedup': speedup,
            'winner': winner
        })
        
        emoji = "🏆" if winner == "GoVecDB" else "🥈"
        print(f"{emoji} {op}")
        print(f"   GoVecDB:  {gov_avg:.3f}ms")
        print(f"   ChromaDB: {chroma_avg:.3f}ms")
        print(f"   Speedup:  {speedup:.2f}x {winner}")
        print()
    
    # Overall summary
    print("\n📊 Overall Summary")
    print("=" * 80)
    
    df = pd.DataFrame(comparison_data)
    
    govecdb_wins = len(df[df['winner'] == 'GoVecDB'])
    chromadb_wins = len(df[df['winner'] == 'ChromaDB'])
    
    print(f"GoVecDB Wins:  {govecdb_wins}/{len(df)} operations 🏆")
    print(f"ChromaDB Wins: {chromadb_wins}/{len(df)} operations")
    print()
    
    avg_speedup = df['speedup'].mean()
    if avg_speedup > 1:
        print(f"Average: GoVecDB is {avg_speedup:.2f}x faster ⚡")
    else:
        print(f"Average: ChromaDB is {1/avg_speedup:.2f}x faster")
    
    # Throughput comparison
    print("\n📈 Throughput Comparison")
    print("=" * 80)
    
    for op in ['batch_insert', 'search_k10']:
        if op in govecdb_df['operation'].values and op in chromadb_df['operation'].values:
            gov_throughput = govecdb_df[govecdb_df['operation'] == op]['throughput'].mean()
            chroma_throughput = chromadb_df[chromadb_df['operation'] == op]['throughput'].mean()
            
            print(f"{op}:")
            print(f"  GoVecDB:  {gov_throughput:,.0f} ops/sec")
            print(f"  ChromaDB: {chroma_throughput:,.0f} ops/sec")
            print()
    
    # Quality metrics
    print("\n🎯 Search Quality Comparison")
    print("=" * 80)
    
    for metric in ['recall', 'search_quality']:
        gov_metric = govecdb_df[govecdb_df[metric] > 0][metric].mean()
        chroma_metric = chromadb_df[chromadb_df[metric] > 0][metric].mean()
        
        if not np.isnan(gov_metric) and not np.isnan(chroma_metric):
            print(f"{metric.replace('_', ' ').title()}:")
            print(f"  GoVecDB:  {gov_metric*100:.2f}%")
            print(f"  ChromaDB: {chroma_metric*100:.2f}%")
            print()
    
    # Dimension analysis
    print("\n📊 Performance by Dimension")
    print("=" * 80)
    
    for dim in sorted(set(govecdb_df['dimension'].unique()) & set(chromadb_df['dimension'].unique())):
        if dim == 0:
            continue
            
        gov_search = govecdb_df[(govecdb_df['dimension'] == dim) & 
                                (govecdb_df['operation'] == 'search_k10')]['avg_time'].mean() * 1000
        chroma_search = chromadb_df[(chromadb_df['dimension'] == dim) & 
                                    (chromadb_df['operation'] == 'search_k10')]['avg_time'].mean() * 1000
        
        if not np.isnan(gov_search) and not np.isnan(chroma_search):
            speedup = chroma_search / gov_search
            winner = "⚡" if speedup > 1 else "  "
            print(f"Dim {dim:4d}: GoVecDB {gov_search:6.2f}ms | ChromaDB {chroma_search:6.2f}ms | {speedup:.2f}x {winner}")
    
    print("\n✅ Comparison Complete!")

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_results.py govecdb_results.json chromadb_results.json")
        sys.exit(1)
    
    govecdb_file = sys.argv[1]
    chromadb_file = sys.argv[2]
    
    print(f"Loading GoVecDB results from: {govecdb_file}")
    print(f"Loading ChromaDB results from: {chromadb_file}")
    print()
    
    try:
        govecdb_df = load_results(govecdb_file)
        chromadb_df = load_results(chromadb_file)
        
        compare_databases(govecdb_df, chromadb_df)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nPlease ensure both benchmark result files exist.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
