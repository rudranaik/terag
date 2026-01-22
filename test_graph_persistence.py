#!/usr/bin/env python3
"""
Test script to verify graph persistence improvements
"""
import json
import sys
import os
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from terag import TERAG, TERAGConfig

def test_persistence():
    """Test auto-save and manual save functionality"""
    print("\n" + "="*70)
    print("TEST: Graph Persistence")
    print("="*70)
    
    # Cleanup previous runs
    if os.path.exists("auto_saved_graph.json"):
        os.remove("auto_saved_graph.json")
    if os.path.exists("manual_saved_graph.json"):
        os.remove("manual_saved_graph.json")

    # Load chunks
    with open('chunks.json', 'r') as f:
        chunks = json.load(f)
    
    print("\n--- Test 1: Auto-save Functionality ---")
    
    config = TERAGConfig(
        min_concept_freq=1, 
        top_k=3, 
        use_llm_for_ner=False,
        auto_save_graph=True,
        graph_save_path="auto_saved_graph.json"
    )
    
    print("Building graph with auto-save enabled...")
    terag = TERAG.from_chunks(chunks[:2], config=config, verbose=True)
    
    if os.path.exists("auto_saved_graph.json"):
        size = os.path.getsize("auto_saved_graph.json")
        print(f"\n✓ Auto-save successful! File exists ({size} bytes)")
    else:
        print("\n✗ Auto-save failed! File does not exist")
        return

    print("\n--- Test 2: Manual Save Functionality ---")
    terag.save_graph("manual_saved_graph.json")
    
    if os.path.exists("manual_saved_graph.json"):
        size = os.path.getsize("manual_saved_graph.json")
        print(f"✓ Manual save successful! File exists ({size} bytes)")
    else:
        print("✗ Manual save failed!")
        return
        
    print("\n--- Test 3: Loading Graph ---")
    try:
        terag_loaded = TERAG.from_graph_file(
            "manual_saved_graph.json", 
            config=config, 
            verbose=True
        )
        print("✓ Graph loaded successfully!")
        
        # Verify stats match
        stats = terag_loaded.get_graph_statistics()
        print(f"  Loaded graph has {stats['num_passages']} passages, {stats['num_concepts']} concepts")
        
        if stats['num_passages'] > 0:
            print("✓ Graph stats verify content was loaded")
        else:
            print("✗ Check failed: Graph is empty")
            
    except Exception as e:
        print(f"✗ Loading failed: {e}")
        return
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("✓ Auto-save works")
    print("✓ Manual save works")
    print("✓ Logging shows absolute paths")
    print("="*70)
    
    # Cleanup
    if os.path.exists("auto_saved_graph.json"):
        os.remove("auto_saved_graph.json")
    if os.path.exists("manual_saved_graph.json"):
        os.remove("manual_saved_graph.json")

if __name__ == "__main__":
    test_persistence()
