#!/usr/bin/env python3
"""
Debug script to check SageMaker paths and data structure
"""
import os
import sys
from pathlib import Path

def check_path_structure():
    print("🔍 Debugging SageMaker paths and data structure...")
    print("=" * 60)
    
    # Check environment variables
    print("📋 Environment Variables:")
    sagemaker_vars = ['SM_CHANNEL_TRAINING', 'SM_MODEL_DIR', 'SM_OUTPUT_DATA_DIR']
    for var in sagemaker_vars:
        value = os.environ.get(var, 'NOT_SET')
        print(f"  {var}: {value}")
    
    print()
    
    # Check common paths
    common_paths = [
        '/opt/ml/input/data/training',
        '/opt/ml/input/data',
        '/opt/ml/model',
        '/opt/ml/output',
        '/app',
        'dataset/tagged',
        '.'
    ]
    
    print("📁 Path Existence Check:")
    for path in common_paths:
        exists = Path(path).exists()
        print(f"  {path}: {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
    
    print()
    
    # List contents of key directories
    key_dirs = ['/opt/ml/input/data/training', '/opt/ml/input/data', '.']
    
    for dir_path in key_dirs:
        if Path(dir_path).exists():
            print(f"📂 Contents of {dir_path}:")
            try:
                items = list(Path(dir_path).iterdir())
                if items:
                    for item in sorted(items):
                        item_type = "📁" if item.is_dir() else "📄"
                        print(f"  {item_type} {item.name}")
                else:
                    print("  (empty)")
            except Exception as e:
                print(f"  ❌ Error listing contents: {e}")
            print()
    
    # Check if we can find the expected data structure
    training_path = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
    print(f"🎯 Checking expected data structure at: {training_path}")
    
    if Path(training_path).exists():
        try:
            for item in Path(training_path).iterdir():
                if item.is_dir():
                    print(f"  📁 {item.name}/")
                    # Check for shard directories
                    shard_dirs = [d for d in item.iterdir() if d.is_dir() and d.name.startswith('shard_')]
                    if shard_dirs:
                        print(f"    🗂️  Found {len(shard_dirs)} shard directories")
                        for shard in sorted(shard_dirs[:3]):  # Show first 3
                            print(f"      - {shard.name}")
                        if len(shard_dirs) > 3:
                            print(f"      - ... and {len(shard_dirs) - 3} more")
                    else:
                        print("    ⚠️  No shard directories found")
        except Exception as e:
            print(f"  ❌ Error checking structure: {e}")
    else:
        print("  ❌ Training data path does not exist")

if __name__ == "__main__":
    check_path_structure() 