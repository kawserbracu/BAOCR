#!/usr/bin/env python3
"""
Combine original, CLAHE, and high-boost datasets for enhanced training data.
This creates a unified dataset with all preprocessing variants.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Any
import argparse

def load_dataset(json_path: Path) -> Dict[str, Any]:
    """Load a recognition dataset JSON file."""
    if not json_path.exists():
        return {"items": []}
    with json_path.open('r', encoding='utf-8') as f:
        return json.load(f)

def combine_datasets(base_dir: Path, output_dir: Path, dataset_names: List[str]):
    """
    Combine multiple dataset variants into a single enhanced dataset.
    
    Args:
        base_dir: Base directory containing dataset variants
        output_dir: Output directory for combined dataset
        dataset_names: List of dataset variant names (e.g., ['merged', 'merged_all_clahe', 'merged_all_highboost'])
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize combined datasets
    combined_train = {"items": []}
    combined_val = {"items": []}
    combined_test = {"items": []}
    
    # Copy vocab from the first available dataset
    vocab_copied = False
    
    for dataset_name in dataset_names:
        dataset_dir = base_dir / dataset_name
        if not dataset_dir.exists():
            print(f"Warning: Dataset {dataset_name} not found, skipping...")
            continue
            
        print(f"Processing dataset: {dataset_name}")
        
        # Copy vocab if not already copied
        if not vocab_copied:
            vocab_src = dataset_dir / 'vocab.json'
            if vocab_src.exists():
                shutil.copy2(vocab_src, output_dir / 'vocab.json')
                vocab_copied = True
                print(f"  Copied vocab from {dataset_name}")
        
        # Load and combine train/val/test splits
        for split_name, combined_split in [
            ('recognition_train.json', combined_train),
            ('recognition_val.json', combined_val), 
            ('recognition_test.json', combined_test)
        ]:
            split_path = dataset_dir / split_name
            if split_path.exists():
                split_data = load_dataset(split_path)
                items = split_data.get('items', [])
                
                # Add dataset variant prefix to avoid path conflicts
                for item in items:
                    # Update crop_path to include dataset variant
                    original_path = Path(item['crop_path'])
                    if original_path.is_absolute():
                        # Convert to relative path from base_dir
                        try:
                            rel_path = original_path.relative_to(base_dir)
                            item['crop_path'] = str(base_dir / rel_path)
                        except ValueError:
                            # Path is not relative to base_dir, keep as is
                            pass
                    else:
                        # Relative path, prepend dataset name
                        item['crop_path'] = str(dataset_dir / original_path)
                    
                    # Add metadata about source dataset
                    item['source_dataset'] = dataset_name
                
                combined_split['items'].extend(items)
                print(f"  Added {len(items)} items from {split_name}")
    
    # Save combined datasets
    for split_name, combined_split in [
        ('recognition_train.json', combined_train),
        ('recognition_val.json', combined_val),
        ('recognition_test.json', combined_test)
    ]:
        output_path = output_dir / split_name
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(combined_split, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(combined_split['items'])} items to {split_name}")
    
    # Print summary
    print(f"\nCombined Dataset Summary:")
    print(f"  Train: {len(combined_train['items'])} items")
    print(f"  Val: {len(combined_val['items'])} items") 
    print(f"  Test: {len(combined_test['items'])} items")
    print(f"  Total: {len(combined_train['items']) + len(combined_val['items']) + len(combined_test['items'])} items")
    print(f"  Output directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Combine multiple dataset variants')
    parser.add_argument('--base_dir', type=str, default='./out', 
                       help='Base directory containing dataset variants')
    parser.add_argument('--output_name', type=str, default='merged_all_combined',
                       help='Output dataset name')
    parser.add_argument('--datasets', nargs='+', 
                       default=['merged', 'merged_all_clahe', 'merged_all_highboost'],
                       help='Dataset variant names to combine')
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_dir = base_dir / args.output_name
    
    print(f"Combining datasets: {args.datasets}")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    combine_datasets(base_dir, output_dir, args.datasets)
    print("\nDataset combination complete!")

if __name__ == '__main__':
    main()
