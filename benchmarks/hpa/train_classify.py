#!/usr/bin/env python3
"""
Quick test script to verify your classification setup works
"""
import os
import torch
import pandas as pd
import numpy as np

def test_data_loading(features_path):
    """Test if the data loads correctly"""
    print("Testing data loading...")
    
    try:
        features = torch.load(features_path, map_location="cpu")
        df = pd.DataFrame(features[0])
        feature_data = features[1]
        print(f"✓ Data loaded successfully")
        print(f"  - DataFrame shape: {df.shape}")
        print(f"  - Feature data shape: {feature_data.shape}")
        print(f"  - DataFrame columns: {list(df.columns)}")
        
        # Check for required columns
        required_cols = ['antibody', 'locations', 'atlas_name']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠ Missing required columns: {missing_cols}")
        else:
            print("✓ All required columns present")
            
        return df, feature_data
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None, None

def test_file_structure():
    """Test if required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        "annotations/train_antibodies.txt",
        "annotations/valid_antibodies.txt", 
        "annotations/test_antibodies.txt"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")

def test_imports():
    """Test if all required packages are available"""
    print("\nTesting imports...")
    
    required_packages = [
        'torch', 'pandas', 'numpy', 'sklearn', 'matplotlib', 
        'seaborn', 'umap', 'colorcet', 'yaml', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} not available")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠ Install missing packages: pip install {' '.join(missing_packages)}")

def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    if torch.cuda.is_available():
        print(f"✓ CUDA available - {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA version: {torch.version.cuda}")
    else:
        print("⚠ CUDA not available - will use CPU")

def test_data_splits(df):
    """Test if antibody splits work"""
    print("\nTesting data splits...")
    
    try:
        train_antibodies = pd.read_csv("annotations/train_antibodies.txt", header=None)[0].to_list()
        val_antibodies = pd.read_csv("annotations/valid_antibodies.txt", header=None)[0].to_list()
        test_antibodies = pd.read_csv("annotations/test_antibodies.txt", header=None)[0].to_list()
        
        train_idxs = df[df["antibody"].isin(train_antibodies)].index.to_list()
        val_idxs = df[df["antibody"].isin(val_antibodies)].index.to_list()
        test_idxs = df[df["antibody"].isin(test_antibodies)].index.to_list()
        
        print(f"✓ Train samples: {len(train_idxs)}")
        print(f"✓ Val samples: {len(val_idxs)}")
        print(f"✓ Test samples: {len(test_idxs)}")
        
        # Check for overlap
        train_set = set(train_antibodies)
        val_set = set(val_antibodies)
        test_set = set(test_antibodies)
        
        if train_set & val_set:
            print("⚠ Overlap between train and val antibodies")
        if train_set & test_set:
            print("⚠ Overlap between train and test antibodies")
        if val_set & test_set:
            print("⚠ Overlap between val and test antibodies")
        
        if not (train_set & val_set) and not (train_set & test_set) and not (val_set & test_set):
            print("✓ No overlap between splits")
            
    except Exception as e:
        print(f"✗ Error testing splits: {e}")

def test_categories(df):
    """Test category processing"""
    print("\nTesting categories...")
    
    try:
        # Test location categories
        locations = df["locations"].dropna().str.split(",").tolist()
        all_locations = set()
        for loc_list in locations:
            all_locations.update(loc_list)
        
        print(f"✓ Found {len(all_locations)} unique locations")
        print(f"  - Sample locations: {list(all_locations)[:5]}")
        
        # Test atlas names
        atlas_names = df["atlas_name"].unique()
        print(f"✓ Found {len(atlas_names)} atlas names: {atlas_names}")
        
    except Exception as e:
        print(f"✗ Error testing categories: {e}")

def main():
    """Run all tests"""
    print("🧪 Testing your classification setup...\n")
    
    # Test imports first
    test_imports()
    
    # Test file structure
    test_file_structure()
    
    # Test CUDA
    test_cuda()
    
    # Test data loading
    features_path = "/scr/data/HPA_features/all_features.pth"  # Update this path
    df, feature_data = test_data_loading(features_path)
    
    if df is not None:
        test_data_splits(df)
        test_categories(df)
    
    print("\n🎉 Test complete! Check above for any issues.")

if __name__ == "__main__":
    main()