"""Explore FNSPID dataset structure"""
from datasets import load_dataset

print("Attempting to load FNSPID dataset without specifying data files...")
try:
    dataset = load_dataset("Zihan1004/FNSPID")
    print(f"Dataset loaded successfully!")
    print(f"Dataset structure: {dataset}")
    if hasattr(dataset, 'keys'):
        print(f"Available splits: {dataset.keys()}")
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying alternative approach...")
    try:
        # Try loading without data_files specification
        from huggingface_hub import list_repo_files
        files = list_repo_files("Zihan1004/FNSPID", repo_type="dataset")
        print(f"\nAvailable files in repository:")
        for f in files:
            print(f"  - {f}")
    except Exception as e2:
        print(f"Alternative approach also failed: {e2}")
