import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import logging

logging.basicConfig(level=logging.INFO)

def prepare_dataset(data_dir='data/aptos2019'):
    base_path = Path(data_dir)
    images_dir = base_path / 'train_images'
    csv_path = base_path / 'train.csv'
    
    if not images_dir.exists() or not csv_path.exists():
        logging.error(f"Dataset not found at {data_dir}. Expected train_images/ and train.csv")
        return False
        
    df = pd.read_csv(csv_path)
    
    # Verify images exist
    valid_rows = []
    missing_count = 0
    
    for idx, row in df.iterrows():
        img_path = images_dir / f"{row['id_code']}.png"
        if img_path.exists():
            valid_rows.append(row)
        else:
            missing_count += 1
            
    if missing_count > 0:
        logging.warning(f"Found {missing_count} missing images.")
        
    df_valid = pd.DataFrame(valid_rows)
    logging.info(f"Verified {len(df_valid)} valid images.")
    
    # Stratified split 80/10/10
    train_df, temp_df = train_test_split(df_valid, test_size=0.2, stratify=df_valid['diagnosis'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['diagnosis'], random_state=42)
    
    logging.info(f"Split distribution: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    print("\n--- ACTUAL CLASS DISTRIBUTION ---")
    print("Overall:")
    print(df_valid['diagnosis'].value_counts().sort_index().to_dict())
    print("\nTrain Split:")
    print(train_df['diagnosis'].value_counts().sort_index().to_dict())
    print("\nVal Split:")
    print(val_df['diagnosis'].value_counts().sort_index().to_dict())
    print("\nTest Split:")
    print(test_df['diagnosis'].value_counts().sort_index().to_dict())
    print("---------------------------------\n")
    
    # Save splits
    train_df.to_csv(base_path / 'split_train.csv', index=False)
    val_df.to_csv(base_path / 'split_val.csv', index=False)
    test_df.to_csv(base_path / 'split_test.csv', index=False)
    
    metadata = {
        "dataset": "APTOS 2019",
        "total_valid_images": len(df_valid),
        "missing_images": missing_count,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "classes": 5
    }
    
    with open(base_path / 'split_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
        
    logging.info("Dataset preparation complete.")
    return True

if __name__ == "__main__":
    prepare_dataset()
