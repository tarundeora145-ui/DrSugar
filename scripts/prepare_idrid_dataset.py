import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def prepare_idrid():
    base_path = Path('data/idrid/B. Disease Grading/2. Groundtruths')
    
    train_csv_path = base_path / 'a. IDRiD_Disease Grading_Training Labels.csv'
    test_csv_path = base_path / 'b. IDRiD_Disease Grading_Testing Labels.csv'
    
    if not train_csv_path.exists() or not test_csv_path.exists():
        logging.error("IDRiD label files not found. Check dataset path.")
        return
        
    train_df = pd.read_csv(train_csv_path)
    # The columns are likely 'Image name' and 'Retinopathy grade', let's normalize them
    # For IDRiD it's usually "Image name" and "Retinopathy grade"
    # Let's inspect the columns by checking the first row or just reading standard names
    
    logging.info(f"Loaded {len(train_df)} official training records.")
    
    # 80/20 train/val split using stratification on Retinopathy grade
    # Use random_state=42 for reproducibility
    try:
        stratify_col = train_df['Retinopathy grade']
    except KeyError:
        # Fallback if column names differ
        stratify_col = train_df.iloc[:, 1]
        
    train_split, val_split = train_test_split(
        train_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=stratify_col
    )
    
    # Process official test set (just copying it over for consistency)
    test_df = pd.read_csv(test_csv_path)
    
    # Save splits
    train_split.to_csv(base_path / 'split_train.csv', index=False)
    val_split.to_csv(base_path / 'split_val.csv', index=False)
    test_df.to_csv(base_path / 'split_test.csv', index=False)
    
    logging.info(f"Saved split_train.csv with {len(train_split)} images")
    logging.info(f"Saved split_val.csv with {len(val_split)} images")
    logging.info(f"Saved split_test.csv with {len(test_df)} images (held out)")
    
if __name__ == "__main__":
    prepare_idrid()
