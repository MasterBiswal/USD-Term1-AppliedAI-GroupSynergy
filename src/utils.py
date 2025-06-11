import pandas as pd
import os

def load_raw_k9_data(
    data_path="data/raw/K9.data",
    tag_path="data/raw/K9.instance.tags"
):
    """
    Loads the raw K9 dataset and instance tags.
    Returns:
        data (pd.DataFrame): Raw feature data.
        tags (pd.Series): Corresponding tags/labels.
    """
    cwd = os.getcwd()
    full_data_path = os.path.join(cwd, data_path)
    full_tag_path = os.path.join(cwd, tag_path)

    # Print for debug
    print(f"[INFO] Current working directory: {cwd}")
    print(f"[INFO] Full path to data file: {full_data_path}")
    print(f"[INFO] Full path to tag file:  {full_tag_path}")

    # Check and explain clearly
    if not os.path.exists(full_data_path):
        print(f"[ERROR] Data file does not exist at: {full_data_path}")
    if not os.path.exists(full_tag_path):
        print(f"[ERROR] Tag file does not exist at: {full_tag_path}")

    if not os.path.exists(full_data_path) or not os.path.exists(full_tag_path):
        raise FileNotFoundError("One or both data files not found. See printed paths above.")

    data = pd.read_csv(full_data_path, header=None, low_memory=False)
    tags = pd.read_csv(full_tag_path, header=None).squeeze()

    return data, tags
