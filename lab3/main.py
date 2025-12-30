import os
import argparse
import kagglehub
from pathlib import Path

from sklearn.model_selection import train_test_split

import torch

from dotenv import load_dotenv

load_dotenv()

PATH_DATA = Path(os.getenv('PATH_DATA'))
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {DEVICE} for inference')


def load_dataset(dst_path: str):
    # Download latest version
    path = kagglehub.dataset_download(handle='bhavikjikadara/dog-and-cat-classification-dataset', path=dst_path)

    print("Path to dataset files:", path)
    X_train, test_df = train_test_split(all_df, test_size=0.2, random_state=42, stratify=all_df['Labels'])


    return X_train, X_val, X_test, y_train, y_val, y_test


def get_model_resnet50():
    resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
    resnet50.eval().to(DEVICE)

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--outdir', default='out')
    args = parser.parse_args()

    dst_path = PATH_DATA / 'input'
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(str(dst_path))
    
    


if __name__ == '__main__':
    main()
