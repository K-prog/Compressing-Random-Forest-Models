import argparse
import os
from pathlib import Path
from joblib import load
from utils import extract_tree_structure
import json
import gzip


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("-model_path", "--model", type=str, help="Path to Random Forest model", required=True)
    args.add_argument("-p", "--precision", type=int, help="Decimal Precision allowed", required=True)

    arg = vars(args.parse_args())
    model_path = Path(arg.get('model'))
    precision = arg.get('precision')
    
    size_mb = round((os.path.getsize(model_path)/1000000),2)
    print("Current size of Random Forest Model:",round(size_mb),"MB")

    model = load(model_path)

    feature_names = [i for i in range(model.n_features_in_)]

    model_dict = [extract_tree_structure(estimator, feature_names, precision=precision) for estimator in model.estimators_]
    print("Extracted Tree from model\u2705")

    file_paths = "conversions"
    json_path = os.path.join(file_paths,model_path.name.split(".")[0]+'.json')
    gzip_path = os.path.join(file_paths,model_path.name.split(".")[0]+'.json.gz')

    os.makedirs(file_paths, exist_ok=True)

    print("Saving model json...")
    with open(json_path, 'w') as f:
        json.dump(model_dict, f)

    print("Saving complete at path",'\033[1m\033[3m'+json_path+'\033[0m',"\u2705")
    with open(json_path, 'r') as f:
        json_data = f.read()

    print("Compressing json with gzip...")
    with gzip.open(gzip_path, 'wt', encoding='UTF-8') as zipfile:
        zipfile.write(json_data)
    print("Compression complete at path",'\033[1m\033[3m'+gzip_path+'\033[0m',"\u2705")

    gzip_size_mb = round((os.path.getsize(gzip_path)/1000000),2)    
    compression_rate = round(((size_mb - gzip_size_mb) / size_mb) * 100,2)


    print("Model size reduced from", size_mb,"MB to",gzip_size_mb,"MB",)
    print(str(compression_rate)+"%","Compression Rate")
