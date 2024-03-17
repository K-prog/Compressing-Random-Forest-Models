import argparse
import os
from pathlib import Path






if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("-model_path", "--model", type=str, help="Path to Random Forest model", required=True)
    args.add_argument("-p", "--precision", type=int, help="Decimal Precision allowed", required=True)

    arg = vars(args.parse_args())
    model_path = Path(arg.get('model'))
    precision = arg.get('precision')

    size = os.path.getsize(model_path)
    print("Current size of Random Forest Model:",round(size/1000000,2),"MB")
