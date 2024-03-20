
# Compressing Random Forest Models with Over 90% Size Reduction 

This repository aims to provide a unique way to compress/store Random Forest Models with over 90% compression rates without sacrificing the models' original accuracy.

The apporach is pretty straight forward, as Random Forest models are binary trees in nature  and can be stored in a dictionary data structure after extracting the tree's content(features, nodes, thresholds, values). We can then in turn reduce the decimal precision of the thresholds and leaf node values of the tree to trim the non required trailing decimal digits while maintaining the same amount of accuracy. The resulting dictionary can then be saved in json format and this can be further zipped with best in class algorithms.

For obtaining the predictions from json a.k.a Random Forest Model we can utilize a recursive tree traversal logic /ᐠ ̥  ̮  ̥ ᐟ\ฅ

The repository works with both Classifiation and Regression models and is dynamically adapted to handle multi _input/output_ models as well.

**Note:** The process of loading converted json models might be slow with large sizes due to json's Parsing Overhead and inefficiency in python. To overcome this one can use these jsons with C++/Rust with the same tree traversal logic and obtain the predictions (ツ)

## Usage
1) Cloning the repository and installing the libraries 
```bash
git clone https://github.com/K-prog/Compressing-Random-Forest-Models.git
pip install -r requirements.txt
```
2) Creating a sample Random Forest using [createSampleModel.ipynb](https://github.com/K-prog/Compressing-Random-Forest-Models/blob/main/convertModel.py)
_If you already have a Random Forest .joblib model, skip directly to step 3._

![App Screenshot](https://cdn.discordapp.com/attachments/829041911028776960/1220000792690429972/image.png?ex=660d5947&is=65fae447&hm=81b054631040d551730ea50785786bd2ce9ed3397522647c5f6f2a289a4da44f&)

3) Converting the model via [convertModel.py](https://github.com/K-prog/Compressing-Random-Forest-Models/blob/main/convertModel.py)
```bash
python convertModel.py -path "conversions/test_small.joblib" -p 4
```
| Parameters | Type | Description |
|-----------------|-----------------|-----------------|
| -path    | String    | Path to joblib model    |
| -p    | int(>=0)    | Decimal precision allowed    |

For single output regression model with precision set to 2.
![App Screenshot](https://cdn.discordapp.com/attachments/829041911028776960/1220024334605619230/image.png?ex=660d6f34&is=65fafa34&hm=382886a42bdd47d3c08700bce4e603db74a87bbd4d476eec6d088803356ff377&)

For multiple output regression model with precision set to 1.
![App Screenshot](https://cdn.discordapp.com/attachments/829041911028776960/1220024755470602341/image.png?ex=660d6f98&is=65fafa98&hm=d82066aaa875953ec8c81ac2814049ce139d078ee8c1f1651af6c84444229967&)


Note: With dense models where size is greater than 200 MB, **_3>=p>=5_** achieves great results. Less complex models can even work with lower precisions, rest is up to your experimentation ;-;