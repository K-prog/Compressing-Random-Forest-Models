
# Compressing Random Forest Models with Over 90% Size Reduction 

This repository aims to provide a unique way to compress/store Random Forest Models with over 90% compression rates without sacrificing the models' original accuracy.

The apporach is pretty straight forward, as Random Forest models are binary trees in nature  and can be stored in a dictionary data structure after extracting the tree's content(features, nodes, thresholds, values). We can then in turn reduce the decimal precision of the thresholds and leaf node values of the tree to trim the non required trailing decimal digits while maintaining the same amount of accuracy. The resulting dictionary can then be saved in JSON format and this can be further zipped with best in class algorithms.

For obtaining the predictions from JSON a.k.a the Random Forest Model we can utilize a simple recursive tree traversal logic  ฅ/ᐠ. ̫ .ᐟ\ฅ 

The repository works with both Classifiation and Regression models and is dynamically adapted to handle multi _input/output_ models as well.

**Note:** The process of loading converted JSON models might be slow with large sizes due to JSON's Parsing Overhead and inefficiency in python. To overcome this one can use these JSONs with C++/Rust with the same tree traversal logic and obtain the predictions (ツ)

## Usage
1) Cloning the repository and installing the libraries 
```bash
git clone https://github.com/K-prog/Compressing-Random-Forest-Models.git
pip install -r requirements.txt
```
2) Creating a sample Random Forest using [createSampleModel.ipynb](https://github.com/K-prog/Compressing-Random-Forest-Models/blob/main/createSampleModel.ipynb)
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

With the below sample model configuration:
```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Generate a synthetic regression dataset
X, y = make_regression(n_samples=5000, n_features=50, n_informative=10, noise=0.1, random_state=42, n_targets=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

# Create the Random Forest regression model
random_forest_model = RandomForestRegressor(n_estimators=500, random_state=42)

# Train the model
random_forest_model.fit(X_train, y_train)

```
The trends for size and R2 score b/w joblib and JSON for this sample model on the bases of decimal precision :

_The compression rate was over 90% for all the precison values as original model size is 200MB_
![App Screenshot](https://cdn.discordapp.com/attachments/829041911028776960/1220379100422737960/image.png?ex=660eb99b&is=65fc449b&hm=d2d5159b563115d5dc1afcb3f4595c7bb7f63c6fb1c88ff6d2f33035cf99a81d&)

**A quick and fun read:**[**ヽ༼ ʘ̚ل͜ʘ̚༽ﾉ**](https://karansingh3267.medium.com/compressing-random-forest-models-with-over-90-reduction-in-size-24c3e7d1f52b)
