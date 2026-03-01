# Lecture 3

- scikit

## ML Lifecycle

### Learning Problem

- target
  - what to predict
  - fatigue level from sleep, workout sessions
  - classification / regression
- objective
  - evaluate success/failure: accuracy, precision
- data
  - features
  - distribution of features
    - sleep hours in N~(6, 1) in hrs
    - workout duration in N~(1.5, 0.25) in hrs
    - RPE Uni~(1,10) (Rate of Perceived Exertion)
  - data types
  - missing values: fill, drop
  - labels / target

- what loss function to use

### Model design

- feature engineering
  - understand the features
  - select input features
    - too many features = overfitting = too curvy
    - there's automatic feature selection
  - encode input features
    - transform certain features: categorical, text, image -> numerical
    - categorical: ints, discrete
    - one-hot encoding
      - label of interest is one
      - others zero
    - skewed features
      - use log
    - feature standardization / creating new feature
      - Z = (x - mean) / stdde
    - encoding text
      - bwg of words encoding
        - count of certain words
        - drop _stop words_: not meaningful words (is, the, of...)
      - learned vector embeddings
        - strings to vectors
    - encoding image
      - tensor: multi dimensional matrix
      - images are in tensors: 3d matrices<rgb>
      - flatten the image: 3d -> 1d
      - transformation: color space, pixel normalization
      - hand craft features: edge detection
      - deep learning representations: neural networks
    - encoding steps
      - contructor: sklearn.preprocessing.OneHotEncoder()
      - fit: df[["color]]
      - transform
  - source: sklearn data transformation
  - create new features using the features
  - feature reduction
- model family
  - space of functions that map input(features) to output(label, target)
  - linear regression model family
    - g(x) = w_0 + sum(w_i, x)
      - w = weight vectors
      - x = input
  - logistic regression model fam
    - g(x) = s(w_0 + sum(w_i, x))
    - target = [0, 1]
  - inductive biases
    - assumptions made in the model designto enable generalization beyond training data
  - linear vs non-linear
    - non-linear: curved boundary, more expressive
  - Fitting
    - Underfitting: barely curved line
    - Overfitting: too curved
  - Regularization
    - adding constraints/penalties to improve generalization (avoid overfitting)
  - Parametric vs Non-parametric
    - parametric = fixed num of parameters
    - non-parametric = parameters grow with the training data
      - nearest neighbor model
- hypothesis space

### Optimization

- iterative optimization algos
  - w = argmin Err[h_w, D_tr] + l \* Reg[w]
- hyperparameter
  - param that's constant during optimization algorithm
  - for grid-search

### Predict

- inference = process of making predictions with a model
- label prediction
- probability prediction

### Evaluate

- accuracy
- Confusoin matrix:
  - True/False
  - Positive/Negative
  - TP = predicted yes when yes
  - FP = predicted yes when no
  - FN = predicted no when yes
  - TN = predicted no when no

## Taxonomy

- #Supervised learning
  - labeled
  - training data
  - Types
    - Regression / Quantitative label
      - y = 1.4
    - Classification / Categorical label
      - "Sunny", Yes/No
      - Binary / Multi-Class
- #Unsupervised learning
  - unlabeled
- #Bias
  - Error for underfitting
  - Too simple model
- #Variance
  - Error for overfitting
  - Too complex model
- #Regularization
  - technique for reducing overfitting
  - adds penalties to loss function: loss(weights) = error(weights) + penalty(weights)
  - penalty(weights) = lambda \* weights
- #Loss functions
  - how far the prediction is from the actual values
  - types
    - regression
      - mean squared error: error^2
      - mean absolute error: |error|
      - huber loss: uses threshold _delta_ value
        - error < _delta_ -> MSE
        - error > _delta_ -> MAE
    - classification
      - binary cross-entropy
        - predicts if yes/no (spam/not spam)
      - categorical cross-entropy
        - predicts from discrete targets (cat, dog, cow?)
      - hinge loss
        - uses SVM, margin-based classification

- #Entropy = unpredictability
  - high entropy = very unpredictable
- #SVM (Support vector machine)
  - algo for finding hyperplane
  - maximizes _margin_ to improve generalization
- #Hyperplane
  - Optimal boundary that separates labels
- ## #Generalization
- #Margin
  - distance between _hyperplane_ and the nearest support vector

## Train-Test split

> for evaluating generalization

- steps
  - shuffle the training data
  - split into 2 parts
    - 60 train
      - for developing the model
    - 20 validation split
      - split - evaluating generalizationg process
    - 20 test
      - evaluating the model
  - validation split
- Generalization
  - ability to perform well on new data

- Epoch
  - one cycle of learning on all training data (not validation section, test section)

```python

from sklearn.model_selection import train_test_split
# 1. train-test split
data = []
labels = [0, 1]
data_tr, data_te, labels_tr, labels_te = train_test_split(data, labels, test_size=.2, random_state=42)

# 2. train-val split
data_tr, data_val, labels_tr, labels_val = train_test_split(data_tr, labels_tr, test_size=.2, random_state=42)

```

- Majority Vote Classifier - ensemble learning method

## Ensemble learning

- multiple classifiers are trained together to produce better model
- types
  - bagging - Bootstrap AGGregating
    - Bootstrap - getting multiple samples from the population to get a better model
      - trains models independently + combines results
      - regression -> averaging
      - classification -> majority vote
  - boosting
    - trains models sequentially + each model works on the previous model's results
  - stacking

# Artificial Intelligence and Machine Learning

## Linear regression

hypothesis class
range of weights

model:
y_pred = f_W(X) = W dot X = w1*x1 + w2*x2 + ...
f_W(X) = W dot phi(X)

inputs
X = {x1, x2, ...}

weight vector
W = {w1, w2, ...}

feature extractor
phi(X) = [1, x, x^2, ...]

example predictor: f_w1(x) = 1 + 2x
w1 = [1, 2]
f_w1(x) = w1 dot phi(x)

residual: distance between prediction and target

loss function

- minimize the loss
- model lost LOSS much
- if LOSS = 0, no loss -> awesomely accurate model

## Linear classifier

decision boundary: - line that classifies the results - end result will be either on the right of it or the left of it

f(x) = sign(weights \* features)

loss function: - how good is the model - loss(input, output, weights) - loss(x, y, weights) = 1[f(x) != y]

training data: - has input and output: x and y - x -> y

zero-one loss: - loss_01(x, y, w) = 1[f(x) != y] - 1 if f(x) != y - 0 if f(x) == y

train_loss: - average of the loss function values - for x,y in training_data: - total_loss += loss(x, y, weights) - train_loss = total_loss / len(training_data)

training the model: - TRAINING_DATA(INPUT, TARGET_LABEL) -> MDOEL -> WEIGHTS

testing the model: - INPUT -> MODEL -> PREDICTED_LABEL

score: - weights \* phi(x) - how confident we are

margin: - (weights _ phi(x)) _ y - how correct we are

find the best model: optimization - find model that has the min tran_loss(w) - gradient descent

gradient descent: - gradient of train_loss(w) = sum of gradient of loss_01(x,y,w)

loss functions: - zero one - step function - 1[f(x) != y] - hinge - kinked function - max(1 - margin, 0)

Build model:

1. which predictors are possible?
   - hypothesis class
   - linear regression, logistic regression, SVM, ensemble, neural networks
2. how good is the predictor?
   - loss function
   - mean square: (y_target - y_pred) \*\* 2
3. how do find the best predictor?
   - optimization
   - optimize loss function
   - find weights that has the least loss function value
