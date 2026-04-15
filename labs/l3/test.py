# run command: 
# python3 majority_vote.py heart_train.tsv heart_test.tsv heart_train_labels.txt heart_test_labels.txt heart_metrics.txt
# python3 majority_vote.py education_train.tsv education_test.tsv education_train_labels.txt education_test_labels.txt education_metrics.txt
import sys

PREDICTION = {}

with open(sys.argv[1], "r") as dtrain:
    d_train = dtrain.readlines()
with open(sys.argv[2], "r") as dtest:
    d_test = dtest.readlines()


def init_prediction(labels):
    for i in labels:
        PREDICTION[i] = 0

def predict_one(feature):
    keys = list(PREDICTION.keys())
    majority = max(PREDICTION.items(), key=lambda i: (i[1], i[0]))[0]
    if feature[keys.index(majority)] == '1':
        return '1'
    return '0'

def train_pred(feature):
    for i in range(len(feature)-1):
        if (feature[i] == '1' and feature[-1]=='1') or (feature[i] == '0' and feature[-1]=='0'):
            PREDICTION[labels[i]] += 1
    
def classifier(data, output_path, train=True):
    error = 0
    with open(output_path, "w") as out_file:
        for row in data:
            feature = row.split()
            pred = predict_one(feature)
            out_file.write(pred + '\n')
            if pred != feature[-1]:
                error += 1
            if train:
                train_pred(feature)
    return error / len(data)

def write_metric(out_path, train_error, test_error):
    with open(out_path, "w") as metric:
        metric.write(f"error(train): {train_error:.4f}\n")
        metric.write(f"error(test): {test_error:.4f}")


labels = d_train[0].split()
d_train = d_train[1:]

init_prediction(labels)

# predict train
train_error_rate = classifier(d_train, sys.argv[3])

# predict test
test_error_rate = classifier(d_test, sys.argv[4], train=False)

write_metric(sys.argv[5], train_error_rate, test_error_rate)