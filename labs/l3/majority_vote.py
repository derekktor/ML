import pandas as pd
import sys

tr_path = sys.argv[1]
te_path = sys.argv[2]
tr_labels_path = sys.argv[3]
te_labels_path = sys.argv[4]
metrics_path = sys.argv[5]

tr = pd.read_csv(tr_path, sep = "\t")
te = pd.read_csv(te_path, sep = "\t")

X_tr = tr.iloc[:, :-1]
y_tr = tr.iloc[:, -1]

X_te = te.iloc[:, :-1]
y_te = te.iloc[:, -1]

def printModel(model: dict):
	print("Printing model:")
	print("-" * 70)
	for c in model:
		print(c.split("_")[0][:4], end="\t")
	print()
	for c in model:
		print(model[c], end="\t")
	print()
	print("-" * 70)

# initialize model
model = {}
for c in tr.columns[:-1]:
	model[c] = 0

printModel(model)

# fit the model
# for i in range(len(X_tr)):
# 	for c in X_tr.columns:
# 		feature = X_tr.loc[i, c]
# 		target = y_tr.loc[i]
# 		# print(i, c, X_tr.loc[i, c], y_tr.loc[i])
# 		if (feature == 0 and target == 0) or (feature == 1 and target == 1):
# 			model[c] += 1

# printModel(model)

commonLabel = y_tr.value_counts().idxmax()

# find the most common label
def getMostCommonLabel(model: dict) -> str:
	return max(model.items(), key = lambda k: (k[1], k[0]))[0]

def getErrorRate(model: dict, X, y, n=10) -> float:
	# commonLabel = getMostCommonLabel(model)
	print("Most common label:", commonLabel)

	error = 0
	for i in range(len(X)):
		# pred = X.loc[i, commonLabel]
		pred = commonLabel
		target = y.loc[i]
		if i < n:
			print(f"{i}: pred[{commonLabel}]={pred}, target={target}, match={pred == target}")
		if pred != target:
			error += 1

	return error / len(X)

with open(tr_labels_path, "w") as tr_labels:
	# commonLabel = getMostCommonLabel(model)
	for i in range(len(X_tr)):
		# pred = X_tr.loc[i, commonLabel]
		pred = commonLabel
		tr_labels.write(f"{pred}\n")

with open(te_labels_path, "w") as te_labels:
	# commonLabel = getMostCommonLabel(model)
	for i in range(len(X_te)):
		# pred = X_te.loc[i, commonLabel]
		pred = commonLabel
		te_labels.write(f"{pred}\n")

with open(metrics_path, "w") as metric:
    error_tr = getErrorRate(model, X_tr, y_tr)
    error_te = getErrorRate(model, X_te, y_te)

    metric.write(f"error(train): {error_tr}\n")
    metric.write(f"error(test): {error_te}\n")

    print("Error rate on training set:", error_tr)
    print("Error rate on test set:", error_te)


# python majority_vote.py heart_train.tsv heart_test.tsv heart_train_labels.txt heart_test_labels.txt heart_metrics.txt
# python majority_vote.py education_train.tsv education_test.tsv education_train_labels.txt education_test_labels.txt education_metrics.txt