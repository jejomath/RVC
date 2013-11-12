import RVC, scipy, random, math
import sklearn.metrics as metrics
import sklearn.linear_model as lm


# Load the OQ file and convert to data and labels
data = scipy.loadtxt("oq.csv")
labels = data[:,-1]
data = data[:,:-1]

# Change labels from 15 and 17 to 0 and 1. This is not
# necessary for the RVC algorithm, but makes the final
# AUC calculation simpler.

for i in range(len(labels)):
    if labels[i] == 15.0:
        labels[i] = 0
    else:
        labels[i] = 1

# Split the data into train and eval sets

# Pick out which indices correspond to which classes
A = [i for i in range(labels.shape[0]) if labels[i] == 1]
B = [i for i in range(labels.shape[0]) if labels[i] == 0]

# Randomize the sets of indices
random.shuffle(A)
random.shuffle(B)

# Calculate how many points to put in the training set
tA = int(len(A) * 0.8)
tB = int(len(B) * 0.8)

# Compile the lists of indices for the two sets
ti = A[:tA] + B[:tB]
ei = A[tA:] + B[tB:]

# Create the traing and test sets
tdata = data[ti]
tlabels = labels[ti]

# Create the labels lists for each
edata = data[ei]
elabels = labels[ei]

# Create the RVC object.
rvc = RVC.randomVoronoiCells(centroid_sharing=3, verbose = 1)

# Run RVC
for j in [4,8,12,16]:
    cw = []
    for i in range(10):
        cw+= [[i*.1, (10-i)*.1]]*j
    for i in range(5):
        print "Run", i+1
        rvc.fit(tdata, tlabels, centroid_weights = cw, iteration_number = 5)

# Calculate and report the AUC on training set
pr = rvc.predict(tdata)

fpr, tpr, thresholds = metrics.roc_curve(tlabels, pr, pos_label=1)
auc = metrics.auc(fpr,tpr)

print "Train AUC:", auc


# Calculate and report theAUC on eval set
pr = rvc.predict(edata)

fpr, tpr, thresholds = metrics.roc_curve(elabels, pr, pos_label=1)
auc = metrics.auc(fpr,tpr)

print "Eval AUC:", auc

# Compare to Logistic regression

print "Running logistic regression..."

LR = lm.LogisticRegression()

LR.fit(tdata, tlabels)

pred = LR.predict_proba(edata)
pred = [p[1] for p in pred]

fpr, tpr, thresholds = metrics.roc_curve(elabels, pred, pos_label=1)
auc = metrics.auc(fpr,tpr)

print "Logistic regression AUC:", auc
