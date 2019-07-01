# load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# data check
print('Shape:')
print(dataset.shape)
print('Head (peek):')
print(dataset.head(20))
print('Statistical summary:')
print(dataset.describe())
print('Class distribution:')
print(dataset.groupby('class').size())

dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)  # box and whiskers plot
plt.show()
dataset.hist()  # histogram
plt.show()
scatter_matrix(dataset)  # scatter plot matrix
plt.show()

# initialize cross validation
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.20
#  We reset the random number seed before each run to ensure that the evaluation of each algorithm is performed using
#  exactly the same data splits. It ensures the results are directly comparable.
seed = 7  # The specific random seed does not matter
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)
# the metric of ‘accuracy‘ evaluates the models. This is a ratio of the number of correctly predicted instances in
# divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate).
scoring = 'accuracy'

# spot check algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))  # Logistic Regression
models.append(('LDA', LinearDiscriminantAnalysis()))  # Linear Discriminant Analysis
models.append(('KNN', KNeighborsClassifier()))  # K-Nearest Neighbors
models.append(('CART', DecisionTreeClassifier()))  # Classification and Regression Trees
models.append(('NB', GaussianNB()))  # Gaussian Naive Bayes
models.append(('SVM', SVC(gamma='auto')))  # Support Vector Machines
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# compare algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
