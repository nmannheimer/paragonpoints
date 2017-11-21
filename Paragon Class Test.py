import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

class ParagonPointsClassifier():

    def __init__(self):
        import numpy as np
        from sklearn.metrics import classification_report, accuracy_score


    def classes(self):
        print(np.unique(self.y))

    def fit(self, X, y):
        self.X = X
        self.y = y

        # Define empty paragon points and largest euclidian distance and seperate classes in samples
        paragon0 = np.array([])
        paragon1 = np.array([])
        class0 = X[y==0,:]
        class1 = X[y==1,:]
        class0_dist = 0
        class1_dist = 0

        #iterate through the samples in each class and find the point that has the largest cumulative euclidian distance from the opposing class
        for i in range(len(class0)):
                dist = 0
                for j in range(len(class1)):
                        dist += np.linalg.norm(class0[i]-class1[j])
                        if j == len(class1)-1 and dist > class0_dist:
                                class0_dist = dist
                                paragon0 = class0[i]
        for i in range(len(class1)):
                dist = 0
                for j in range(len(class0)):
                        dist += np.linalg.norm(class1[i]-class0[j])
                        if j == len(class0)-1 and dist > class1_dist:
                                class1_dist = dist
                                paragon1 = class1[i]
        self.paragon0 = paragon0
        self.paragon1 = paragon1
        return np.array([paragon0,paragon1])

    def predict(self,x):
        paragon0 = self.paragon0
        paragon1 = self.paragon1

        classifications = np.array([])
        for datum in x:
            if np.linalg.norm(paragon0 - datum) < np.linalg.norm(paragon1 - datum):
                classifications = np.append(classifications, [0])
                # print('Point in Class 0')
            elif np.linalg.norm(paragon1 - datum) < np.linalg.norm(paragon0 - datum):
                classifications = np.append(classifications, [1])
                # print('Point in Class 1')
            else:
                print('RandClass', str(np.random.randint(0, 2)))
        return classifications

X = np.loadtxt('titanic-train.csv',delimiter=',')

# Pull out the labels from the dataset as y, then remove them from the main dataset as X and drop the ID column
y = X[:,-1]
X = X[:, 1:-1]

# Create a training and test set with labels by randomly splitting the samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)


# ParagonPoints is sensitive to outliers and noisy data.
# Using scaling and PCA we can reduce the noise and spread of the data.
scaler = StandardScaler()
pca = PCA(n_components=4)

# Fit the Standard Scaler and PCA on the training data and then transform the training and test data.
X_train = scaler.fit_transform(X_train)
X_train = pca.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_test = pca.transform(X_test)
print(pca.explained_variance_ratio_)

# Call the Paragon Points Classifier and fit it to the training dataset
model = ParagonPointsClassifier()
model.fit(X_train,y_train)

# Make label predictions for the test data and save the predictions.
predictions = model.predict(X_test)

# Evaluate model and compare performance
print('')
print('ParagonPoints Classifier Performance:')
print('')
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions)*100,'% Accuracy')

print('')
print('Support Vector Classifier (Radial Kernel) Performance:')
print('')

# Compare the results of the ParagonPoints model to a cross validation optimized Support Vector Machine

kf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
svc_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

svc = SVC(class_weight='balanced')
model = GridSearchCV(svc, svc_parameters,cv=kf, scoring='accuracy', verbose=True)
model.fit(X_train,y_train)
svc_predictions = model.predict(X_test)

print(classification_report(y_test, svc_predictions))
print(accuracy_score(y_test, svc_predictions)*100,'% Accuracy')

print('')
print('Null (always 0) Classifier Performance:')
print('')
print(classification_report(y_test, np.zeros((len(y_test),1))))
print(accuracy_score(y_test, np.zeros((len(y_test),1)))*100,'% Accuracy')