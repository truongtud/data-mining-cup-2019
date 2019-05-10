import pydotplus
from IPython.display import Image
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.externals.six import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.tree import export_graphviz
from .preprocessing import *

class ModelFactory:

    @staticmethod
    def createDecisionTree():
        return DecisionTreeClassifier()

    @staticmethod
    def createSVM():
        return svm.SVC(kernel='linear', C=0.5, degree=3)

    @staticmethod
    def createLogistic():
        return LogisticRegression()

    @staticmethod
    def createOneClassSVM():
        return svm.OneClassSVM(kernel='rbf', degree=3, gamma=0.01)


feature_cols = ['trustLevel_1', 'trustLevel_2', 'trustLevel_3', 'trustLevel_4', 'trustLevel_5', 'trustLevel_6',
                'totalScanTimeInSeconds', 'grandTotal', 'lineItemVoids', 'scansWithoutRegistration',
                'quantityModifications', 'scannedLineItemsPerSecond', 'valuePerSecond', 'lineItemVoidsPerPosition']


def decisionTreeClassifier(X_train, X_test, y_train, y_test):
    dtc = ModelFactory.createDecisionTree()
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    DT_cm = metrics.confusion_matrix(y_test, y_pred)
    # print("Confusion matrix using Decision Tree:", DT_cm.ravel())
    print("Confusion matrix using Decision Tree:", DT_cm)
    print("DT scores:", DT_cm[1][1] * 5 - DT_cm[0][1] * 25 - DT_cm[1][0] * 5)
    dot_data = StringIO()
    export_graphviz(dtc, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('fraud_retailer.png')
    Image(graph.create_png())


def svmClassifier(X_train, X_test, y_train, y_test):
    SVM = ModelFactory.createSVM()
    SVM.fit(X_train, y_train)
    y_SVM_pred = SVM.predict(X_test)
    SVM_cm = metrics.confusion_matrix(y_test, y_SVM_pred)
    print("Confusion matrix using SVM", SVM_cm)
    print("SVM scores:", SVM_cm[1][1] * 5 - SVM_cm[0][1] * 25 - SVM_cm[1][0] * 5)


def oneClassSVM(X_train, y_train, X_test, y_test):
    ocSVM = ModelFactory.createOneClassSVM()
    ocSVM.fit(X_train)
    y_ocSVM_pred = ocSVM.predict(X_test)
    print(y_ocSVM_pred)
    # ocSVM_cm=metrics.confusion_matrix(normal_y_test,y_ocSVM_pred)
    # print("Confusion matrix using One class SVM", ocSVM_cm)


def logisticClassifier(X_train, X_test, y_train, y_test):
    lc = ModelFactory.creatLogistic()
    lc.fit(X_train, y_train)
    y_lc_pred = lc.predict(X_test)
    lc_cm = metrics.confusion_matrix(y_test, y_lc_pred)
    print("Confusion matrix using Logistic", lc_cm)
    print("Logistic scores:", lc_cm[1][1] * 5 - lc_cm[0][1] * 25 - lc_cm[1][0] * 5)


def votingClassifier(X_train, X_test, y_train, y_test):
    votingClassifier = VotingClassifier(estimators=[
        ('dc', ModelFactory.createDecisionTree()), ('svm', ModelFactory.createSVM()),
        ('lg', ModelFactory.createLogistic())
    ], voting='hard')
    votingClassifier.fit(X_train, y_train)
    y_voted = votingClassifier.predict(X_test)
    voting_cm = metrics.confusion_matrix(y_test, y_voted)
    print("Confusion matrix using Voting", voting_cm)
    print("Voting scores:", voting_cm[1][1] * 5 - voting_cm[0][1] * 25 - voting_cm[1][0] * 5)



def load_labeled_data(scaler):
    df = read_data('../DMC_2019_task/train.csv')
    # fraud_df = fraud_instances(df)
    # normal_df=normal_instances(df)
    # groupByTrustLevel = fraud_instances_groupby(fraud_df, 'trustLevel')

    X, y = labeled_data(df)
    X = scale(X, scaler)
    # y=np.array(y)
    # pca = PCA(n_components=2)
    # plot_2d_space(pca.fit_transform(X), y, 'Imbalanced dataset (2 PCA components)')
    # t_sne=TSNE(n_components=2)
    # plot_2d_space(t_sne.fit_transform(X), y, 'Imbalanced dataset (2 TSNE components)')
    # X_sm, y_sm = oversampling(X, y)
    # X_sm=normalize(X_sm)
    # X_train, X_test, y_train, y_test = split_data(X, y)
    # normal_X,normal_y=training_data(normal_df)
    # normal_X_sm,normal_y_sm=oversampling(normal_X,normal_y)
    # normal_X=scale(normal_X,MinMaxScaler())

    # normal_X_train,normal_X_test,normal_y_train,normal_y_test=split_data(normal_X,normal_y)
    return X, y