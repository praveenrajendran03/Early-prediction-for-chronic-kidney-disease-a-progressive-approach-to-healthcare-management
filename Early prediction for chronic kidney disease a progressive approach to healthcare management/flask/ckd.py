data=pd.read_csv("/kaggle/input/ckdisease/kidney_disease.csv")
data.head()
data.columns
data.isnull().sum()
data.classification.unique()
data.classification=data.classification.replace("ckd\t","ckd")
data.classification.unique()
data=data.drop("id",axis=1)
data.shape
data['classification']=data['classification'].replace(['ckd',"notckd"],[1,0])
data.head()
data.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
              'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
              'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
              'aanemia', 'class']
data.describe()
data.info()

data['packed_cell_volume'] = pd.to_numeric(data['packed_cell_volume'], errors='coerce')
data['white_blood_cell_count'] = pd.to_numeric(data['white_blood_cell_count'], errors='coerce')
data['red_blood_cell_count'] = pd.to_numeric(data['red_blood_cell_count'], errors='coerce')
data.isnull().sum()
cat_col=[col for col in data.columns if data[col].dtype=="object"]
num_col=[col for col in data.columns if data[col].dtype!="object"]
for col in cat_col:
    print(f"{col} has {data[col].unique()} values\n")
    data.diabetes_mellitus = data["diabetes_mellitus"].replace(["\tno", '\tyes'], ["no", "yes"])

    data.coronary_artery_disease = data['coronary_artery_disease'].replace('\tno', 'no')
    cols = ['diabetes_mellitus', 'coronary_artery_disease']
    for col in cols:
        print(f"{col} has {data[col].unique()} values\n")
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(20, 15))
        plotnumber = 1
        for column in num_col:
            if plotnumber <= 14:
                ax = plt.subplot(5, 3, plotnumber)
                sns.distplot(data[column])
                plt.xlabel(column)

            plotnumber += 1
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(20, 15))
        plotnumber = 1

        for column in cat_col:
            if plotnumber <= 11:
                ax = plt.subplot(3, 4, plotnumber)
                sns.countplot(data[column], palette='rocket')
                plt.xlabel(column)

            plotnumber += 1

        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(15, 8))
        sns.heatmap(data.corr())
        plt.show()
for col in num_col:
    data[col]=data[col].fillna(data[col].median())
    for col in cat_col:
        print(f" {col} most frequent value is {data[col].mode()}\n")
        data['red_blood_cells'].fillna('normal', inplace=True)
        data['aanemia'].fillna('no', inplace=True)
        data['peda_edema'].fillna('no', inplace=True)
        data["coronary_artery_disease"].fillna('no', inplace=True)
        data['diabetes_mellitus'].fillna('no', inplace=True)
        data['pus_cell_clumps'].fillna('notpresent', inplace=True)
        data["coronary_artery_disease"].fillna('no', inplace=True)
        data["hypertension"].fillna("no", inplace=True)
        data["appetite"].fillna("good", inplace=True)
        data["pus_cell"].fillna("normal", inplace=True)
        data["bacteria"].fillna("notpresent", inplace=True)
        data.isnull().sum()
        for col in cat_col:
            print(f"{col} has {data[col].nunique()} categories\n")
            data.head()
            X = data.drop("class", axis=1)
            y = data["class"]
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
            knn = KNeighborsClassifier()
            # accuracy score, confusion matrix and classification report of knn
            knn_acc = accuracy_score(y_test, knn.predict(X_test))
            print(f"Training Accuracy of KNN is {accuracy_score(y_train, knn.predict(X_train))}")
            print(f"Test Accuracy of KNN is {knn_acc} \n")
            print(f"Confusion Matrix :- \n{confusion_matrix(y_test, knn.predict(X_test))}\n")
            print(f"Classification Report :- \n {classification_report(y_test, knn.predict(X_test))}")

            rd_clf = RandomForestClassifier(criterion='entropy', max_depth=11, max_features='auto', min_samples_leaf=2,
                                            min_samples_split=3, n_estimators=130)
            rd_clf.fit(X_train, y_train)

            # accuracy score, confusion matrix and classification report of random forest

            rd_clf_acc = accuracy_score(y_test, rd_clf.predict(X_test))

            print(
                f"Training Accuracy of Random Forest Classifier is {accuracy_score(y_train, rd_clf.predict(X_train))}")
            print(f"Test Accuracy of Random Forest Classifier is {rd_clf_acc} \n")

            print(f"Confusion Matrix :- \n{confusion_matrix(y_test, rd_clf.predict(X_test))}\n")
            print(f"Classification Report :- \n {classification_report(y_test, rd_clf.predict(X_test))}")