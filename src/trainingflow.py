from metaflow import FlowSpec, step
import pandas as pd

class ClassifierTrainFlow(FlowSpec):

    @step
    def start(self):
        from sklearn.model_selection import train_test_split

        df = pd.read_csv('./data/transformed_df.csv')
        y = df['y'].copy().values

        X = df.drop(columns='y').copy()

        self.train_data, self.test_data, self.train_labels, self.test_labels  = train_test_split(X, y, 
                                                            test_size=0.2, 
                                                            random_state=42,
                                                            stratify=y)
        print("Data loaded successfully")
        self.next(self.train_knn, self.train_svm)

    @step
    def train_adaboost(self):
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        base_estimator = DecisionTreeClassifier(
            max_depth=2,
            class_weight='balanced',
            random_state=42
        )

        self.model = AdaBoostClassifier(
            estimator=base_estimator,
            algorithm='SAMME',
            random_state=42,
        )
        self.model.fit(self.train_data, self.train_labels)
        self.next(self.choose_model)

    @step
    def train_lr(self):
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(
            penalty='l1',
            solver='saga',
            C=0.01,
            class_weight='balanced',
            random_state=42,
        )
        self.model.fit(self.train_data, self.train_labels)
        self.next(self.choose_model)

    @step
    def choose_model(self, inputs):
        import mlflow
        mlflow.set_tracking_uri('https://adam-197919234028.us-west2.run.app/')
        mlflow.set_experiment('metaflow-experiment')

        def score(inp):
            return inp.model, inp.model.score(inp.test_data, inp.test_labels)

        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]
        with mlflow.start_run():
            mlflow.sklearn.log_model(self.model, artifact_path = 'metaflow_train', registered_model_name="metaflow-wine-model")
            mlflow.end_run()
        self.next(self.end)

    @step
    def end(self):
        print('Scores:')
        print('\n'.join('%s %f' % res for res in self.results))
        print('Model:', self.model)


if __name__=='__main__':
    ClassifierTrainFlow()