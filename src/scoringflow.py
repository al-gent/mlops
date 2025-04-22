from metaflow import FlowSpec, step
import mlflow

class InferenceFlow(FlowSpec):

    @step
    def start(self):
        # point at your MLflow server
        mlflow.set_tracking_uri('https://adam-197919234028.us-west2.run.app/')
        
        # load the latest version of your registered model
        self.model = mlflow.sklearn.load_model('models:/metaflow-wine-model/latest')
        
        self.next(self.predict)

    @step
    def predict(self):
        import pandas as pd
        unseen = pd.read_csv('data/transformed_df.csv').drop(columns=['y']).iloc[:50, :]
        
        # make predictions
        self.predictions = self.model.predict(unseen)
        self.next(self.end)

    @step
    def end(self):
        print("Predictions on unseen data:")
        for pred in self.predictions:
            print(pred)

if __name__ == '__main__':
    InferenceFlow()
