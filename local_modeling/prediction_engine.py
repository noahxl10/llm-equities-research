
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from kerastuner import RandomSearch
from dataclasses import dataclass
from tensorflow import keras
import tensorflow as tf
import pandas as pd
from db import db

import dev_setup
dev_setup.set_environment_variables()


# Possibly try
# LOG regression
# SVM
# non-negative-matrix-factorization
# https://medium.com/codex/what-is-non-negative-matrix-factorization-nmf-32663fb4d65#:~:text=Non%2Dnegative%20Matrix%20Factorization%20or,data%20into%20lower%2Ddimensional%20spaces.
# back propogation

@dataclass
class ML:
    dbase = db.DB()
    data: pd.DataFrame = dbase.query_to_df('select * from dw.filtered_binary_feature_responses')
    X = data.drop(columns=['company_id', 'is_overperformer'])
    y = data['is_overperformer']
    random_state: int = 42 # defines random state for sampling reproducibility
    test_size: float = 0.2 # defines the test fraction for the train/test split
    X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y,
                            test_size=test_size,
                            random_state=random_state
                        )

    def random_forest_classifier(self):

        # Initializing the RF classifier
        rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100)
        # training
        rf_classifier.fit(self.X_train, self.y_train)
        return rf_classifier


    def random_forest(self):
        rf_classifier = self.random_forest_classifier()
        # Predicting on the test set
        y_pred = rf_classifier.predict(self.X_test)

        # Evaluating the model
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        accuracy, precision, recall, f1


    def get_feature_performances(self):

        # Find which features are most impactful on the dataset
        rf_classifier = self.random_forest_classifier()
        feature_importances = rf_classifier.feature_importances_

        features_df = pd.DataFrame({
            'Feature': self.X.columns,
            'Importance': feature_importances
        })

        # Sorting the features based on importance
        sorted_features = features_df.sort_values(by='Importance', ascending=False).head(20)

        return sorted_features


    def ff_nn(self):
        """
        Defines a mehod for a feed-froward neural network
        """
        layer_num_neurons = [310, 85, 60, 10, 5, 1]
        layer_activations = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid']
        # Possibly use TanH instead of sigmoid, then normalize the tanH func
        optimizer = 'adam'
        loss = 'binary_crossentropy'
        epochs = 20
        batch_size = 48


        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(self.X_train.shape[1],)),  # Input layer
            keras.layers.Dense(layer_num_neurons[0], activation=layer_activations[0]),     # hidden layer
            keras.layers.Dense(layer_num_neurons[1], activation=layer_activations[1]),   # hidden layer        
            keras.layers.Dense(layer_num_neurons[2], activation=layer_activations[2]),   # hidden layer        
            keras.layers.Dense(layer_num_neurons[3], activation=layer_activations[3]),   # hidden layer        
            keras.layers.Dense(layer_num_neurons[4], activation=layer_activations[4]),   # hidden layer        
            # keras.layers.Dense(layer_num_neurons[5], activation=layer_activations[5]),   # hidden layer        
            keras.layers.Dense(layer_num_neurons[-1], activation=layer_activations[-1])   # layer for binary classification
        ])


        model.compile(optimizer=optimizer, 
              loss=loss,
              metrics=['accuracy'])

        # Display the model architecture
        # model.summary()


        history = model.fit(
                self.X_train, 
                self.y_train, 
                epochs=epochs, 
                batch_size=batch_size, 
                validation_data=(self.X_test, self.y_test)
            )
        
        test_loss, test_accuracy = model.evaluate(self.X_test, self.y_test)
        return [test_loss, test_accuracy]


    def build_ff_nn(self, hp):
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(self.X_train.shape[1],)))
        
        # Tuning the number of units in the first Dense layer
        hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
        model.add(keras.layers.Dense(units=hp_units, activation='relu'))
        
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        
        # Tuning the learning rate
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        
        return model


    def nn_tuner(self):
        tuner = RandomSearch(
            self.build_ff_nn,
            objective='val_accuracy',
            max_trials=5,
            executions_per_trial=2,
            directory='my_dir',
            project_name='helloworld')

        tuner.search(self.X_train, self.y_train, epochs=10, validation_data=(self.X_test, self.y_test))
        return tuner

    def get_best_model(self, tuner):
        best_model = tuner.get_best_models(num_models=1)[0]
        return best_model

    def get_best_hyperparameters(self, tuner):
        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
        return best_hyperparameters

    def nn_predict(self, model, new_data: pd.DataFrame, binary_predict=True):
        
        predictions = model.predict(new_data)
        
        if binary_predict:
            binary_predictions = (predictions > 0.5).astype(int)

