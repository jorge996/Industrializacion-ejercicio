import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from flask import Flask, request, jsonify

# Cargar el modelo del pickle

with open('C:/Users/USUARIO/Desktop/Bootcamp/Carpeta_alumno_PT/DS_PT_02_2024/04-Industrializacion/Industrializacion_ejercicio/model_and_test_train.pkl', 'rb') as file:
    df, xgb_cls, X_train, X_test, y_train, y_test = pickle.load(file)
#with open('model_and_test_train.pkl', 'rb') as file:
#    df, xgb_cls, X_train, X_test, y_train, y_test = pickle.load(file)

app = Flask(__name__)
app.config['DEBUG'] = False

@app.route('/', methods=['GET'])
def home():
	return "<h1>Estudio de enfermedades renales crónicas</h1><p>Esta API nos aporta los datos de unos pacientes sobre los que se ha estudiado enfermedades renales crónicas.</p>"


# TODOS LOS DATOS
@app.route('/api/v1/resources/data/all', methods=['GET']) 
def api_all():
    df_json = df.to_json(orient='records')
    return df_json


# TEST
@app.route('/api/v1/resources/data/test', methods=['GET'])
def api_test():
    y_test_join = pd.DataFrame(y_test)
    y_test_reset = y_test_join.reset_index()
    y_test_reset

    X_test_join = pd.DataFrame(X_test)
    X_test_reset = X_test_join.reset_index()
    X_test_reset

    test = X_test_reset.merge(y_test_reset, on='index')
    test = test.drop(columns=['index'])
    test_json = test.to_json(orient='records')
    return test_json


# TRAIN
@app.route('/api/v1/resources/data/train', methods=['GET'])
def api_train():
    y_train_join = pd.DataFrame(y_train)
    y_train_reset = y_train_join.reset_index()
    y_train_reset

    X_train_join = pd.DataFrame(X_train)
    X_train_reset = X_train_join.reset_index()
    X_train_reset

    train = X_train_reset.merge(y_train_reset, on='index')
    train = train.drop(columns=['index'])
    train_json = train.to_json(orient='records')
    return train_json


# OBSERVACION N DEL DATASET
@app.route('/api/v1/resources/data/all/<int:index>', methods=['GET'])
def api_observacion(index):
    df_index = df.reset_index()
    results = df_index[df_index['index'] == index]
    
    if not results.empty: 
        results_json = results.to_json(orient='records')
        return results_json
    
    else:
        return jsonify('No se ha encontrado ningún paciente con ese index')
         

# QUERY CON LA QUE OBTENEMOS UNA ORQUILLA DE EDADES
@app.route('/api/v1/resources/data/query', methods=['GET']) 
def api_query():

    from_ = int(request.args["from"])
    to_ =  int(request.args["to"])
    results_query = df[(df["Age"]>=from_) & (df["Age"]<=to_)]
    results_query_json = results_query.to_json(orient='records')
    return results_query_json


# QUERY CON LA QUE OBTENEMOS LAS PREDICCIONES
@app.route('/api/v1/resources/data/predictions', methods=['GET']) 
def api_predictions_query():

    type_ = str(request.args["type"])
    
    if type_ == 'test':
            predictions_test = pd.DataFrame(xgb_cls.predict(X_test), columns=['prediction'])
            predictions_test['patient'] = df.loc[X_test.index].index
            predictions_test = predictions_test.reindex(['patient', 'prediction'], axis=1)

            predictions_test_json = predictions_test.to_json(orient='records')

            return predictions_test_json
    
    elif type_ == 'train':
            predictions_train = pd.DataFrame(xgb_cls.predict(X_train), columns=['prediction'])
            predictions_train['patient'] = df.loc[X_train.index].index
            predictions_train = predictions_train.reindex(['patient', 'prediction'], axis=1)

            predictions_train_json = predictions_train.to_json(orient='records')

            return predictions_train_json

app.run(port=5000)

