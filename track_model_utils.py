import os
import pickle
import datetime
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    mean_squared_error,
    explained_variance_score,
    r2_score,
    roc_auc_score,
    average_precision_score
)
warnings.filterwarnings("ignore")
def save_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    user_name,
    tipo_model,
    comentario="",
    params_extras=None,
    metrics_extras=None,
    stats_extras=None,
    split_random_state=None,
    clf_umbral=0.5,
    target_enc_model=None,
    dicc_models_predict_nans=None,
    feature_validation_dict=None,
    save=0,
    dir_results_files="results_files",
    dir_models="models"
):
    """
    tipo_model: 1 = Clasificacion
    tipo_model: 2 = Regression
    """
    if params_extras is None:
        params_extras = dict()
    if metrics_extras is None:
        metrics_extras = dict()
    if stats_extras is None:
        stats_extras = dict()

    today = (
        str(datetime.datetime.now().day)
        + "-"
        + str(datetime.datetime.now().month)
        + "-"
        + str(datetime.datetime.now().year)
        + " "
        + str(datetime.datetime.now().hour)
        + ":"
        + str(datetime.datetime.now().minute)
    )
    today_format = datetime.datetime.strptime(today, "%d-%m-%Y %H:%M")
    def unique_id():
        dt = datetime.datetime.now()
        return (
            f"{dt.year:02d}"
            f"{dt.month:02d}"
            f"{dt.day:02d}"
            + f"{dt.hour:02d}"
            + f"{dt.minute:02d}"
            + f"{dt.second:02d}"
        )
    # Generamos el id del modelo
    model_id = unique_id()
    # Parametros del modelo
    dicc_params = model.get_params()
    dicc_params["Algoritmo"] = str(model).split("(")[0]
    dicc_params["Tipo_modelo"] = tipo_model
    # Parametros de las notas
    dicc_params.update(params_extras)
    # DF
    df_params = pd.DataFrame(dicc_params, index=[user_name + "-" + model_id])
    # Metricas del modelo
    metrics_dct = {}
    if tipo_model == 1:
        # Classification
        metrics_dct["Tipo_metrica"] = "Clasificacion"
        metrics_dct["Tipo_modelo"] = tipo_model
        metrics_dct["Algoritmo"] = str(model).split("(")[0]
        metrics_dct["umbral"] = clf_umbral
        # preds_proba_train = model.predict_proba(X_train)[:, 1]
        # preds_train = preds_proba_train >= clf_umbral
        preds_proba = model.predict_proba(X_test)[:, 1]
        preds = preds_proba >= clf_umbral
        # Training
        # acc_train = accuracy_score(y_train, preds_train)
        # f1_train = f1_score(y_train, preds_train)
        # recall_train = recall_score(y_train, preds_train)
        # precision_train = precision_score(y_train, preds_train)
        # auc_train = roc_auc_score(y_train, preds_proba_train)
        # gini_train = auc_train * 2 - 1
        # Testing
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        recall = recall_score(y_test, preds)
        precision = precision_score(y_test, preds)
        auc_score = roc_auc_score(y_test, preds_proba)
        gini = auc_score * 2 - 1
        prauc = average_precision_score(y_test, preds_proba)
        # gini_diff_testing_train = gini_train - gini
        metrics_dct["AUC"] = auc_score
        metrics_dct["Gini"] = gini
        metrics_dct["PRAUC"] = prauc
        metrics_dct["F1_score"] = f1
        metrics_dct["Accuracy"] = acc
        metrics_dct["Recall"] = recall
        metrics_dct["Precision"] = precision
    elif tipo_model == 2:
        # Regression
        metrics_dct["Tipo_metrica"] = "Regresion"
        metrics_dct["Tipo_modelo"] = tipo_model
        metrics_dct["Algoritmo"] = str(model).split("(")[0]
        metrics_dct["MSE"] = mean_squared_error(y_test, model.predict(X_test))
        metrics_dct["R2"] = r2_score(y_test, model.predict(X_test))
        metrics_dct["Explained_variance"] = explained_variance_score(
            y_test, model.predict(X_test)
        )
    # metrics_extras
    metrics_dct.update(metrics_extras)
    metrics_dct["Fecha"] = today_format
    metrics_dct["Comentario"] = comentario
    # print_metrics
    print()
    print("saved metrics ")
    print("MODEL ID: ", user_name + "-" + model_id)
    for k in metrics_dct.keys():
        print(f"{k}: {metrics_dct[k]}")
    print()
    # Df
    df_metrics = pd.DataFrame(metrics_dct, index=[user_name + "-" + model_id])
    # Stats de la data del modelo
    stats = {}
    stats["Algoritmo"] = str(model).split("()")[0]
    stats["Tipo_modelo"] = tipo_model
    stats["mean_xtrain"] = X_train.describe().iloc[1, :].mean()
    stats["mean_ytrain"] = y_train.mean()
    stats["mean_xtest"] = X_test.describe().iloc[1, :].mean()
    stats["mean_ytest"] = y_test.mean()
    stats["split_random_state"] = split_random_state
    # stats_extras
    stats.update(stats_extras)
    stats["Fecha"] = today_format
    stats["Comentario"] = comentario
    # DF
    df_stats = pd.DataFrame(stats, index=[user_name + "-" + model_id])
    # Features de la data de train
    train = pd.concat([X_train, y_train], axis=1)
    # ^ ES IMPORTANTE que el target este de ultimo!
    df_feats = pd.DataFrame(
        train.columns, columns=[user_name + "-" + model_id]
    ).T
    df_feats.columns = [i for i, c in enumerate(df_feats.columns)]
    # Mean de cada feature
    train = pd.concat([X_train, y_train], axis=1)
    df_feats_train_mean = pd.DataFrame(
        train.mean(), columns=[user_name + "-" + model_id]
    ).T
    # target_enc_model y dicc_models_predict_nans
    if target_enc_model:
        target_enc_model_dir = (
            f"{dir_models}/target_enc_model_"
            + user_name
            + "-"
            + model_id
            + ".pkl"
        )
        with open(target_enc_model_dir, "wb") as file:
            pickle.dump(target_enc_model, file)
    if dicc_models_predict_nans:
        dicc_models_predict_nans_dir = (
            f"{dir_models}/dicc_predict_nans_"
            + user_name
            + "-"
            + model_id
            + ".pkl"
        )
        with open(dicc_models_predict_nans_dir, "wb") as file:
            pickle.dump(dicc_models_predict_nans, file)
    if feature_validation_dict:
        feature_validation_dict_dir = (
            f"{dir_models}/feature_validation_dict_"
            + user_name
            + "-"
            + model_id
            + ".pkl"
        )
        with open(feature_validation_dict_dir, "wb") as file:
            pickle.dump(feature_validation_dict, file)
    if save:
        # Guardamos el modelo en la carpeta de models
        Pkl_Filename = (
            f"{dir_models}/model_" + user_name + "-" + model_id + ".pkl"
        )
        if dir_results_files.startswith("s3"):
            with fs.open(Pkl_Filename, "wb") as file:
                pickle.dump(model, file)
        else:
            with open(Pkl_Filename, "wb") as file:
                pickle.dump(model, file)
        # Guardamos los parametros, metricas y stats
        df_params.to_csv(
            f"{dir_results_files}/params_{user_name}-{model_id}.csv",
            index_label="Model",
        )
        df_metrics.to_csv(
            f"{dir_results_files}/metrics_{user_name}-{model_id}.csv",
            index_label="Model",
        )
        df_stats.to_csv(
            f"{dir_results_files}/stats_{user_name}-{model_id}.csv",
            index_label="Model",
        )
        df_feats.to_csv(
            f"{dir_results_files}/features_train_cols_{user_name}-{model_id}.csv",
            index_label="Model",)

        df_feats_train_mean.to_csv(
            f"{dir_results_files}/features_train_mean_"
            + f"{user_name}-{model_id}.csv",
            index_label="Model",
        )
        print("Modelo, parametros, metricas y stats guardados exitosamente")
class ClassModelResults:
    def __init__(self, last_results=0, df_results=None):
        self.last_results = last_results
        self.df_results = {}
        self.df_results["params"] = pd.DataFrame()
        self.df_results["metrics"] = pd.DataFrame()
        self.df_results["stats"] = pd.DataFrame()
        self.df_results["features_train_cols"] = pd.DataFrame()
        self.df_results["features_train_mean"] = pd.DataFrame()
    def get_model_results(self, dir_results_files="results_files",only_last_files=0):
        tipo_file = ["params","metrics","stats","features_train_cols","features_train_mean"]
        df_results = {}
        for t in tipo_file:
            if dir_results_files.startswith("s3"):
                results_files = fs.glob(dir_results_files + "/*")
                results_files = [f.split("/")[-1] for f in results_files]
            else:
                results_files = os.listdir(f"{dir_results_files}/")
                if len(results_files) == 0:
                    os_walk = os.walk(f"{dir_results_files}/", topdown=False)
                    results_files = [[n for n in fls] for _, _, fls in os_walk][
                        0
                    ]
            wanted_files = [f for f in results_files if f.startswith(t)]
            #### ESTO ES PARA SOLO OBTENER LOS ULTIMOS X FILES 
            if only_last_files>0:
                def get_date_from_id(unique_id):
                    return f"{unique_id[:4]}-{unique_id[4:6]}-{unique_id[6:8]} {unique_id[8:10]}:{unique_id[10:12]}:{unique_id[12:14]}"
                id_wanted_files = [f.split('-')[-1].split('.')[0] for f in wanted_files]
                ## Para todos los files antes de este fix
                for i,f in enumerate(id_wanted_files):
                    if len(f)!=14:
                        id_wanted_files[i]="202208"+f
                date_wanted_files = [get_date_from_id(v) for v in id_wanted_files]
                date_wanted_files = [ datetime.datetime.strptime(v,'%Y-%m-%d %H:%M:%S') for v in date_wanted_files]
                date_wanted_files_sort = list(np.argsort(date_wanted_files))
                date_wanted_files_sort.reverse()
                wanted_files = np.array(wanted_files)[date_wanted_files_sort]
                wanted_files = wanted_files[:only_last_files]
            # Para eliminar los files que correspondan a otro tipo
            not_wanted_files = []
            otros_tipo_file = list(set(tipo_file) - set([t]))
            for tf in otros_tipo_file:
                # Por ejemplo si el file comienza con features_train, hay que quitar
                # todos los que digan features_train_mean
                # Sin embargo si el file comienza con features_train_mean no hace
                # falta quitar los que comienzan con features_train porque no estaran en
                # el listado
                not_wanted_files += [
                    f
                    for f in wanted_files
                    if (f.startswith(tf)) & (len(tf) > len(t))
                ]
            wanted_files = list(set(wanted_files) - set(not_wanted_files))
            dfs = []
            for w in wanted_files:
                if int(w.split("-")[-1].split(".")[0]) > self.last_results:
                    try:
                        df = pd.read_csv(f"{dir_results_files}/" + w)
                        dfs.append(df)
                    except BaseException:
                        pass
            if len(dfs) > 0:
                df_concated = pd.concat(dfs, axis=0)
                df_results[t] = df_concated
                self.df_results[t] = pd.concat(
                    [self.df_results[t], df_results[t]], axis=0
                ).drop_duplicates()
        if len(df_results) > 0:
            key = list(df_results.keys())[0]
            self.last_results = int(
                str(df_results[key]["Model"].iloc[-1]).split("-")[-1]
            )
        return self.df_results


    def load_best_model(
        self,
        metrica="F1_score",
        dir_results_files="results_files",
        dir_models="models",
    ):
        """
        metrica:
        - F1_score
        - Accuracy
        - Recall
        - Precision
        - MSE
        - R2
        - Explained_variance
        """
        df_results = self.get_model_results(dir_results_files)
        df_metrics = df_results["metrics"].set_index("Model")
        best_model_idx = df_metrics[metrica].idxmax()
        best_model_name = df_metrics.loc[best_model_idx].name
        # print(best_model_name)
        best_model = load_model(best_model_name, dir_models)
        return best_model

def file_exists_custom(path_to_file):
    try:
        open(path_to_file)
        return True
    except BaseException:
        return False

def load_model(chosen_model_name, dir_models):
    dict_results = {}
    # dicc_predict_nans
    path_to_file = f"{dir_models}/dicc_predict_nans_{chosen_model_name}.pkl"
    if file_exists_custom(path_to_file):
        with open(path_to_file, "rb") as input_file:
            dicc_predict_nans = pickle.load(input_file)
        dict_results["dicc_predict_nans"] = dicc_predict_nans
    # target_enc_model
    path_to_file = f"{dir_models}/target_enc_model_{chosen_model_name}.pkl"
    if file_exists_custom(path_to_file):
        with open(path_to_file, "rb") as input_file:
            target_enc_model = pickle.load(input_file)
        dict_results["target_enc_model"] = target_enc_model
    # chosen_model
    with open(
        f"{dir_models}/model_{chosen_model_name}.pkl", "rb"
    ) as input_file:
        chosen_model = pickle.load(input_file)
    dict_results["chosen_model"] = chosen_model
    return dict_results
def export_model(chosen_model_name, dir_models,dir_export):
    dict_results = {}
    # dicc_predict_nans
    path_to_file = f"{dir_models}/dicc_predict_nans_{chosen_model_name}.pkl"
    filehandler = open(path_to_file, "rb")
    file_obj = pickle.load(filehandler)
    filename = path_to_file.split('/')[-1]
    dir_export_file = dir_export+f'/{filename}'
    filehandler2 = open(dir_export_file, "wb")
    pickle.dump(file_obj,filehandler2)
    # target_enc_model_
    path_to_file = f"{dir_models}/target_enc_model_{chosen_model_name}.pkl"
    filehandler = open(path_to_file, "rb")
    file_obj = pickle.load(filehandler)
    filename = path_to_file.split('/')[-1]
    dir_export_file = dir_export+f'/{filename}'
    filehandler2 = open(dir_export_file, "wb")
    pickle.dump(file_obj,filehandler2)
    # model_
    path_to_file = f"{dir_models}/model_{chosen_model_name}.pkl"
    filehandler = open(path_to_file, "rb")
    file_obj = pickle.load(filehandler)
    filename = path_to_file.split('/')[-1]
    dir_export_file = dir_export+f'/{filename}'
    filehandler2 = open(dir_export_file, "wb")
    pickle.dump(file_obj,filehandler2)