import os
import pickle
import datetime
import warnings

# from time import sleep

import pandas as pd

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import svm
# from joblib import dump, load
# from sklearn.utils import shuffle

# from sklearn.naive_bayes import MultinomialNB
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import (
#     train_test_split,
#     cross_validate,
# )
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    # cohen_kappa_score,
    # confusion_matrix,
    mean_squared_error,
    explained_variance_score,
    r2_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")




def save_model(model,
               X_train,
               X_test,
               y_train,
               y_test,
               user_name,
               tipo_model,
               comentario = "",
               params_extras={},
               metrics_extras={},
               stats_extras={},
               target_enc_model = "",
               dicc_models_predict_nans = "",
               save=0,
              dir_results_files = 'results_files',
              dir_models = 'models'):
    
    """
    tipo_model: 1 = Clasificacion
    tipo_model: 2 = Regression
    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    from sklearn.model_selection import cross_validate
    from sklearn.utils import shuffle
    
    if os.path.exists(dir_models)==False:
        os.mkdir(dir_models)
    
    if os.path.exists(dir_results_files)==False:
        os.mkdir(dir_results_files)
        
    today = str(datetime.datetime.now().day) +'-'+ str(datetime.datetime.now().month) +'-'+ str(datetime.datetime.now().year) + f" {datetime.datetime.now().hour}:{datetime.datetime.now().minute}"
    today_format = datetime.datetime.strptime(today,'%d-%m-%Y %H:%M')  

    def unique_id():
        from datetime import datetime
        dt = datetime.now()
        return f'{dt.day:02d}'+f'{dt.hour:02d}'+f'{dt.minute:02d}'+f'{dt.second:02d}'
    
    ## Generamos el id del modelo
    model_id = unique_id()
    
    ## Parametros del modelo
    dicc_params = model.get_params()
    dicc_params["Algoritmo"] = str(model).split("(")[0]
    dicc_params["Tipo_modelo"] = tipo_model
    
    ## Parametros de las notas
    dicc_params.update(params_extras)
    
    ## DF
    df_params = pd.DataFrame(dicc_params,index=[user_name+"-"+model_id])
    
    ## Metricas del modelo
    if tipo_model==1:
        #Classification
#         from sklearn.metrics import accuracy_score
#         from sklearn.metrics import recall_score
#         from sklearn.metrics import precision_score
#         from sklearn.metrics import f1_score
#         from sklearn.metrics import cohen_kappa_score
#         from sklearn.metrics import confusion_matrix
        
        metrics= {}
        metrics["Tipo_metrica"] = 'Clasificacion'
        metrics["Tipo_modelo"] = tipo_model
        metrics["Algoritmo"] = str(model).split("(")[0]
        metrics["umbral"] =  0.5
        
        X_test_shuffled, y_test_shuffled = shuffle(X_test, y_test, random_state=0)
        scores = cross_validate(model, X_test_shuffled, y_test_shuffled, cv=5,scoring=['f1_macro','precision_macro',
                                                           'recall_macro','accuracy','roc_auc'])
        
        metrics["AUC"] = scores["test_roc_auc"].mean()
        metrics["Gini"] = (metrics["AUC"]*2)-1
        metrics["F1_score"] = scores["test_f1_macro"].mean()
        metrics["Accuracy"] = scores["test_accuracy"].mean()
        metrics["Recall"] = scores["test_recall_macro"].mean()
        metrics["Precision"] = scores["test_precision_macro"].mean()
        
            
        
    
    elif tipo_model==2:
        #Regression
        from metrics import mean_squared_error
        from metrics import explained_variance_score
        from metrics import r2_score

        metrics= {}
        metrics["Tipo_metrica"] = 'Regresion'
        metrics["Tipo_modelo"] = tipo_model
        metrics["Algoritmo"] = str(model).split("(")[0]
        metrics["MSE"] = mean_squared_error(y_test,model.predict(X_test))
        metrics["R2"] = r2_score(y_test,model.predict(X_test))
        metrics["Explained_variance"] = explained_variance_score(y_test,model.predict(X_test))
        

    ## metrics_extras
    metrics.update(metrics_extras)
    
    metrics["Fecha"] = today_format
    metrics["Comentario"] = comentario
    
    ## print_metrics
    print()
    print('saved metrics ')
    print('MODEL ID: ',user_name+"-"+model_id)
    for k in metrics.keys():
        print(f"{k}: {metrics[k]}")
    print()
    
    ## Df
    df_metrics = pd.DataFrame(metrics,index=[user_name+"-"+model_id])
    

    ## Stats de la data del modelo
    stats = {}
    stats["Algoritmo"] = str(model).split("()")[0]
    stats["Tipo_modelo"] = tipo_model
    stats['mean_xtrain'] = X_train.describe().iloc[1,:].mean()
    stats['mean_ytrain'] = y_train.mean()
    stats['mean_xtest'] = X_test.describe().iloc[1,:].mean()
    stats['mean_ytest'] = y_test.mean()
    
    ## stats_extras
    stats.update(stats_extras)
    
    stats["Fecha"] = today_format
    stats["Comentario"] = comentario
    
    ## DF
    df_stats = pd.DataFrame(stats,index=[user_name+"-"+model_id])
    
    
    # Features de la data de train
    train = pd.concat([X_train, y_train], axis=1)
    # ^ ES IMPORTANTE que el target este de ultimo!
    df_feats = pd.DataFrame(
        list(train.columns), columns=[user_name + "-" + model_id]
    ).T

    # Mean de cada feature
    train = pd.concat([X_train, y_train], axis=1)
    df_feats_train_mean = pd.DataFrame(
        train.mean(), columns=[user_name + "-" + model_id]
    ).T

      
    
    ## target_enc_model y dicc_models_predict_nans
    if target_enc_model !="":
        target_enc_model_dir = f"{dir_models}/target_enc_model_"+user_name+"-"+model_id+".pkl"
        with open(target_enc_model_dir, 'wb') as file: 
                pickle.dump(target_enc_model, file)
                
    if dicc_models_predict_nans !="":
        dicc_models_predict_nans_dir = f"{dir_models}/dicc_predict_nans_"+user_name+"-"+model_id+".pkl"
        with open(dicc_models_predict_nans_dir, 'wb') as file:  
                pickle.dump(dicc_models_predict_nans, file)


    if save:
        ## Guardamos el modelo en la carpeta de models
        Pkl_Filename = f"{dir_models}/model_"+user_name+"-"+model_id+".pkl"
        
        if dir_results_files.startswith('s3'):
            with open(Pkl_Filename, 'wb') as file:  
                pickle.dump(model, file)
        else:
            with open(Pkl_Filename, 'wb') as file:  
                pickle.dump(model, file)
        
        
        
        ## Guardamos los parametros, metricas y stats
        df_params.to_csv(f"{dir_results_files}/params_"+user_name+"-"+model_id+".csv",index_label='Model')
        df_metrics.to_csv(f"{dir_results_files}/metrics_"+user_name+"-"+model_id+".csv",index_label='Model')
        df_stats.to_csv(f"{dir_results_files}/stats_"+user_name+"-"+model_id+".csv",index_label='Model')
        df_feats.to_csv(f"{dir_results_files}/features_train_cols_{user_name}-{model_id}.csv",index_label="Model",)
        df_feats_train_mean.to_csv(f"{dir_results_files}/features_train_mean_"+ f"{user_name}-{model_id}.csv",index_label="Model",)
        print("Modelo, parametros, metricas y stats guardados exitosamente")
    
    
    
    
    
    


class ClassModelResults: 
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    def __init__(self, last_results= 0,df_results=None):
        self.last_results= last_results
        self.df_results = {}
        self.df_results["params"] = pd.DataFrame()
        self.df_results["metrics"] = pd.DataFrame()
        self.df_results["stats"] = pd.DataFrame()
        self.df_results["features_train_cols"] = pd.DataFrame()
        self.df_results["features_train_mean"] = pd.DataFrame()
        
    
    def get_model_results(self,
              dir_results_files = 'results_files'):
        
        import os
        
        tipo_file = ["params","metrics","stats","features_train_cols","features_train_mean"]
        df_results = {}
        for t in tipo_file:
            
            if dir_results_files.startswith('s3'):
                results_files = glob.glob(dir_results_files+"/*")
                results_files = [f.split('/')[-1] for f in results_files]
                
            else:
                results_files = os.listdir(f"{dir_results_files}/")
                if len(results_files)==0:
                    results_files = [[name for name in files] for root, dirs, files in os.walk(f"{dir_results_files}/", topdown=False)][0]

            wanted_files = [f for f in results_files if f.startswith(t)]

            dfs = []
            for w in wanted_files:
                if int(w.split('-')[-1].split(".")[0])>self.last_results:
                    try:
                        df = pd.read_csv(f"{dir_results_files}/"+w)
                        dfs.append(df)
                        #print("Nuevo file")
                    except:
                        pass

            if len(dfs) >0:
                df_concated = pd.concat(dfs,axis=0)
                df_results[t] = df_concated
                self.df_results[t] = pd.concat([self.df_results[t],df_results[t]],axis=0).drop_duplicates()
        
        if len(df_results)>0:
            key = list(df_results.keys())[0]
            self.last_results = int(str(df_results[key]["Model"].iloc[-1]).split("-")[-1])
        
        return self.df_results
    
    
    def load_model(self,model_id,dir_models = 'models'):
        import pickle
        
        if dir_models.startswith('s3'):
            with self.open(f"{dir_models}/model_{model_id}.pkl", "rb") as input_file:
                model_loaded = pickle.load(input_file)
        else:
            with open(f"{dir_models}/model_{model_id}.pkl", "rb") as input_file:
                model_loaded = pickle.load(input_file)
            
        return model_loaded
    
    
    def load_best_model(self,metrica="F1_score",
              dir_results_files = 'results_files',
              dir_models = 'models'):
        
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
        df_metrics = df_results['metrics'].set_index("Model")
        best_model_idx = df_metrics[metrica].idxmax()
        best_model_name = df_metrics.loc[best_model_idx].name
        
        #print(best_model_name)
        best_model = self.load_model(best_model_name,dir_models)
        
            
        return best_model
        
        
        


### Funcion para load model por fuera
def load_model(chosen_model_name,dir_models):
    import pickle
    dict_results = {}
    
    ## dicc_predict_nans
    path_to_file = f"{dir_models}/dicc_predict_nans_{chosen_model_name}.pkl"
    if os.path.isfile(path_to_file):
        with open(path_to_file, "rb") as input_file:
            dicc_predict_nans = pickle.load(input_file)
        dict_results['dicc_predict_nans'] = dicc_predict_nans
        
    
    ## target_enc_model
    path_to_file = f"{dir_models}/target_enc_model_{chosen_model_name}.pkl"
    if os.path.isfile(path_to_file):
        with open(path_to_file, "rb") as input_file:
            target_enc_model = pickle.load(input_file)
        dict_results['target_enc_model'] = target_enc_model
    
    ## chosen_model
    with open(f"{dir_models}/model_{chosen_model_name}.pkl", "rb") as input_file:
        chosen_model = pickle.load(input_file)
    dict_results['chosen_model'] = chosen_model
    
    return dict_results