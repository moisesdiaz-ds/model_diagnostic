#!/usr/bin/env python
# coding: utf-8

# # Mis funciones

# ## Para limpieza/manejo de datos

# In[1]:


def revisado_datos_na(df):
    
    import pandas as pd

    cols = list(df.columns)
    nas_count = 0
    for c in cols:
        
        nas = pd.isnull(df[c]).sum()
        if nas > 0:
            nas_count += 1
            
            print(f"La columna {c} tiene {nas} valores NA")
            
    if nas_count < 1:
        print(f"El dataset no tiene valores NA")


# In[ ]:


def fillna_with_sample_value(df,col):

    import pandas as pd

    """Fill NaNs with values of the same columns"""
    
    rows_nans = df[pd.isnull(df[col])]
    nans_n = len(rows_nans)
    sample_nans = list(df[col].dropna().sample(nans_n).values)
    
    for i,n in enumerate(list(rows_nans.index.values)):
        
        df.loc[n,col] = sample_nans[i]
    
    return df


# In[ ]:


def procesado_df(df,y_columns,col_selection,training_data):
    
    X = []
    for c in col_selection:
        
        X.append(df[c])


    if training_data == 1:
        Y = df[y_columns]
    else:
        Y = 0
    
    return X,Y


# In[3]:


def train_test_split_folds(df,folds,test_size=0.25,training_data=0):

    import numpy as np

    ## Para cuando los folds sean menores que 4
    import sys
    limit_fold = 3
    if(folds <= limit_fold and folds != 1):
        sys.exit(f"Los folds debe ser mayores a {limit_fold} o igual a 1") 


        
        
    ##Para cuando los folds sean igual a 1
    if(folds == 1):

        from sklearn.model_selection import train_test_split 

        train_sets = []
        test_sets = []
        
        if(training_data == 1):

            train, test = train_test_split(df, test_size = test_size)
            
            train_sets.append(train)
            test_sets.append(test)
            
        else:
            
            train_sets.append(df)

        return train_sets,test_sets

    
    

    ## Comportamiento normal de la funcion
    import sklearn 

    df = sklearn.utils.shuffle(df) 

    split = round(100/folds)/100

    train_sets = []
    test_sets = []
    subset = int(np.floor(split * len(df)))
    df_copy = df

    for f in range(folds):

        df = df_copy

        subset_for = subset * f
        testing_set = df[subset_for:subset_for+subset]

        test_sets.append(testing_set)

        list_testing = list(testing_set.index)

        training_set = df.drop(list_testing, axis=0)

        train_sets.append(training_set)

    return train_sets,test_sets





def crossvalidation(df,features,target,folds,model,test_size=0.25,train_model=1):
    
    """
    If the model is already trained, train_model should be 0
    
    """

    
    import mis_funciones as mf
    
    train_sets,test_sets = mf.train_test_split_folds(df,folds)
    
    models_sets = []
    for tr in train_sets:
        if train_model == 1:
            model.fit(tr[features],tr[target])
            models_sets.append(model)
        else:
            models_sets.append(model)
    
    scores_sets = []
    for tes in test_sets:
        
        score = model.score(tes[features],tes[target])
        scores_sets.append(score)
        
        
    return scores_sets
    




# In[1]:


def cero_uno(data):
    value = 0
    if data >= 0.5:
        value = 1
    
    return value


# In[ ]:


def rstr(df, pred=None): 

    import pandas as pd

    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ration = (df.isnull().sum()/ obs) * 100
    skewness = df.skew()
    kurtosis = df.kurt() 
    print('Data shape:', df.shape)
    
    if pred is None:
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing ration', 'uniques', 'skewness', 'kurtosis']
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis], axis = 1)

    else:
        corr = df.corr()[pred]
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis, corr], axis = 1, sort=False)
        corr_col = 'corr '  + pred
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing_ration', 'uniques', 'skewness', 'kurtosis', corr_col ]
    
    str.columns = cols
    dtypes = str.types.value_counts()
    print('___________________________\nData types:\n',str.types.value_counts())
    print('___________________________')
    return str

def graficar_correlaciones(df,corr_list,target,cant_cols=3,package="matplotlib"):
    """Grafica las variables que se le agreguen como imputs
       Es necesario importar make_subplots y graph_objects y numpy """

    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    cant_rows = int(np.ceil(len(corr_list)/cant_cols))
    
    if package == "plotly":
        fig = make_subplots(rows=cant_rows, cols=cant_cols,shared_yaxes=True)
        
    elif package == "matplotlib":
        fig = plt.figure(figsize=(25,15))
        axes = []

    col_count = 1
    row_count = 1
    row_count_plt = 1
    for i,g in enumerate(corr_list):
        
        if package == "plotly":
            fig.add_trace(go.Scatter(x=df[g],y=df[target],mode="markers"),row=row_count,col=col_count)
            fig.update_xaxes(title_text=g, row=row_count, col=col_count)
            fig.update_yaxes(title_text=target, row=row_count, col=col_count)
            
        elif package == "matplotlib":
            axes.append("ax"+"1")
            axes[i] = fig.add_subplot(str(cant_rows)+str(cant_cols)+str(row_count_plt))
            axes[i].plot(df[g], df[target], "o")
            axes[i].set_xlabel(g, fontsize=16)
            axes[i].set_ylabel(target, fontsize=16)
        

        if col_count == cant_rows:
            col_count = 1
            row_count += 1
        else:
            col_count += 1
            
        row_count_plt += 1
    
    if package == "plotly":
        fig.update_layout(width=950,height=900)
        fig.show()
        
    elif package == "matplotlib":
        plt.tight_layout(pad=1, w_pad=1, h_pad=2.0)
        #fig.suptitle("Title centered above all subplots", fontsize=22)
        plt.show()




def import_all_alg_sklearn(tipo,show_import=0,rare_algs=0):
    """
    tipo 1 = classifier
    tipo 2 = regressor
    tipo 3 = cluster
    
    rare_algs es si se desea incluir los algoritmos los algoritmos que son raros o dan problemas
    """
    
    import warnings
    warnings.filterwarnings('ignore')
    
    from sklearn.utils.testing import all_estimators
    
    if tipo == 1:
        tipo_alg = "classifier"
    elif tipo == 2:
        tipo_alg = "regressor"
    elif tipo == 3:
        tipo_alg = "cluster"
        
    estimators = all_estimators(type_filter=tipo_alg)

    no_tener_encuenta =["GeneralizedLinearRegressor","CheckingClassifier","HashingVectorizer","TfidfTransformer","_BaseEncoder","_BaseImputer","_BinMapper"] 
    #Estos son los que no funcionan al importarse

    from_list = []
    
    ##### show_import
    if show_import == 1:
        alg = []
        for name, class_ in estimators:
            if name not in no_tener_encuenta:
                from_item = str(class_).split(".")
                from_item = (from_item[0]+"."+from_item[1]).replace("<class '","")
                print(f"from {from_item} import {name}")
         
    
    if tipo == 1:
        
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.ensemble import BaggingClassifier
        from sklearn.naive_bayes import BernoulliNB
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.naive_bayes import CategoricalNB
        from sklearn.multioutput import ClassifierChain
        from sklearn.naive_bayes import ComplementNB
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.dummy import DummyClassifier
        from sklearn.tree import ExtraTreeClassifier
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.semi_supervised import LabelPropagation
        from sklearn.semi_supervised import LabelSpreading
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.svm import LinearSVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.neural_network import MLPClassifier
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.neighbors import NearestCentroid
        from sklearn.svm import NuSVC
        from sklearn.multiclass import OneVsOneClassifier
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.multiclass import OutputCodeClassifier
        from sklearn.linear_model import PassiveAggressiveClassifier
        from sklearn.linear_model import Perceptron
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        from sklearn.neighbors import RadiusNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import RidgeClassifier
        from sklearn.linear_model import RidgeClassifierCV
        from sklearn.linear_model import SGDClassifier
        from sklearn.svm import SVC
        from sklearn.ensemble import StackingClassifier
        from sklearn.ensemble import VotingClassifier
        
        alg_list = [AdaBoostClassifier,
                     BaggingClassifier,
                     BernoulliNB,
                     CalibratedClassifierCV,
                     CategoricalNB,
                     ClassifierChain,
                     ComplementNB,
                     DecisionTreeClassifier,
                     ExtraTreeClassifier,
                     ExtraTreesClassifier,
                     GaussianNB,
                     GaussianProcessClassifier,
                     GradientBoostingClassifier,
                     HistGradientBoostingClassifier,
                     KNeighborsClassifier,
                     LinearDiscriminantAnalysis,
                     LinearSVC,
                     LogisticRegression,
                     LogisticRegressionCV,
                     MLPClassifier,
                     MultiOutputClassifier,
                     MultinomialNB,
                     NearestCentroid,
                     NuSVC,
                     OneVsOneClassifier,
                     OneVsRestClassifier,
                     OutputCodeClassifier,
                     PassiveAggressiveClassifier,
                     Perceptron,
                     QuadraticDiscriminantAnalysis,
                     RadiusNeighborsClassifier,
                     RandomForestClassifier,
                     RidgeClassifier,
                     RidgeClassifierCV,
                     SGDClassifier,
                     SVC,
                     StackingClassifier,
                     VotingClassifier]
        
        ## Lista de algoritmos quitados porque dan problemas
        alg_rare_list = [DummyClassifier,
                         LabelPropagation,
                         LabelSpreading,]

        ## para incluir dichos algoritmos raros
        if rare_algs:
            alg_list.extend(rare_algs)
        

    elif tipo == 2:
        
        from sklearn.linear_model import ARDRegression
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import BaggingRegressor
        from sklearn.linear_model import BayesianRidge
        from sklearn.cross_decomposition import CCA
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.dummy import DummyRegressor
        from sklearn.linear_model import ElasticNet
        from sklearn.linear_model import ElasticNetCV
        from sklearn.tree import ExtraTreeRegressor
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import HuberRegressor
        from sklearn.isotonic import IsotonicRegression
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.linear_model import Lars
        from sklearn.linear_model import LarsCV
        from sklearn.linear_model import Lasso
        from sklearn.linear_model import LassoCV
        from sklearn.linear_model import LassoLars
        from sklearn.linear_model import LassoLarsCV
        from sklearn.linear_model import LassoLarsIC
        from sklearn.linear_model import LinearRegression
        from sklearn.svm import LinearSVR
        from sklearn.neural_network import MLPRegressor
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.linear_model import MultiTaskElasticNet
        from sklearn.linear_model import MultiTaskElasticNetCV
        from sklearn.linear_model import MultiTaskLasso
        from sklearn.linear_model import MultiTaskLassoCV
        from sklearn.svm import NuSVR
        from sklearn.linear_model import OrthogonalMatchingPursuit
        from sklearn.linear_model import OrthogonalMatchingPursuitCV
        from sklearn.cross_decomposition import PLSCanonical
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.linear_model import PassiveAggressiveRegressor
        from sklearn.linear_model import RANSACRegressor
        from sklearn.neighbors import RadiusNeighborsRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import RegressorChain
        from sklearn.linear_model import Ridge
        from sklearn.linear_model import RidgeCV
        from sklearn.linear_model import SGDRegressor
        from sklearn.svm import SVR
        from sklearn.ensemble import StackingRegressor
        from sklearn.linear_model import TheilSenRegressor
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.ensemble import VotingRegressor
        from sklearn.calibration import _SigmoidCalibration
        
        alg_list = [ARDRegression,
                     AdaBoostRegressor,
                     BaggingRegressor,
                     BayesianRidge,
                     DecisionTreeRegressor,
                     ElasticNet,
                     ElasticNetCV,
                     ExtraTreeRegressor,
                     ExtraTreesRegressor,
                     GradientBoostingRegressor,
                     HistGradientBoostingRegressor,
                     HuberRegressor,
                     IsotonicRegression,
                     KNeighborsRegressor,
                     KernelRidge,
                     Lars,
                     LarsCV,
                     Lasso,
                     LassoCV,
                     LassoLars,
                     LassoLarsCV,
                     LassoLarsIC,
                     LinearRegression,
                     LinearSVR,
                     MLPRegressor,
                     MultiOutputRegressor,
                     MultiTaskElasticNet,
                     MultiTaskElasticNetCV,
                     MultiTaskLasso,
                     MultiTaskLassoCV,
                     OrthogonalMatchingPursuitCV,
                     PLSRegression,
                     RANSACRegressor,
                     RandomForestRegressor,
                     RegressorChain,
                     Ridge,
                     RidgeCV,
                     StackingRegressor,
                     TheilSenRegressor,
                     TransformedTargetRegressor,
                     VotingRegressor,
                     _SigmoidCalibration]
    
        ## Lista de algoritmos quitados porque dan problemas
        alg_rare_list = [GaussianProcessRegressor,
                         DummyRegressor,
                         SVR,
                         SGDRegressor,
                         RadiusNeighborsRegressor,
                         PLSCanonical,
                         OrthogonalMatchingPursuit,
                         NuSVR,
                         CCA,
                         PassiveAggressiveRegressor]
    
        ## para incluir dichos algoritmos raros
        if rare_algs:
            alg_list.extend(rare_algs)
    
    elif tipo == 3:
        
        from sklearn.cluster import AffinityPropagation
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.cluster import Birch
        from sklearn.cluster import DBSCAN
        from sklearn.cluster import FeatureAgglomeration
        from sklearn.cluster import KMeans
        from sklearn.cluster import MeanShift
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.cluster import OPTICS
        from sklearn.cluster import SpectralClustering
        
        alg_list = [AffinityPropagation,
                     AgglomerativeClustering,
                     Birch,
                     DBSCAN,
                     FeatureAgglomeration,
                     KMeans,
                     MeanShift,
                     MiniBatchKMeans,
                     OPTICS,
                     SpectralClustering]
        
                    
    return alg_list






def plot_best_festure_sel(df,target,alg_list,tipo,show_results=1):
    
    """tipo = 1 significa clasificacion
       tipo = 2 significa regresion
       tipo = 3 significa clustering"""
    
    import pandas as pd
    import mis_funciones as mf
    from sklearn.model_selection import train_test_split
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    import warnings
    warnings.filterwarnings('ignore')
    
    #Solo tomamos las columnas numericas
    numeric_cols = set(df._get_numeric_data().columns)
    all_cols = set(df.columns) 
    categorical_cols = all_cols - numeric_cols
    
    
    #Para tratar variables categoricas
    dummy_cols = []
    for c in categorical_cols:
        if len(df[c].unique()) < 11 and df[c].isnull().sum() < 1:
            dummy_cols.append(pd.get_dummies(df[c], prefix=c))
        
        df = df.drop(c, axis=1)

    dummy_cols.insert(0,df)
    df = pd.concat(dummy_cols, axis = 1)
    
    
    #Llenamos los valores NA
    for n in numeric_cols:
        df[n] = df[n].fillna(df[n].mean())
    
    #Correlacionar variables
    features = np.abs(df.corr()[target])
    features.colums = ["score"]
    features = features.sort_values(ascending=False)
    
    
    #Algoritmos
    for alg in alg_list:
        
        alg_name = []
        features_names_list = []
        feature_sel_pos = []
        results = []
        try:
            alg = alg()
            for i,f in enumerate(range(3,len(features.index))):

                features_names = features.index[1:f]
                X = df[features_names]
                y = df[target]
                alg_name.append(str(alg).strip())
                features_names_list.append(str(list(features_names))) #Solo para tenerlos anotados
                feature_sel_pos.append(i)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
                
                if tipo < 3:
                    model_fitted = alg.fit(X_train, y_train)
                else:
                    model_fitted = alg.fit(X_train)
                
                score = model_fitted.score(X_test, y_test)

                results.append(np.round(score,4))
                
        except:
            print(f"Hubo un error con el algoritmo {alg}")
            continue
            
            
        #### GUARDAMOS LOS RESULTADOS
        alg_scores = pd.DataFrame({"alg":alg_name,"feature_sel":features_names_list,
                                   "feature_sel_pos":feature_sel_pos,"score":results})
        
        if "df_scores" in locals(): #Confirmar si la variable existe
            df_scores = pd.concat([df_scores,alg_scores],axis=0)
        else:
            df_scores = alg_scores

        #### MOSTRAMOS LOS RESULTADOS
        if show_results:
            fig = plt.figure(figsize=(12,4))
            plt.plot(results)
            plt.xticks(np.arange(len(results)))
            plt.grid(True)
            plt.show()
        
            best_score_p = np.argmax(results)
            best_score = round(np.max(results),6)
            print("Algoritmo utilizado: ",alg)
            print("La mejor seleccion de features es ",features_names_list[best_score_p])
            print("En la posicion ",best_score_p)
            print("Con un score de ",best_score)
            
      
    #### AGRUPAMOS los resultados obtenidos
    def listing(lista_elem):
        return list(lista_elem)[0]

    df_scores_grouped = df_scores.groupby("feature_sel_pos").sum().sort_values(by="score",ascending=False)
    df_feat_list = pd.DataFrame(df_scores.groupby("feature_sel_pos")["feature_sel"].apply(listing))
    df_scores_list = pd.DataFrame(df_scores.groupby("feature_sel_pos")["score"].apply(list))
    df_scores_list.columns = ['score_list']
    df_alg_list = pd.DataFrame(df_scores.groupby("feature_sel_pos")["alg"].apply(list))

    df_merged = pd.merge(left = df_scores_grouped, right = df_feat_list,
                           how = "inner", left_on = "feature_sel_pos", right_on = "feature_sel_pos")
        
    df_merged = pd.merge(left =  df_merged, right = df_scores_list,
                           how = "inner", left_on = "feature_sel_pos", right_on = "feature_sel_pos")
    
    df_merged = pd.merge(left = df_merged, right = df_alg_list,
                           how = "inner", left_on = "feature_sel_pos", right_on = "feature_sel_pos")
    



    return df_merged


def remover_acentos(x):
    import unidecode
    return unidecode.unidecode(x)


def buscar_en_nbs_walk(frase_buscar):
    import os
    from shutil import copyfile
    import datetime
    import re

    x = datetime.datetime.now()
    day = x.strftime("%m-%d-%Y")


    for root, dirs, files in os.walk(".", topdown=False):
       for name in files:
        if ".ipynb_checkpoints" not in os.path.join(root, name):
            if (name.endswith(".py") or name.endswith(".ipynb") or name.endswith(".sql")) and ("\\BACKUP\\" not in os.path.join(root, name)):

                try:
                    ruta_file = "C:\\Users\\mediaz\\Jupyter"+ os.path.join(root, name)
                    open_file = open(ruta_file,"r",encoding="cp437", errors='ignore')

                    encontrado = re.search(frase_buscar,  open_file.read())
                    if encontrado:
                        print(os.path.join(root, name))
                except Exception as e:
                    print(e)
                    continue



def buscar_en_nbs(frase_buscar,madre_dir = "C:/Users/mediaz/Jupyter",open_link=0):
    
    import os
    import re
    import webbrowser

    rutas_encontradas = []
    frases_encontrado = []
    #errores = []
    
    
    dirs = [f for f in os.listdir(madre_dir) if os.path.isdir(f)]
    dirs.insert(0,"") #Para que tambien busque en la carpeta actual

    for i,d in enumerate(dirs):
        d = madre_dir+"/"+d
        files_encontrado = buscar_files_in_dir(i,d,frase_buscar)
        rutas_encontradas.extend(files_encontrado)

    if open_link == 1:
        for link in rutas_encontradas:
            webbrowser.open(link)
            
    return rutas_encontradas


def buscar_files_in_dir(i,d,frase_buscar):

    import os
    import re
    import webbrowser
    
    files_encontrado = []
    
    if d.endswith("/"):
        d = d.split("/")[:-1] #Traeme todo menos el ultimo backslash
        d = "/".join(d)

    if ".ipynb_checkpoints" not in d:
        files = os.listdir(d)
        for f in files:
            if f.endswith(".ipynb") or f.endswith(".py"): #Si es un NB
                try:
                    if i < 1:
                        open_file = open(f,"r",encoding="cp437", errors='ignore')
                    else:
                        open_file = open(f"{d}/{f}","r",encoding="cp437", errors='ignore')

                    encontrado = re.search(frase_buscar,  open_file.read())
                    if encontrado:
                        notebook_dir = f"{d}/{f}"
                        notebook_dir = notebook_dir.split("/Dropbox/")[1]
                        notebook_dir = "http://localhost:8888/notebooks/Dropbox/"+notebook_dir
                        files_encontrado.append(notebook_dir)
                        #frases_encontrado.append(encontrado.string)
                except Exception as e:
                    pass
                    #errores.append(f"{d}/{f} ERROR:"+str(e))

            #elif os.path.isdir(f) and ".ipynb_checkpoints" not in f: #Si es un folder
                #print(d+"/"+f)
                #files_profundos_encontrados = buscar_en_nbs(frase_buscar,d+"/"+f) #Funcion recursiva para buscar a varios niveles de profundidad
                #files_encontrado.extend(files_profundos_encontrados) 
    
    
    return files_encontrado



def download_file(file_url, local_path):
    import os
    import tarfile
    import zipfile
    import urllib
 
    if not os.path.isdir(local_path):
        os.makedirs(local_path)
        
    # Download file
    print(">>> downloading")
    filename = os.path.basename(file_url)
    file_local_path = os.path.join(local_path, filename)
    urllib.request.urlretrieve(file_url, file_local_path)
    
    # untar/unzip file 
    if filename.endswith("tgz") or filename.endswith("tar.gz"):
        print(">>> unpacking file:", filename)
        tar = tarfile.open(file_local_path, "r:gz")
        tar.extractall(path = local_path)
        tar.close()
    elif filename.endswith("tar"):
        print(">>> unpacking file:", filename)
        tar = tarfile.open(file_local_path, "r:")
        tar.extractall(path = local_path)
        tar.close()
    elif filename.endswith("zip"):
        print(">>> unpacking file:", filename)
        zip_file = zipfile.ZipFile(file_local_path)
        zip_file.extractall(path = local_path)
        zip_file.close()
    print("Done")



def ppscores_matrix(df,cross_validation=4,random_seed=123):
    """
    Fuente: https://towardsdatascience.com/rip-correlation-introducing-the-predictive-power-score-3d90808b9598
    """
    import warnings
    import pandas as pd
    import ppscore as pps
    warnings.filterwarnings('ignore')

    dicc = {}
    for c in df.columns:
        list_scores = []
        for c2 in df.columns:
            score = pps.score(df, c, c2,cross_validation=cross_validation,random_seed=random_seed)["ppscore"]
            list_scores.append(score)
        dicc[c] = list_scores

    df_scores = pd.DataFrame(dicc)
    df_scores.index = df.columns
    df_scores = df_scores.T

    return df_scores




def corr_no_lineal(df,plotear=0,ver_corr_lineal=0):
    """
    ver_corr_lineal = Es para ver si se desea tambien generar las relacionales lineales (de pearson)
    """
    limit_dif_corr = 0.10
    if ver_corr_lineal:
        limit_dif_corr = 0.00
        
    df_corr = df.corr("spearman") - df.corr()
    df_corr = df_corr.unstack().sort_values(ascending=False)
    df_corr = df_corr[df_corr >= limit_dif_corr].drop_duplicates()

    corrs_non_lin = []
    list_corr_spear = []
    list_corr_pears = []
    for ind in df_corr.index:
        df_corr_spear = df[list(ind)].corr("spearman").unstack().drop_duplicates()
        df_corr_pears = df[list(ind)].corr().unstack().drop_duplicates()
        validacion_map = (df_corr_spear >= 0.4) & (df_corr_spear < 1)

        if validacion_map.sum() > 0:
            corrs_non_lin.append(ind)
            list_corr_spear.append(np.round(df_corr_spear[validacion_map].values,5)[0])
            list_corr_pears.append(np.round(df_corr_pears[validacion_map].values,5)[0])


    if plotear:
        import matplotlib.pyplot as plt
        import seaborn as sns

        for corrs in corrs_non_lin:
            df.plot(corrs[0],corrs[1],kind="scatter")
            plt.show()


    df_output = pd.DataFrame({"variables":corrs_non_lin,
                              "Spearman_corr":list_corr_spear,
                              "Pearson_corr":list_corr_pears})

    df_output["Diferencia_corr"] = df_output["Spearman_corr"] - df_output["Pearson_corr"]

    return df_output




def compress_dataset(data):
    """
        Compress datatype as small as it can
        Parameters

        Author: Hangzhou, Zhejiang
        ----------
        path: pandas Dataframe

        Returns
        -------
            None
    """

    print("ESTA FUNCION NO DEVUELVE OBJETOS, SOLO LOS MODIFICA")
    
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


    INT8_MIN    = np.iinfo(np.int8).min
    INT8_MAX    = np.iinfo(np.int8).max
    INT16_MIN   = np.iinfo(np.int16).min
    INT16_MAX   = np.iinfo(np.int16).max
    INT32_MIN   = np.iinfo(np.int32).min
    INT32_MAX   = np.iinfo(np.int32).max

    FLOAT16_MIN = np.finfo(np.float16).min
    FLOAT16_MAX = np.finfo(np.float16).max
    FLOAT32_MIN = np.finfo(np.float32).min
    FLOAT32_MAX = np.finfo(np.float32).max

    def memory_usage(data, detail=1):
        if detail:
            display(data.memory_usage())
        memory = data.memory_usage().sum() / (1024*1024)
        print("Memory usage : {0:.2f}MB".format(memory))
        return memory


    memory_before_compress = memory_usage(data, 0)
    print()
    length_interval      = 50
    length_float_decimal = 4

    print('='*length_interval)
    for col in data.columns:
        col_dtype = data[col][:100].dtype

        if col_dtype != 'object':
            print("Name: {0:24s} Type: {1}".format(col, col_dtype))
            col_series = data[col]
            col_min = col_series.min()
            col_max = col_series.max()

            if col_dtype == 'float64':
                print(" variable min: {0:15s} max: {1:15s}".format(str(np.round(col_min, length_float_decimal)), str(np.round(col_max, length_float_decimal))))
                if (col_min > FLOAT16_MIN) and (col_max < FLOAT16_MAX):
                    data[col] = data[col].astype(np.float16)
                    print("  float16 min: {0:15s} max: {1:15s}".format(str(FLOAT16_MIN), str(FLOAT16_MAX)))
                    print("compress float64 --> float16")
                elif (col_min > FLOAT32_MIN) and (col_max < FLOAT32_MAX):
                    data[col] = data[col].astype(np.float32)
                    print("  float32 min: {0:15s} max: {1:15s}".format(str(FLOAT32_MIN), str(FLOAT32_MAX)))
                    print("compress float64 --> float32")
                else:
                    pass
                memory_after_compress = memory_usage(data, 0)
                print("Compress Rate: [{0:.2%}]".format((memory_before_compress-memory_after_compress) / memory_before_compress))
                print('='*length_interval)

            if col_dtype == 'int64':
                print(" variable min: {0:15s} max: {1:15s}".format(str(col_min), str(col_max)))
                type_flag = 64
                if (col_min > INT8_MIN/2) and (col_max < INT8_MAX/2):
                    type_flag = 8
                    data[col] = data[col].astype(np.int8)
                    print("     int8 min: {0:15s} max: {1:15s}".format(str(INT8_MIN), str(INT8_MAX)))
                elif (col_min > INT16_MIN) and (col_max < INT16_MAX):
                    type_flag = 16
                    data[col] = data[col].astype(np.int16)
                    print("    int16 min: {0:15s} max: {1:15s}".format(str(INT16_MIN), str(INT16_MAX)))
                elif (col_min > INT32_MIN) and (col_max < INT32_MAX):
                    type_flag = 32
                    data[col] = data[col].astype(np.int32)
                    print("    int32 min: {0:15s} max: {1:15s}".format(str(INT32_MIN), str(INT32_MAX)))
                    type_flag = 1
                else:
                    pass
                memory_after_compress = memory_usage(data, 0)
                print("Compress Rate: [{0:.2%}]".format((memory_before_compress-memory_after_compress) / memory_before_compress))
                if type_flag == 32:
                    print("compress (int64) ==> (int32)")
                elif type_flag == 16:
                    print("compress (int64) ==> (int16)")
                else:
                    print("compress (int64) ==> (int8)")
                print('='*length_interval)

    print()
    memory_after_compress = memory_usage(data, 0)
    print("Compress Rate: [{0:.2%}]".format((memory_before_compress-memory_after_compress) / memory_before_compress))



def organizamos_df(df,proporcion=0.7):

    train_rows = int(len(df)*proporcion)
    test_rows =  len(df) - int(len(df)*proporcion)
    print(df.shape,train_rows,test_rows)

    train = df[0:train_rows]
    test = df[train_rows:train_rows+test_rows]

    train = train.iloc[:,0:]
    test = test.iloc[:,0:]

    return train,test







def extract_mini_dataset(df,porcentaje=0.3,significancia=0.001,print_it=1):
    import pandas as pd
    import numpy as np
    from scipy.stats import ks_2samp

    numeric_cols = set(df._get_numeric_data().columns)
    all_cols = set(df.columns) 
    categorical_cols = all_cols - numeric_cols
    
    dataset_no_listo = 1
    
    while dataset_no_listo:
    
        ###Creamos el mini dataset
        new_len = round(len(df)*porcentaje)
        df_sm = df.sample(new_len,replace=False)
        
        #### Seccion de comprobacion
        
        ##Para las columnas categoricas
        lista_results_cat = []
        for cat in categorical_cols:
            if print_it:
                print(f"Comparando sample de {cat}")
            counts_df = df[cat].value_counts()/sum(df[cat].value_counts())
            counts_df_sm = df_sm[cat].value_counts()/sum(df_sm[cat].value_counts())
            
            diff_cat = abs(counts_df_sm - counts_df)
            resul_diff_cat = sum(diff_cat > significancia)
            
            if resul_diff_cat > 0:
                lista_results_cat.append(resul_diff_cat)
                dataset_no_listo = 1
                if print_it:
                    print(f"La columna {cat} No es estadisticamente igual al {significancia*100}%, se volvera a hacer el sample")
                break
            else:
                lista_results_cat.append(resul_diff_cat)
            
            
        ##Para las columnas numericas
        lista_results_num = []
        if sum(lista_results_cat) == 0:
        
            for num in numeric_cols:
                if print_it:
                    print(f"Comparando sample de {num}")

                diff_num = ks_2samp(df[num], df_sm[num])
                resul_diff_num = diff_num[1] < (1-significancia) #El P-value

                if resul_diff_num > 0:
                    lista_results_num.append(resul_diff_num)
                    dataset_no_listo = 1
                    if print_it:
                        print(f"La columna {num} No es estadisticamente igual al {significancia*100}%, se volvera a hacer el sample")
                    break
                else:
                    lista_results_num.append(resul_diff_num)

                
                
        if (sum(lista_results_cat) == 0) & (sum(lista_results_num) == 0):
            dataset_no_listo = 0
            if print_it:
                print(f"Todas las columnas son estadisticamente iguales al {significancia*100}%")
            return df_sm
        else:
            pass



def tokenizar_col(df,col):
    import pandas as pd
    from nltk.tokenize import sent_tokenize, word_tokenize 

    #df = df.iloc[:100,:] #Solo 100 filas

    ##### Guardamos las diferentes palabras (tokens) en una lista
    tokens = []
    for l in range(len(df)-1):
        text = df[col].iloc[l]
        tokens.extend(word_tokenize(text))

    tokens = set(tokens)
    tokens = list(tokens)
    print(len(tokens)," tokens") #Imprimimos la cantidad de tokens

    def add_token(x,*args):
        token = "".join(args)
        #print(token)
        if str(token) in str(x):
            return 1
        else:
            return 0

    for tok in tokens:
        df[tok] = df[col].apply(add_token,args=tok)

    return df




def generate_fake_dataframe(size, cols, col_names = None, intervals = None, seed = None):
    import pandas as pd
    import numpy as np
    from itertools import cycle

    categories_dict = {'animals': ['cow', 'rabbit', 'duck', 'shrimp', 'pig', 'goat', 'crab', 'deer', 'bee', 'sheep', 'fish', 'turkey', 'dove', 'chicken', 'horse'],
                       'names'  : ['James', 'Mary', 'Robert', 'Patricia', 'John', 'Jennifer', 'Michael', 'Linda', 'William', 'Elizabeth', 'Ahmed', 'Barbara', 'Richard', 'Susan', 'Salomon', 'Juan Luis'],
                       'cities' : ['Stockholm', 'Denver', 'Moscow', 'Marseille', 'Palermo', 'Tokyo', 'Lisbon', 'Oslo', 'Nairobi', 'Río de Janeiro', 'Berlin', 'Bogotá', 'Manila', 'Madrid', 'Milwaukee'],
                       'colors' : ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'purple', 'pink', 'silver', 'gold', 'beige', 'brown', 'grey', 'black', 'white']
                      }
    default_intervals = {"i" : (0,10), "f" : (0,100), "c" : ("names", 5), "d" : ("2020-01-01","2020-12-31")}
    rng = np.random.default_rng(seed)

    first_c = default_intervals["c"][0]
    categories_names = cycle([first_c] + [c for c in categories_dict.keys() if c != first_c])
    default_intervals["c"] = (categories_names, default_intervals["c"][1])
    
    if isinstance(col_names,list):
        assert len(col_names) == len(cols), f"The fake DataFrame should have {len(cols)} columns but col_names is a list with {len(col_names)} elements"
    elif col_names is None:
        suffix = {"c" : "cat", "i" : "int", "f" : "float", "d" : "date"}
        col_names = [f"column_{str(i)}_{suffix.get(col)}" for i, col in enumerate(cols)]

    if isinstance(intervals,list):
        assert len(intervals) == len(cols), f"The fake DataFrame should have {len(cols)} columns but intervals is a list with {len(intervals)} elements"
    else:
        if isinstance(intervals,dict):
            assert len(set(intervals.keys()) - set(default_intervals.keys())) == 0, f"The intervals parameter has invalid keys"
            default_intervals.update(intervals)
        intervals = [default_intervals[col] for col in cols]
    df = pd.DataFrame()
    for col, col_name, interval in zip(cols, col_names, intervals):
        if interval is None:
            interval = default_intervals[col]
        assert (len(interval) == 2 and isinstance(interval, tuple)) or isinstance(interval, list), f"This interval {interval} is neither a tuple of two elements nor a list of strings."
        if col in ("i","f","d"):
            start, end = interval
        if col == "i":
            df[col_name] = rng.integers(start, end, size)
        elif col == "f":
            df[col_name] = rng.uniform(start, end, size)
        elif col == "c":
            if isinstance(interval, list):
                categories = np.array(interval)
            else:
                cat_family, length = interval
                if isinstance(cat_family, cycle):
                    cat_family = next(cat_family)
                assert cat_family in categories_dict.keys(), f"There are no samples for category '{cat_family}'. Consider passing a list of samples or use one of the available categories: {categories_dict.keys()}"
                categories = rng.choice(categories_dict[cat_family], length, replace = False, shuffle = True)
            df[col_name] = rng.choice(categories, size, shuffle = True)
        elif col == "d":
            df[col_name] = rng.choice(pd.date_range(start, end), size)
    return df       


def feature_frecuency_binary_class(df,target,porcentaje_signifi=4,
                                   diferen_minima=0.15,show_table=1,
                                   output_tables=0,inc_num_cols=1,qcut_len = 4,
                                   signi_pval=0.01,kmean_cut=0,date_feats=0,
                                   dicc_get_autocorr={},dicc_feats_by_groups={}):
    """Funcion creada para extraer los features que mejor discriminen el target en base 
    a la frecuencia de sus categorias
    
    porcentaje_signifi = 4 #Porcentaje de significancia estadistica
    diferen_minima = 0.15 #Diferencia minima para discriminar
    signi_pval = 0.01 # El valor minimo de pvalue que debe tener para decidir si son distribuciones diferentes, mientras mas alto mas diferentes son
    """
    
    """
    Esta funcion trabajara en base a todas las variables categoricas del dataframe por lo que es importante
    Asegurarse de que cada variable categorica que el dataframe contenga es importante
    """
    import numpy as np
    import pandas as pd 
    from tqdm.notebook import trange, tqdm
    import sys
    from scipy.stats import ks_2samp
    import mis_funciones as mf
    import datetime
    import warnings
    warnings.filterwarnings('ignore')
    
    
    
    

    ### Creamos algunas variables nuevas para las variables de fecha
    date_cols = []
    if date_feats:
        date_cols = [c for c in df.columns if "fecha" in c.lower()]

        for d in date_cols:
            #print(d)
            if d != "fecha_actual":
                df["trimes_"+d] = pd.to_datetime(df[d]).dt.quarter

                x = datetime.datetime.now()
                df["fecha_actual"] = x.strftime("%d-%m-%Y")
                df["diff_act_"+d] = (pd.to_datetime(df["fecha_actual"]) - pd.to_datetime(df[d])).dt.days


    ## Para convertir en cat las columnas con menos de 6 valores por si acaso
    bin_cols =  df.nunique()[df.nunique() <= 6].index
    for b in bin_cols:
        if b != target:
            df[b] = df[b].astype(str)

    ## Para convertir a target en bool
    if df[target].nunique()>2:
        tipo_target = 2
    elif df[target].nunique()==2:
        df[target].astype(bool)
        tipo_target = 1
    else:
        print('El target no es ni booleano ni numerico')
    #df[target] = df[target].astype(bool)
    
    ## Tomamos las columnas cat
    numeric_cols = set(df._get_numeric_data().columns)
    all_cols = set(df.columns) 
    categorical_cols = all_cols - numeric_cols
    categorical_cols = categorical_cols - set(date_cols)
    categorical_cols = categorical_cols - set([target])
    categorical_cols = list(categorical_cols)
    

    ### qcutear columnas numericas
    qcut_cols = []
    if inc_num_cols:
        mapeo = [df[d].nunique()>2 for d in df._get_numeric_data().columns] #Obtener cols con nunique > 2
        num_for_cat_cols = set(df._get_numeric_data().columns[mapeo]) #Obtener cols con nunique > 2
        for num_col in tqdm(num_for_cat_cols):

            ### Solo vamos a qcutear si las distribuciones entre las clases son lo suficientemente diferentes
            kolmo_dist_pval = 0
            dist1 = df[df[target]==0][num_col]
            dist2 = df[df[target]==1][num_col]
            if len(dist1) > 0 and len(dist2) > 0:
                kolmo_dist = ks_2samp(dist1, dist2)
                kolmo_dist_pval = kolmo_dist[1]

            if kolmo_dist_pval <=signi_pval:
                    
                qcut_col_name = ""
                for q in reversed(range(2,qcut_len)): ## Si el qcut da error haremos uno menor
                    
                    try:
                        IQR = df[num_col].quantile(0.75) - df[num_col].quantile(0.25)
                        lower_std = df[num_col].quantile(0.25) - (3.5*IQR)
                        upper_std = df[num_col].quantile(0.75) + (3.5*IQR)

                        label_cuts = pd.DataFrame([pd.qcut(df[(df[num_col]<=upper_std) & (df[num_col]>=lower_std)][num_col],q).cat.categories.right],duplicates='drop')
                        #label_cuts = list(label_cuts.iloc[0,:]) + [df[num_col].max()]
                        label_cuts = list(label_cuts.iloc[0,:]) + [df[num_col].max()]

                        qcut_col_name = str(num_col)+"_qcut"
                        qcut_col = pd.cut(df[num_col],label_cuts)
                        df[qcut_col_name] = qcut_col
                        df[qcut_col_name] = df[qcut_col_name].astype(str).str.replace(" ","").str.replace("  ","").str.replace("\t","")
                    except:
                        qcut_col_name = ""
                        #print("No se logro hacer qcut de ",q," con ",num_col)
                        continue
                        
                if qcut_col_name != "": #Sino logra qcutear no lo agregara  
                    qcut_cols.append(qcut_col_name)

    
    ## k_means cutear las variables numericas
    kcut_cols = []
    if kmean_cut:

        ## Formula calcular un buen df_mini_size

        create_mini_df = 0
        if len(df) >30000:
            create_mini_df = 1
        mini_df_size = np.round((30000/len(df)),2)
        if mini_df_size > 0.3:
            mini_df_size = 0.3

        mapeo = [df[d].nunique()>2 for d in df._get_numeric_data().columns] #Obtener cols con nunique > 2
        num_for_cat_cols = set(df._get_numeric_data().columns[mapeo]) #Obtener cols con nunique > 2
        #num_for_cat_cols = k_means_cols
        for num_col in tqdm(num_for_cat_cols):

            ### Solo vamos a kcutear si las distribuciones entre las clases son lo suficientemente diferentes
            kolmo_dist_pval = 0
            dist1 = df[df[target]==0][num_col]
            dist2 = df[df[target]==1][num_col]
            if len(dist1) > 0 and len(dist2) > 0:
                kolmo_dist = ks_2samp(dist1, dist2)
                kolmo_dist_pval = kolmo_dist[1]

            if kolmo_dist_pval <=signi_pval:
                try:
                    kcut_col_name = str(num_col)+"_kcut"
                    df[kcut_col_name] = mf.kmeans_bins(df,num_col,multiplo_IQR=3.5,create_mini_df=create_mini_df,mini_df_size=mini_df_size,mini_df_signi=0.001)
                    df[kcut_col_name] = df[kcut_col_name].astype(str).str.replace(" ","").str.replace("  ","").str.replace("\t","")
                    kcut_cols.append(kcut_col_name)
                except:
                    continue


    df_cross_tab_list = []
    for cat_col in tqdm(categorical_cols + qcut_cols + kcut_cols):
        
        #print(cat_col)
        feature = cat_col
        df_tab = df.copy()
        df_tab[feature] = df_tab[feature].astype(str)
        #print(len(df_tab[feature]))
        df_tab = df_tab[[target,feature]].dropna()
        
        if (len(df_tab)>0) and (df_tab[target].nunique() > 1):

                ### Calculamos las proporciones para los casos de cat y las medias para los casos numericos
                df_type_crossed_totals = mf.get_df_tab(df_tab,target,feature,tipo_target)
                #display(df_type_crossed_totals)

                features_encontradas = []
                for col in df_type_crossed_totals.columns:
                    total_col = df_type_crossed_totals.loc["Total_abs",col]
                    total_df = df_type_crossed_totals.loc["Total_abs","Prior_proportion"]
                    total_col_propor = (total_col/total_df)*100
                    total_false = df_type_crossed_totals.loc[False,"Prior_proportion"]
                    total_true = df_type_crossed_totals.loc[True,"Prior_proportion"]

                    if total_col >= 30 and total_col_propor >= porcentaje_signifi: #Si la muestra es significatica
                        if (df_type_crossed_totals.loc[False,col] == 1) | (df_type_crossed_totals.loc[False,col] == 0): #Si hay una prob de un 100%
                            features_encontradas.append(col)

                        if abs(df_type_crossed_totals.loc[False,col] - total_false) >= diferen_minima: #Si hay una diferencia minima
                            features_encontradas.append(col)

                ## Guardar la tabla si encontro algun feat interesante
                if len(features_encontradas) > 0:
                    df_cross_tab_list.append(df_type_crossed_totals[features_encontradas+["Prior_proportion"]]) # Para splo mostrar las columnas significativas

                #print(features_encontradas)
                for feat in features_encontradas:
                    feat = str(feat)
                    df[feature+"_"+feat] = np.where((df[feature]==feat), 1,0)

        ### Calculamos las proporciones prior para los casos de cat y las medias prior para los casos numericos
        tab_deafult = mf.calc_priors_tab(df,target,tipo_target)
        df_cross_tab_list.insert(0,tab_deafult)
    if output_tables:
        return (df,df_cross_tab_list)
    else:
        return df
    
    
def plot_int_tabs(tabs):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    for t in tabs[1:]: # La primera tab es la deafult
        num_c = len(t.columns)
        if num_c < 15:
            
            tabla = t 

            ax =tabla.T.iloc[:,:-2].plot(kind='bar',
                        stacked=False,
                        title=f'Proporcion y cantidad de registros de {t.index.name} segun el target', figsize=(8, 6))


            # set individual bar lables using above list
            c1 = (tabla.T.iloc[:,:-2] * (tabla.loc["Total_abs"].values).reshape(len(tabla.T),1))[False]
            c2 = (tabla.T.iloc[:,:-2] * (tabla.loc["Total_abs"].values).reshape(len(tabla.T),1))[True]
            abs_patches = pd.concat([c1,c2],axis=0).values
            for i,a in enumerate(ax.patches):
                # get_x pulls left or right; get_height pushes up or down
                ax.text(a.get_x()+.03, a.get_height()+.03, 
                        int(abs_patches[i]), fontsize=15,
                            color='black')


            plt.legend(loc='upper right', frameon=False)
            plt.ylim(0, 1.1)
            plt.yticks(np.arange(0, 1, step=0.1))
            plt.grid(True)

            
        else:
            print(t.index.name," No se puede graficar porque tiene ",num_c," columnas")
            display(t)
            

def get_nc(n,*args):
    import scipy.stats as st
    import numpy as np
    
    p = args[0]
    q = 1-p
    error = args[1]

    z_score  = error/ np.sqrt( (p*q)/n )
    nc = 1 - st.norm.pdf(z_score)
    return nc


def get_nc_de_media(n,*args):
    import scipy.stats as st
    import numpy as np
    
    x_mean = args[0]
    x_std = args[1]

    se = x_std/(np.sqrt(n))
    e = x_mean*0.2
    nc = e/(se)

    return nc


def get_ic(x,*args):
    import numpy as np
    
    p = args[0]
    q = 1-p
    error = args[1]

    ic = f"De {np.round(x-error,2)} a {np.round(x+error,2)}" 
    return ic

def get_diff_table(tabs,error=0,tipo_target=1):
    import pandas as pd
    import numpy as np
    tabst = tabs
    tabst = [t.T for t in tabst]
    names = [t.index.name for t in tabs]
    for i,t in enumerate(tabst):
        #print(names)
        t.index = [str(names[i])+" | "+str(ti) for ti in t.index]

    tabst = [t.T for t in tabst]

    tabs2 = [t.iloc[:,:-1] for t in tabst[1:]]
    tabs2.append(tabs[0])
    tabs3 = pd.concat(tabs2,axis=1)
    tabs3 = tabs3.T
    tabs3["diff"] = tabs3[True] - tabs3[False]
    tabs3["diff_abs"] = np.abs(tabs3[False] - tabs3[True])
    #filt = tabs3["diff_abs"] <1
    tabs_final = tabs3.drop("Prior_proportion",axis=0)
    t_prior = pd.DataFrame(tabs3.loc["Prior_proportion"]).T
    t_prior = pd.concat([tabs_final,t_prior])
    
    t_prior["diff_prior_abs"] = np.abs(t_prior[True] - t_prior.loc["Prior_proportion"][True])
    t_prior["diff_prior"] =   t_prior[True] - t_prior.loc["Prior_proportion"][True]
    t_prior["Prior_proportion_True"] = t_prior.loc["Prior_proportion",True]
    t_prior["Prior_proportion_False"] = t_prior.loc["Prior_proportion",False]
    
    if tipo_target==2:
        p = t_prior["Prior_proportion_True"].iloc[0] ## Media de la poblacion
        poblacion_std = t_prior["Prior_proportion_False"].iloc[0] ## En la version para conitnuas el std se guardo en Prior_proportion_False
        if error==0: #Si el error no fue definido
            error=p*0.1 # Que sea entre un -20% y 20% de la media del prior True
        t_prior["nivel_confianza"] = t_prior["Total_abs"].apply(get_nc_de_media,args=[poblacion_std,error])
        t_prior["intertvalo_confianza_True"] = t_prior[True].apply(get_ic,args=[p,error])
    elif tipo_target==1:
        p = t_prior["Prior_proportion_True"].iloc[0]
        if error==0: #Si el error no fue definido
            error=p*0.1 # Que sea entre un -20% y 20% de la proporcion del prior True
        t_prior["nivel_confianza"] = t_prior["Total_abs"].apply(get_nc,args=[p,error])
        t_prior["intertvalo_confianza_True"] = t_prior[True].apply(get_ic,args=[p,error])
    

        
    return t_prior.sort_values("diff_prior_abs",ascending=False)

def get_combs_tab_diff(df,tab_diff,target,error=0,nivel_confianza=0.85):
    import pandas as pd
    import numpy as np
    from tqdm.notebook import trange, tqdm
    from itertools import combinations
    import itertools
    
    prior_tab = pd.DataFrame(tab_diff.loc['Prior_proportion']).T
    tab_diff2 = tab_diff.drop('Prior_proportion',axis=0)
    tab_diff2["index_1"] = [l[0].replace(" ","") for l in tab_diff2.index.str.split("|")]
    tab_diff2["index_2"] = [l[1][1:] for l in tab_diff2.index.str.split("|")]

    #
    ## Combinamos todas los valores de las variables interesantes
    comb_bin_cols = []
    comb_cols = set([v for v in tab_diff2["index_1"].values]) 
    
    
    for r in tqdm(range(len(comb_cols)+1)):

        comb = combinations(comb_cols,r)
        for i in list(comb):
            comb_bin_cols.append(list(i))


    all_combs_list = []
    for c in tqdm(comb_bin_cols[len(comb_cols)+1:]):

        list_cols = c
        lis_lis = []
        for l in list_cols:
            lis_lis.append(list(tab_diff2[tab_diff2["index_1"] == l].index.unique()))

        a = lis_lis
        all_combs = list(itertools.product(*a))
        all_combs_list.append(all_combs)


    all_combs_list2 = []
    for a in all_combs_list:
        all_combs_list2 += a

    
    
    ## Creamos el dataset segun las combinaciones
    dicc = {}
    cols_len = len(tab_diff.index[:-1])

    for comb in tqdm(all_combs_list2):
        #print(comb)
        cols = [c.split(" | ")[0] for c in comb]
        vals = [c.split(" | ")[1] for c in comb]
        df_results = df[(df[cols]==vals).sum(axis=1)==len(vals)]
        if len(df_results) > 0:
            df_results[target] = df_results[target].astype(bool)
            df_results = df_results.agg(target_pr = (target,"mean"),cant = (target,"count") )
            dicc[str(comb)] = [df_results.loc["target_pr"].values[0],df_results.loc["cant"].values[0]]

    
    if len(dicc)>0:
        ## Creamos las mismas variables que ya estaban de tab_diff
        #tab_diff = pd.concat([tab_diff,prior_tab],axis=0)
        #return tab_diff
        df_combs = pd.DataFrame(dicc).T.sort_values(0,ascending=False)
        df_combs.columns = [True,"Total_abs"]
        df_combs[False] = df_combs[True] == False
        df_combs.head()
        df_combs["Total_rel"] = 1.0
        df_combs["diff"] = df_combs[True] - df_combs[False]
        df_combs["diff_abs"] = np.abs(df_combs[False] - df_combs[True])
        df_combs["diff_prior_abs"] = np.abs(df_combs[True] - tab_diff.loc["Prior_proportion"][True])
        df_combs["diff_prior"] =   df_combs[True] - tab_diff.loc["Prior_proportion"][True]
        df_combs["Prior_proportion_True"] = tab_diff.loc["Prior_proportion",True]

        p = tab_diff["Prior_proportion_True"].iloc[0]
        if error==0: #Si el error no fue definido
            error=p*0.2 # Que sea entre un -20% y 20% de la proporcion del prior True

        df_combs["nivel_confianza"] = df_combs["Total_abs"].apply(get_nc,args=[p,error])
        df_combs["intertvalo_confianza_True"] = df_combs[True].apply(get_ic,args=[p,error])

        return df_combs[df_combs["nivel_confianza"]> nivel_confianza]



def get_bestcombs_tabdiff(df_combs,en_base_al,nivel_confianza=0.85):

    """
    Para obtener las combinaciones que sean significativas, comparandolas con las probabilidades sin los grupos
    """
    df_combs["cant_feats"] = [len(i.split(" | "))-1 for i in df_combs.index]
    df_combs = df_combs[df_combs["nivel_confianza"]> nivel_confianza].sort_values("cant_feats",ascending=True)

    worst_index = []
    for i in df_combs.index[:]:
        #print(i)
        for i2 in df_combs.index:
            a = i.replace("'","").replace(")","").replace("(","").replace("]","").replace("[","")
            b = i2.replace("'","").replace(")","").replace("(","").replace("]","").replace("[","")
            if (a in b) & (a != b) :
                
                if en_base_al == True:
                    bool_res = ( ( (df_combs.loc[i][True] >= df_combs.loc[i2]["Prior_proportion_True"]) & 
                            (df_combs.loc[i][True] >= df_combs.loc[i2][True]) )  )
                
                elif en_base_al == False:
                    bool_res = ( ( (df_combs.loc[i][True] <= df_combs.loc[i2]["Prior_proportion_True"]) & 
                            (df_combs.loc[i][True] <= df_combs.loc[i2][True]) )  )
                
                if bool_res:
                    #print(i," vs",i2,"\n")
                    worst_index.append(i2)

    best_index = set(df_combs.index) -  set(worst_index)

    return df_combs.loc[best_index].drop_duplicates().sort_values([True,"cant_feats"],ascending=[False,True])


def feature_statistics_binary_class(df,target,num_cols,diferen_minima=0.4):
    import pandas as pd
    import numpy as np
    
    """Funcion creada para extraer los features que mejor discriminen el target en base 
    a los estadisticos (media, mediana, min, max, etc.)
    
    diferen_minima = 0.4 #Diferencia minima para discriminar
    """
    
    """
    Esta funcion trabajara en base a todas las variables continuas del dataframe que se le pasen
    """
    
      #### SUM
    res_class0 = df[df[target]==0][num_cols].sum(axis=1).median()
    res_class1 = df[df[target]==1][num_cols].sum(axis=1).median()
    dif_class = np.abs(res_class0 - res_class1)

    if (dif_class+res_class0) >= (res_class0*(diferen_minima+1)): #Si la diferencia es de al menos [diferen_minima]
        df["sum_num_cols"] = df[num_cols].sum(axis=1)
        
    #### Mean
    res_class0 = df[df[target]==0][num_cols].mean(axis=1).median()
    res_class1 = df[df[target]==1][num_cols].mean(axis=1).median()
    dif_class = np.abs(res_class0 - res_class1)

    if (dif_class+res_class0) >= (res_class0*(diferen_minima+1)): #Si la diferencia es de al menos [diferen_minima]
        df["mean_num_cols"] = df[num_cols].mean(axis=1)


    #### STD
    res_class0 = df[df[target]==0][num_cols].std(axis=1).median()
    res_class1 = df[df[target]==1][num_cols].std(axis=1).median()
    dif_class = np.abs(res_class0 - res_class1)

    if (dif_class+res_class0) >= (res_class0*(diferen_minima+1)): #Si la diferencia es de al menos [diferen_minima]
        df["std_num_cols"] = df[num_cols].std(axis=1)


    #### SKEW
    res_class0 = df[df[target]==0][num_cols].skew(axis=1).median()
    res_class1 = df[df[target]==1][num_cols].skew(axis=1).median()
    dif_class = np.abs(res_class0 - res_class1)

    if (dif_class+res_class0) >= (res_class0*(diferen_minima+1)): #Si la diferencia es de al menos [diferen_minima]
        df["skew_num_cols"] = df[num_cols].skew(axis=1)


    #### kurt
    res_class0 = df[df[target]==0][num_cols].kurt(axis=1).median()
    res_class1 = df[df[target]==1][num_cols].kurt(axis=1).median()
    dif_class = np.abs(res_class0 - res_class1)

    if (dif_class+res_class0) >= (res_class0*(diferen_minima+1)): #Si la diferencia es de al menos [diferen_minima]
        df["kurt_num_cols"] = df[num_cols].kurt(axis=1)


    #### min
    res_class0 = df[df[target]==0][num_cols].min(axis=1).median()
    res_class1 = df[df[target]==1][num_cols].min(axis=1).median()
    dif_class = np.abs(res_class0 - res_class1)

    if (dif_class+res_class0) >= (res_class0*(diferen_minima+1)): #Si la diferencia es de al menos [diferen_minima]
        df["min_num_cols"] = df[num_cols].min(axis=1)


    #### max
    res_class0 = df[df[target]==0][num_cols].max(axis=1).median()
    res_class1 = df[df[target]==1][num_cols].max(axis=1).median()
    dif_class = np.abs(res_class0 - res_class1)

    if (dif_class+res_class0) >= (res_class0*(diferen_minima+1)): #Si la diferencia es de al menos [diferen_maxima]
        df["max_num_cols"] = df[num_cols].max(axis=1)
        
        
    return df



def transform_skew_cols(df):

    skew_cols = df.iloc[:,:-1].skew()[(df.iloc[:,:-1].skew()>1) | (df.iloc[:,:-1].skew()<-1)]
    print("10 cols antes de la transformacion")
    display(skew_cols[:10])

    skew_cols_names = skew_cols.index
    df[skew_cols_names] = df[skew_cols_names].apply(lambda x: x**(1/3))

    skew_cols = df.skew()[(df.skew()>1) | (df.skew()<-1)]
    print("10 cols luego de la transformacion")
    display(skew_cols[:10])
    
    return df



def normalize_cols(df):

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler() 

    for col in df.select_dtypes(include='number').columns:
        try:
            df[col] = scaler.fit_transform(df[[col]])
        except:
            print("Error con ",col)

    return df



def detect_multcoli(df):
    from sklearn.linear_model import LinearRegression

    df_corrscores = df.corr()

    mapeo = df_corrscores.unstack().apply(lambda x: x >= 0.75 and x < 1)
    correlaciones_corr = df_corrscores.unstack()[mapeo]
    #display(correlaciones_corr.head())

    for correla in correlaciones_corr.index:
        X,y = df[[correla[0]]], df[[correla[1]]]
        r_squared = LinearRegression().fit(X, y).score(X, y)

        # calculate VIF
        vif = 1/(1 - r_squared)
        if vif > 4:
            print(correla[0]," con ",correla[1]," VIF = ",vif)
            



def get_proportion_by_comb_feat(df,comb_cols,target,min_sample):
    """
    Funcion para obtener asignarle el grupo mas correspondiente a cada registro
    """
    from tqdm.notebook import trange, tqdm
    from itertools import combinations

    ### Hacemos todas las combinaciones
    comb_bin_cols = []
    for r in range(len(comb_cols)+1):

        comb = combinations(comb_cols,r)
        for i in list(comb):
            comb_bin_cols.append(list(i))

   
    ### Creamos grupos de datasets de cada una de las combinaciones
    grouped_dfs = list()
    for comb_cols in tqdm(comb_bin_cols[1:]):
        binning_cols_for = list(comb_cols)
        grouped_df = df.groupby(binning_cols_for).agg(target_pr = (target,"mean")
                                                      ,cant = (target,"count") )
        grouped_dfs.append(grouped_df.reset_index())

        df_concatenated = pd.concat(grouped_dfs)
    
    
    ### Organizamos las columnas
    set_cols = list(set(df_concatenated.columns) - set(["cant","target_pr"]))
    set_cols.extend(["cant","target_pr"])
    df_concatenated = df_concatenated[set_cols]
    
    ##Min sample
    df_concatenated = df_concatenated[df_concatenated.cant>min_sample]
       
    return df_concatenated


def get_proportions(df,feats,df_probs):
    
    ### Calculamos la columna all
    df["all"] = pd.Series()

    for i,c in enumerate(feats):
        if i ==0:
            t = df[c].astype(str) + "|"
            df["all"] = t
        elif i ==len(feats)-1:
            t = df[c].astype(str)
            df["all"] = df["all"].astype(str) + t
        else:
            t = df[c].astype(str) + "|"
            df["all"] = df["all"].astype(str) + t
    
    ### Calculamos la categoria donde pertenece
    def get_category(x,*args):

        x = np.array(x.split("|"))
        return (x == y).sum(axis=1).idxmax()

    y = df_probs[feats]
    idx_probs = df["all"].apply(get_category,args=[y])
    
    ### Le asignamos la proporcion de la categoria a cada fila
    df["prob"] = df_probs.loc[idx_probs].target_pr.values
    
    return df



def hacer_analisis_express(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = set(df.columns) - set(num_cols)

    ### Shape
    print("Shape ", df.shape)
    
    ### Ver elementos unicos de las variables
    print("\nVer elementos unicos y nulos\n")
    for c in df.columns:
        print(c,df[c].nunique(),sum(df[c].isnull()))
    
    ### Frencuencia de las variables categoricas   
    print("\nFrencuencia de las variables categoricas \n")
    cols_no_impr = []
    for column in cat_cols:
        try:
            print(column, df[column].notna().sum()," registros de ",len(df))
            if df[column].nunique() < 10:
                freq = df[column].value_counts()
                freq = freq.sort_values(ascending=False)

                plt.figure(figsize=(25,5))
                sns.barplot(freq.index,freq.values)
                plt.show()
            else:
                freq = df[column].value_counts()
                freq = freq.sort_values(ascending=False)
                display(freq)
        except Exception as e:
            print("Hubo un error con ",column,e,"\n")
      
    ### Histograma de las variables numericas
    print("\nHistograma de las variables numericas\n")
    cols_no_impr = []
    for column in num_cols:
        print(column, df[column].notna().sum()," registros de ",len(df))
        plt.figure(figsize=(25,5))
        sns.histplot(df[column])
        plt.show()



def fill_na_mean_mode(df,median=0):
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = set(df.columns) - set(num_cols)
    
    for c in num_cols:
        nans = df[c].isnull().sum()
        if nans > 0:
            if median:
                new_val = df[c].median()
                print(nans," encontrados en ",c," y fueron reemplazados con ",new_val)
                df[c] = df[c].fillna(new_val)
            else:
                new_val = df[c].mean()
                print(nans," encontrados en ",c," y fueron reemplazados con ",new_val)
                df[c] = df[c].fillna(new_val)
            
    for c in cat_cols:
        nans = df[c].isnull().sum()
        if nans > 0:
            new_val = df[c].mode()
            print(nans," encontrados en ",c," y fueron reemplazados con ",new_val)
            df[c] = df[c].fillna(new_val)
        
    return df
  

def check_multicolinearity(df):
    from sklearn.linear_model import LinearRegression
    df_corrscores = df.corr()

    mapeo = df_corrscores.unstack().apply(lambda x: x >= 0.75 and x < 1)
    correlaciones_corr = df_corrscores.unstack()[mapeo]
    #display(correlaciones_corr.head())
        
    for correla in correlaciones_corr.index:
        X,y = df[[correla[0]]], df[[correla[1]]]
        r_squared = LinearRegression().fit(X, y).score(X, y)

        # calculate VIF
        vif = 1/(1 - r_squared)
        if vif > 4:
            print(correla[0]," con ",correla[1]," VIF = ",vif)


def get_obs(df,cols_gr,target,umbral=0.02):
    import numpy as np
    import pandas as pd
    df = df.groupby(cols_gr)[target].count()/df.groupby(cols_gr)[target].count().sum()
    df.sort_values(ascending=False,inplace=True)
    ind = []
    vals = []
    for i in range(len(df)):
        if df.iloc[i] >= umbral:
            ind.append(df.index[i])
            vals.append(np.round(df.iloc[i]*100,2))
            m = f"- Un {np.round(df.iloc[i]*100,2)}% {df.index[i]}"
            print(m)
    return pd.DataFrame({"Name":ind,"Value":vals})


def compare_var_dist(df,col1,col2,rel=0):
    import numpy as np
    import pandas as pd
    crosst = pd.crosstab(index = df[col1],columns = df[col2],margins = True)
    if rel:
        crosst = pd.crosstab(index = df[col1],columns = df[col2],margins = True)
        for c in crosst.columns[:-1]:
            crosst[c] = crosst[c]/crosst["All"]
        crosst['All_rel'] = crosst['All']/crosst.loc["All","All"]
        return crosst
    else:
        return crosst


def compare_var_dist_target(df,col1,col2,target,rel=0):
    import numpy as np
    import pandas as pd
    crosst = pd.crosstab(index = df[target], #Hacemos la tabla de frecuencias
                                columns = [df[col1],df[col2]]
                                ,margins = True) 
    
    if rel:
        return crosst/crosst.loc["All"]
    else:
        return crosst


def check_influencia_var(df,tab_diff,col_target,val_target,
                         porcentaje_signifi=0.01,diferen_minima=0.03):   
    
    import pandas as pd
    import numpy as np
    import mis_funciones as mf
    
    dfs = []
    for i in tab_diff.index[:-1]:
        col = i.split(" | ")[0]
        if col != col_target:
            #print(col)
            try:
                df_influ = mf.compare_var_dist(df,col,col_target,rel=0)[[val_target,"All"]]
                df_influ["Total_"+val_target] = len(df[df[col_target]==val_target][col_target])
                df_influ[val_target+"_de_"+"Total_"+val_target] = df_influ[val_target]/df_influ["Total_"+val_target]
                df_influ["Total_nans"] = df[col].isnull().sum()
                df_influ["prop"] = df_influ[val_target]/df_influ["All"]
                df_influ["Lo_normal"] = df_influ.loc["All"]["prop"]
                df_influ["diff_all"] = df_influ["prop"] - df_influ["Lo_normal"]
                df_influ["diff_all_abs"] = np.abs(df_influ["prop"] - df_influ["Lo_normal"])
                df_influ["Col"] = col
                #display(df_influ)
                dfs.append(df_influ)
            except Exception as e:
                print("Problemas con la col ",col," \n",e)
    
    if len(dfs) > 0:
        df_influ_final = pd.concat(dfs).sort_values("diff_all_abs",ascending=False)


        cant_minima = len(df)*porcentaje_signifi
        relacion_es = tab_diff.loc[f"{col_target} | {val_target}"]["diff_prior"] > 0
        filt = (df_influ_final["All"]>=cant_minima) & (df_influ_final["diff_all"]>=diferen_minima)

        return df_influ_final[filt].drop_duplicates()


def resumir_variables(df,tab_diff_filt,targ,diff=0.05):
    var_target = targ+" |"

    df[targ+"_alto_true"] = 0
    df[targ+"_bajo_true"] = 0

    for t in tab_diff_filt.index:
        if var_target in t:
            val = t.split(" | ")[1]
            val_true = tab_diff_filt.loc[t, True]
            prior_true = tab_diff_filt.loc[t,"Prior_proportion_True"]
            if (val_true - prior_true) > diff:
                #print(t, val_true)
                df[targ+"_alto_true"] = np.where(df[targ]==val,1,df[targ+"_alto_true"])
            elif (val_true - prior_true) < diff:
                #print(t, val_true)
                df[targ+"_bajo_true"] = np.where(df[targ]==val,1,df[targ+"_bajo_true"])
                
                
    return df




def kmeans_bins(df,col,multiplo_IQR=3.5,create_mini_df=0,mini_df_size=0.3,mini_df_signi=0.001,print_it=0):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import KBinsDiscretizer
    import numpy as np
    import pandas as pd
    import mis_funciones as mf
    
    #### Creamos df_vdd para guardar el original
    df_vdd = df.copy()
    
    ### Vamos eliminar los outliers aqui
    IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
    lower_std = df[col].quantile(0.25) - (multiplo_IQR*IQR)
    upper_std = df[col].quantile(0.75) + (multiplo_IQR*IQR)
    max_label = df[col].max()
    
    if create_mini_df:
        df = mf.extract_mini_dataset(df[[col]],porcentaje=mini_df_size,significancia=mini_df_signi,print_it=print_it)

    ## Crearemos los cuts sin los outliers y luego agregaremos el max_label
    df = df[(df[col]<=upper_std) & (df[col]>=lower_std)][[col]].dropna()
    X  = df[[col]]

    df_scores = pd.DataFrame()
    scores = [] 
    idx = [] 
    for n in range(2,10):
        #print(n)
        #print(n)
        kmeans = KMeans(n_clusters=n, random_state=0,max_iter=70).fit(X)
        #display(kmeans.labels_)
        score = silhouette_score(X, kmeans.labels_)
        idx.append(n)
        scores.append(score)
        if print_it:
            print('Con ',n,"Clusters, Score: ",score)

    df_scores = pd.DataFrame()
    df_scores['index'] = idx
    df_scores["scores"] = scores
    df_scores["scores_5"] = 2.5 *np.round(df_scores["scores"]/2.5,2)

    df_scores["diff_scores"] = df_scores["scores"].diff()
    df_scores = df_scores[df_scores["diff_scores"].notna()]
    df_scores = df_scores.sort_values(["scores_5","diff_scores"],ascending=[False,False])
    best_n = df_scores["index"].iloc[0]
    #display(df_scores)
    #print(best_n)

    est = KBinsDiscretizer(n_bins=best_n, encode='ordinal', strategy='kmeans')
    est.fit(X)
    
    cuts = est.bin_edges_

    cuts = list(cuts[0]) + [max_label]
    
    df_vdd[str(col)+"_kcut"] = pd.cut(df_vdd[col],cuts,duplicates='drop')
    
    return df_vdd[str(col)+"_kcut"]



def find_bad_cuts(tab_diff_filt,df,target):
    import numpy as np
    import pandas as pd
    import mis_funciones as mf
    import warnings
    warnings.filterwarnings('ignore')
    cols_rep = []
    vals_rep = []
    prior_tab = tab_diff_filt.iloc[-1,:]
    for i in tab_diff_filt.index[:-1]:
        col = i.split(" | ")[0]
        val = i.split(" | ")[1]
        if ("_kcut" in col) or ("_qcut" in col):
            #print(col,val)
            cols_rep.append(col)
            vals_rep.append(val)


    df_cols_rep = pd.DataFrame({"cols":cols_rep,"vals":vals_rep})

    df_cols_rep_freq = df_cols_rep["cols"].value_counts()

    #Ya tenemos las columnas que se repiten
    cols = df_cols_rep_freq[df_cols_rep_freq>1].index

    for c in cols:
        #print(c)
        df_comp = mf.compare_var_dist(df,c,target,rel=1).iloc[:-1,:].reset_index()

        df_comp["ini_interv"] = df_comp[c].apply(lambda x: str(x).split(", ")[0].replace("(","")).astype(float)
        df_comp["ini_interv_round"] = np.round(df_comp["ini_interv"])
        df_comp = df_comp.sort_values("ini_interv")
        df_comp["True_diff"] = df_comp[True].diff()
        df_comp["True_diff2"] = df_comp[True].diff(-1)
        df_comp = df_comp.reset_index()
        #display(df_comp)

        filt = df_cols_rep["cols"]==c
        for v in df_cols_rep[filt]["vals"].values:
            #print(v)
            v_ini = np.round(float(v.split(", ")[0].replace("(","")))
            filt2 = ((np.abs(df_comp["ini_interv_round"] - v_ini) <= 0.001))
            #display(df_comp)
            if ( (len(df_comp[filt2])>0) and (np.abs(df_comp[filt2]["True_diff"]).values[0] <= 0.005)):
                #continue
                print("True_diff ",c,v)
                df_founded = pd.concat([df_comp.loc[df_comp[filt2].index-1],df_comp[filt2]])
                ini_interv = str(df_founded[c].iloc[0]).split(", ")[0].replace("(","").replace("(","")
                final_interv = str(df_founded[c].iloc[1]).split(", ")[1].replace("]","").replace("(","")

                if ini_interv != final_interv:  ## Agregamos las filas merged
                    tab_diff_filt = tab_diff_filt.rename(index={f"{c} | {v}":f"{c} | ({ini_interv}, {final_interv}]"})
                else:  ## Eliminamos las filas repetidas
                    #tab_diff_filt.drop(f"{c} | {v}",axis=0,inplace=True)
                    continue
                
            if ((len(df_comp[filt2])>0) and (np.abs(df_comp[filt2]["True_diff2"]).values[0] <= 0.005) ):
                #continue
                print("True_diff2 ",c,v)
                df_founded = pd.concat([df_comp.loc[df_comp[filt2].index+1],df_comp[filt2]])
                ini_interv = str(df_founded[c].iloc[0]).split(", ")[0].replace("(","").replace("(","")
                final_interv = str(df_founded[c].iloc[1]).split(", ")[1].replace("]","").replace("(","")
                
                if ini_interv != final_interv:  ## Agregamos las filas merged
                    tab_diff_filt = tab_diff_filt.rename(index={f"{c} | {v}":f"{c} | ({ini_interv}, {final_interv}]"})
                else:  ## Eliminamos las filas repetidas
                    #tab_diff_filt.drop(f"{c} | {v}",axis=0,inplace=True)
                    continue

    tab_diff_filt = pd.concat([tab_diff_filt,prior_tab],axis=0)

    return tab_diff_filt



def stratified_sampling_by_feats(df,
                                 col_list,
                                 col_val_dicc,
                                 prop_dicc,
                                 last_sample_size,
                                 organizar_by_mincol=1,
                                 margen_error=0.04,
                                iteraciones_error=100):

    import pandas as pd
    import numpy as np
    import mis_funciones as mf

    def one_stratified_sampling_by_feats(df,col_list,col_val_dicc,prop_dicc,last_sample_size,organizar_by_mincol=1):

        ## Organizar en base a los que tengan el mayor min sample (Osea el min(len(de cada level)) mas alto)
        if organizar_by_mincol:
            dfs_temp = []
            for i,col in enumerate(col_list,1):
                df_temp = pd.DataFrame()
                for v in col_val_dicc[col]:
                    df_temp['col'] = col
                    df_temp['val'] = v
                    df_temp['len_val'] = len(df[df[col]==v])
                dfs_temp.append(df_temp)
            df_temp_concat = pd.concat(dfs_temp,axis=0)
            df_temp_concat = df_temp_concat.groupby('col').len_val.min().sort_values(ascending=False).reset_index()
            col_list = df_temp_concat['col']


        for i,col in enumerate(col_list,1):
            #print(col," comienza")
            idx_col_val_list = []
            for v in col_val_dicc[col]:
                idx_col_val_list.append(df[df[col]==v].index)

            ### Cada level debe ser del tamanio de su level mas pequenio
            proportions = prop_dicc[col]
            sizes = list(map(len, idx_col_val_list))
            sample_size = np.min(sizes)*len(sizes) ## Este es el maximo tamaño que cada level puede tener siendo uniformes

            if i==len(col_list):
                sample_size = last_sample_size

            #print(col," sample size ",sizes)

            sample_arrays = []
            for i2,idx_col_val in enumerate(idx_col_val_list):
                sample_arrays.append(np.random.choice(idx_col_val,size=int(sample_size*proportions[i2]), replace=False))

            idx = []
            for l in list(map(list, sample_arrays)):
                idx += l

            df_backup = df.copy()
            df = df.loc[idx]
            #print(col," ",len(df))

            if i==len(col_list):
                ## Al mostrar toda la info hacemos return
                return df



    def verificar_dist(df,col_list,col_val_dicc,prop_dicc,margen_error=0.04):
        errores = []
        for col in col_list:
            #print(col)
            df_result = df[col].value_counts(normalize=True) ## Estp es el resultado de lo realizado
            df_temp = pd.DataFrame() ## Este es un df para reflejar lo solicitado
            cols = []
            ps = []
            for i,p in enumerate(prop_dicc[col]):
                cols.append(col_val_dicc[col][i])
                ps.append(p)
            df_temp['col'] = cols
            df_temp['prop'] = ps

            df_temp = df_temp.set_index('col')


            df_result = pd.DataFrame(df_result)
            df_result.columns = ['prop']
            df_result.index = df_result.index.set_names('col')

            diff_resul_realidad = np.max(np.abs(df_result - df_temp)).iloc[0]
            if diff_resul_realidad > margen_error:
                errores.append(diff_resul_realidad)

            #print(errores)
            if len(errores)==0:

                return 1

    for n in range(iteraciones_error):
        df_sample_stratified = one_stratified_sampling_by_feats(df,col_list,col_val_dicc,prop_dicc,last_sample_size,0)

        respuesta_verif = verificar_dist(df_sample_stratified,col_list,col_val_dicc,prop_dicc,margen_error)

        if respuesta_verif:
            break

    print(n,' corridas','\n')
    for col in col_list:
        print('Variable: ',col)
        display(df_sample_stratified[col].value_counts(normalize=True)*100)
        
    return df_sample_stratified




def find_anomalies(df,drop_rows=True,get_feats=False):    
    """
    drop_rows =  Elimina las filas que tienen outliers
    get_feats = Crear un feat indicando que ese row tiene un outlier en ese feat
    """
    for col in df._get_numeric_data().columns:
        if get_feats:
            df["con_out_"+col] = 0
            
        anomalies = []
        df = df[~(df[col].isnull())]
        data = df[col]
        np.random.seed(1)
        # Set upper and lower limit to 3 standard deviation
        data_std = np.std(data)
        data_mean = np.mean(data)
        anomaly_cut_off = data_std * 3

        lower_limit  = data_mean - anomaly_cut_off 
        upper_limit = data_mean + anomaly_cut_off
        # Generate outliers
        for outlier in data:
            if (outlier > upper_limit) | (outlier < lower_limit):
                anomalies.append(outlier)

        ### Repetimos hasta que no haya mas outliers
            if len(anomalies) > 0:
                for a in anomalies:
                    if drop_rows:
                        df = df.drop(df[col][df[col]==a].index,axis=0)
                        
                    if get_feats:
                        idx = df[col][df[col]==a].index
                        #print(idx)
                        df.loc[idx,"con_out_"+col] = 1
                        
            
    return df







def bag_of_words(df,col,n_grams,evitar_words,cant_words=10,show_words=1,output_words=False,sin_stopwords=0,stemmize=1):
    import pandas as pd
    import numpy as np
    import mis_funciones as mf
    import re
    from nltk.stem.snowball import SnowballStemmer
    import warnings
    ss = SnowballStemmer('spanish')
    warnings.filterwarnings('ignore')
        
    def del_endwith_cion(x):
        if x.endswith('cion'):
            return "".join(x.split('cion')[:-1])
        else:
            return x

    
    def freq_words(df,col,evitar_words,
                   cant_words=cant_words
                   ,show_words=show_words
                   ,output_words=output_words
                   ,sin_stopwords=sin_stopwords
                  ,stemmize=stemmize):
        
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import nltk
        from nltk.corpus import stopwords
        import re
        import warnings

        
        warnings.filterwarnings('ignore')
        

        df[col] = df[col].str.lower()
        bag = []
        for d in df[col].str.split():
            bag.append(pd.DataFrame(d))

        df_bag = pd.concat(bag)
        df_bag_freq = df_bag.value_counts()
        words = set(df_bag_freq.index.get_level_values(0))
        #words = words - set(evitar_words)
        
        

        ## Heuristicamente tengo la hipotesis de que si la freq de una palabra es mayor al size*1.25 del df entonces no ayuda
        words_todas = df_bag_freq.sort_values(ascending=False)
        words_todas = words_todas[words_todas<=(len(df)*1.25)]
        
        if show_words:
            display(words_todas.head(cant_words))
        words = list(df_bag_freq.loc[words].sort_values(ascending=False)[:cant_words+1].index.get_level_values(0))
        #print("Palabras elegidas: ",words)

        def find_word(x,*args):
            x = x.replace(" ","").replace("  ","").replace("   ","") #Para que los espacios no sean un problema
            if re.findall(f, x, re.IGNORECASE):
                return 1
            else:
                return 0

        for f in words:
            df[f] = 0
            df[f] = df[col].apply(find_word,args=[f])
            
            
        ## Por ultimo stemizamos la variable descripcion
        df[col] = df[col].apply(lambda x: ' '.join([del_endwith_cion(w) for w in x.split(" ")])) ## Llevar las palabras que terminan en 'cion' a su forma mas primitiva
        
        if stemmize:
            df[col+'_stem'] = df[col].apply(lambda x: ' '.join([ss.stem(w) for w in x.split(" ")]))
        
       
        #
        
        if output_words:
            return (df,words_todas)
        else:
            return df




    def create_grams(df,col,n_grams,evitar_words=evitar_words,stemmize=stemmize):


        def splitTextTo(string,crit=n_grams,evitar_words=evitar_words,stemmize=stemmize):
            
            import re
            from nltk.stem.snowball import SnowballStemmer
            
            string = mf.remover_acentos(string)
            
            dicc_nums = {str(n):"" for n in range(10)}
            for i in dicc_nums.keys(): ## Para eliminar los numeros
                string = string.replace(i,dicc_nums[i])
            
            words = str(string).split()
            
            if stemmize:
                ss = SnowballStemmer('spanish')
                words = [del_endwith_cion(w) for w in words] ## Llevar las palabras que terminan en 'cion' a su forma mas primitiva
                words = [ss.stem(w.lower()) for w in words]
            
            evitar_words = [w.lower() for w in evitar_words]
            
            #Este proceso puede hacerse antes de forma manual, descargar las stopwords de la librería nltk
            #nltk.download('stopwords')
            
            if sin_stopwords:
                #stop_words_sp = set(stopwords.words('spanish'))
                #stop_words_en = set(stopwords.words('english'))
                stop_words_sp = set()
                stop_words_en = set()
            else:
                stop_words_sp = pd.read_csv("C:/Users/mediaz/Jupyter/data/stopwords_esp.txt",sep=" ",header=None)[0].values
                stop_words_sp = set([w.lower() for w in stop_words_sp])
                stop_words_en = []
                stop_words_en = set([w.lower() for w in stop_words_en])

            words = [re.sub(r'[^a-zA-z0-9\s]', '', w) for w in words]
            words = [w for w in words if w not in stop_words_sp]
            words = [w for w in words if w not in stop_words_en]
            words = [w for w in words if w not in evitar_words]
            
            grouped_words = [' '.join(words[i: i + crit]) for i in range(0, len(words), crit) ]

            grouped_words = " ".join([n.replace(" ","") for n in grouped_words])

            return grouped_words
        
        name_new_col = f"{col}_split"+str(n_grams)
        df[name_new_col] = df[col].apply(splitTextTo)

        return (df,name_new_col)

    
    out = create_grams(df,col,n_grams)
    
    out = freq_words(out[0],out[1],evitar_words,cant_words=cant_words,show_words=show_words,output_words=output_words)

    if output_words:
        return (out[0],out[1])
    else:
        return out
    



def remove_special_characters(text, remove_digits=False):
    import re
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text



def contains_and(x,*args):
    ## Para verificar si un registro tiene tiene varios contains
    words = args
    
    counter = 0
    for w in words:
        if w in x:
            counter+=1
            
    if counter==len(words):
        return 1
    else:
        return 0



def get_autocorr(data,date_col,groups,cols,max_lags):
    """Funcion para obtener el porcentaje de cambio en una temporalidad y usarlo como variable"""
    import pandas as pd
    import numpy as np
    
    """
    groups hace referencia a la variable donde ocurre la temporalidad, 
    por ejemplo si la data es del comportamiento de diferentes clientes en el tiempo, el codigo cliente seria uno de los groups
    
    date_col es la variable temporal que se usara para realizar los calculos
    
    cols son las variables continuas a calcular sus variaciones en el tiempo
    
    max_lags la cantidad maxima de lags
    """

    for g in groups:
        data_time = data.sort_values([g,date_col])
        
        for l in range(1,max_lags):
            
            for c in cols:
                
                new_col_name = f'diff_{c}_lag{l}'
                data_time[new_col_name] = data_time[c].pct_change(l) ## Aqui estamos calculando las diferencias temporales
                mask = data_time[g] != data_time[g].shift(l) # Aqui estamos detectando cuando el group esta cambiando
                data_time[new_col_name][mask] = np.nan # Y aqui estamos anulando esos valores calculados
    
    return data_time




def get_feats_by_groups(data,groups,cols,date_col,get_last_n_rows=0,aggs_dicc={},no_incluir_last=1):
    
    """
    groups hace referencia a la variable donde haremos la agregacion
    por ejemplo calcular la media de facturacion de cada cliente, un group es codigo_cliente
        
    cols son las variables continuas a calcular las agregaciones

    """
    import pandas as pd
    import numpy as np
    
    if len(aggs_dicc) == 0:
        aggs_dicc = {}
        for c in cols:
            aggs_dicc[c] = ['sum','mean','std','max','min','nunique','skew','count','median']

    for g in groups:
        
        if get_last_n_rows == 0: ## Si queremos solo sacar los estadisticos de los ultimos n registros
            get_last_n_rows = 9999999999999 
        
        if no_incluir_last:
            data_temp = data.sort_values([g,date_col]).groupby(g).apply(lambda x: x.tail(get_last_n_rows).iloc[:-1,:]).reset_index(drop=True)
            
        else:
            data_temp = data.sort_values([g,date_col]).groupby(g).apply(lambda x: x.tail(get_last_n_rows)).reset_index(drop=True)
        
        #display(data_temp)
        data_gr = data_temp.groupby(g)
        

        for c in cols:
            #print(c)
            aggs = aggs_dicc[c]
            for ag in aggs:
                #print(ag)
                data_gr_ag = data_gr.agg(temp=(c,ag))
                data_gr_ag = data_gr_ag.rename(columns={'temp':f"agg_{g}_{c}_{ag}"})
                
                data = pd.merge(data,data_gr_ag,how='left',left_on=g,right_on=g)
    
    return data



def crear_leyenda_data_grouped(data,col_grouped,col):
    data_leyenda = pd.DataFrame()
    max_len = max([len(data[data[col_grouped]==d][col].unique()) for d in data[col_grouped].unique() if ((d!=0) & (d!=np.nan))] )
    for d in data[col_grouped].unique():
        if (d!=0) & (d!=np.nan):
            series_data = list(data[data[col_grouped]==d][col].unique())

            data_leyenda[d] = series_data + list(np.zeros(max_len-len(series_data)))
            data_leyenda.index.name = col
    return data_leyenda



def clean_tab(tab,del_index_contains):
    
    drop_index_list = []
    for idx in del_index_contains:
        mapeo = [(idx in i)==True for i in tab.index]
        drop_index_list += list(tab.index[mapeo])
        
    drop_index_list = list(set(drop_index_list))   

    tab = tab.drop(drop_index_list)
    
    tab['Aumento_porcentaje_abs'] = tab['Aumento_porcentaje'].abs()
    try:
        tab = tab.loc[tab.Aumento_porcentaje.iloc[:,0].sort_values(ascending=False).index]
    except:
        tab = tab.loc[tab.Aumento_porcentaje.sort_values(ascending=False).index]
    return tab




def get_main_ideas_from_text(out,col,col_stem,min_size_comb=3,max_size_comb=6,cant_ejemplos=3,cant_words_choose=20,cant_ideas=5):
    
    """
    out = Tupla que contiene el output del bag of words
    min_size_comb = Minimo tamano de las combinaciones
    max_size_comb = Maximo tamano de las combinaciones
    cant_words_choose = Cuantas palabras vamos a elegir de las que nos pasaron
    cant_ideas = Cuantas ideas extraer por combinacion
    cant_ejemplos = Cuantos ejemplos queremos que tenga cada combinacion/idea encontrada
    
    """
    
    
    from itertools import combinations
    from tqdm.notebook import trange, tqdm
    warnings.filterwarnings('ignore')
    
    ## Funcion para delete | de los strings
    def del_chars_begin_end(x):
        if len(x)>=2:
            if (x[0] == '|') & (x[-1] != '|'):
                return x[1:]
            elif (x[0] != '|') & (x[-1] == '|'):
                return x[:-1]
            elif (x[0] == '|') & (x[-1] == '|'):
                return x[1:-1]
            else:
                return x
        else:
            return x


    ## Funcion custom para calcular distancia jaccard
    def jaccard_custom_metric(x,*args):
        text1 = x
        text2 = args[0]

        tokens1 = str(text1).split(" ")
        tokens2 = str(text2).split(" ")

        intersec = len(set(tokens1).intersection(set(tokens2)))
        union = len(set(tokens1).union(set(tokens2)))

        jaccard_distance = intersec/union

        return jaccard_distance


    ## Funcion custom para calcular distancia jaccard para apply
    def jaccard_custom_metric_apply(df,col):
        for t in df[col]:
            df['jaccard'] = df[col].apply(jaccard_custom_metric,args=[t])

        score = df['jaccard'].mean()

        return score


    ## Definimos la funcion para crear la columna donde van la combinacion de palabras
    def get_comb_words_dist(df,col_stem,words_from_bag,min_size_comb):

        for w in words_from_bag:
            df[w] = df[col_stem].str.contains(f'{w}').astype(int).astype(str)
            df[w] = df[w].str.replace('1',w)


        df["all"] = ''
        for i,w in enumerate(words_from_bag):
            if i ==0:
                t = df[w].astype(str) + " | "
                df["all"] = t
            else:
                t = df[w].astype(str) + " | "
                df["all"] = df["all"].astype(str) + t

        df["all_cleaned"] = df["all"].str.replace(" ","").str.replace("0","").str.replace(r'(.)\1+','|').apply(del_chars_begin_end)
        df.loc[df["all_cleaned"]=='','all_cleaned'] = 'Otras palabras'
        df.loc[df["all_cleaned"]=='|','all_cleaned'] = 'Otras palabras'

        df['cant_words'] = df["all_cleaned"].apply(lambda x: len(x.split("|")))

        df.loc[df.cant_words<min_size_comb,'all_cleaned'] = 'Otras palabras'

        return df


    ## Sacamos el df y la columna que ya ha sido stemmezida
    df = out[0]
    #col_stem = df.columns[-1]

    words_frombag = [o[0] for o in out[1].index][:cant_words_choose]

    comb_cols = words_frombag

    
    ### Hacemos todas las combinaciones de las palabras mas frecuentes
    comb_bin_cols = []
    for r in range(len(comb_cols)+1):

        comb = combinations(comb_cols,r)
        for i in list(comb):
            comb_bin_cols.append(list(i))

    comb_bin_cols = [c for c in comb_bin_cols if ((len(c) >=min_size_comb) & (len(c) <=max_size_comb))]

    #print(comb_bin_cols)

    ## Calculamos la desviacion estandar de las distribuciones de todas las combinaciones de palbras que coincicidan
    ## con cada texto
    ## Tomaremos la que de una std mayor debido a que eso significa que separa mejor los diferentes textos
    
    comb_frombag = []
    comb_score = []
    for comb in tqdm(comb_bin_cols):

        df_comb_from_bag = get_comb_words_dist(df,col_stem,comb,min_size_comb)
        
        
        if len(df_comb_from_bag.all_cleaned.value_counts().drop('Otras palabras'))>0:
            #score = jaccard_custom_metric_apply(df_comb_from_bag,'Descripcion_split1_stem')
            
            ## Procederemos a usar la distancia de jaccard para calcular que tan homogeneos son los grupos
            ## Osea que de la combinacion de palabras x|z vamos a calcular que tanto se parecen entre si 
            ## luego tomaremos el score de las 5 (cant_ideas) mas altos y los sumaremos
            ## Ese sera el score final 
            combs_groups = set(df_comb_from_bag["all_cleaned"].unique()) - set(['Otras palabras'])
            
            dicc_groups = {}
            combs_groups_list = []
            score_list = []
            
            ## Calculamos el score por cada comb de palabras
            for c in combs_groups:
                df_jaccard = df_comb_from_bag[df_comb_from_bag.all_cleaned==c]
                
                combs_groups_list.append(c)
                score_list.append(jaccard_custom_metric_apply(df_jaccard,col_stem) )
            
            ## Creamos el dicc para luego convertirlos en df
            dicc_groups['comb'] = combs_groups_list
            dicc_groups['score'] = score_list
            
            ## Hacemos el calculo final del score de las 5 combs mas altas
            df_jaccard2 = pd.DataFrame(dicc_groups).sort_values('score',ascending=False)
            
            ## Vamos a tomar un poderado entre el jaccard score y que tambien dichas palabras separan la data
            score_dist = df_comb_from_bag.all_cleaned.value_counts(normalize=True).head(cant_ideas).sum()
            score_jaccard = sum(df_jaccard2['score'].head(cant_ideas)) 
            score = score_dist * score_jaccard
            
        else:
            score = 0
        
        comb_frombag.append(comb)
        comb_score.append(score)


    ## Obtenemos la que tenga mayor separacion
    p = np.array(comb_score).argmax()
    best_comb = comb_frombag[p]
    best_score = comb_score[p]

    print("La mejor combinacion fue: ",best_comb,best_score)

    tab_freq = get_comb_words_dist(df,col_stem,best_comb,min_size_comb).all_cleaned.value_counts(normalize=True).drop('Otras palabras')
    words_clave = tab_freq[tab_freq>0.01].index
    values_clave = tab_freq[tab_freq>0.01].values


    ### Creamos un df con las palabras claves, porcentaje de aparicion y 3 ejemplos
    dfs = []
    n = cant_ejemplos
    area = df['Área Empleado'].str.strip().unique()[0]
    puesto = df['Puesto Empleado'].str.strip().unique()[0]
    
    area = "".join(["-"+s[:4] for s in area.split(" ")]).replace("-","",1) # Achicar el nombre
    puesto = "".join(["-"+s[:4] for s in puesto.split(" ")]).replace("-","",1) # Achicar el nombre
    
    for w,v in zip(words_clave,values_clave):
        df_temp = pd.DataFrame(df[df["all_cleaned"]==w][col].values[:n],columns=['Ejemplo'])
        df_temp["palabra_clave"] = np.repeat(np.array([w]),len(df_temp))
        df_temp['Porcentaje_aparicion'] = np.repeat(np.array([v]),len(df_temp))
        df_temp['Usuario'] = str(list(df['Usuario Final Afectado'].str.strip().unique()))
        df_temp['Area'] = area
        df_temp['Puesto'] = puesto
        dfs.append(df_temp)

    df_ideas = pd.concat(dfs)
    df_ideas = df_ideas[['palabra_clave','Porcentaje_aparicion','Usuario','Area','Puesto','Ejemplo']]
    
    df_ideas = df_ideas.set_index('palabra_clave')
    
    #df_ideas.to_excel(f'out/Ideas_principales_{area}_{puesto}.xlsx')
    
    print("Shape del df ",df.shape)
    
    display(tab_freq.head())
    
    print(tab_freq.head().sum())
    
    return df_ideas




def get_col_sin_out(y_train,metodo=1,multiplo_IQR=3.5):
    
    import mis_funciones as mf
    import pandas as pd
    import numpy as np
    
    ### Metodo de la desviacion estandar
    ### Teniendo en cuenta el teorema de Chebischev que dice que toda distribucion no importa cual debe tener el 94% de su data
    ### Dentro de 4 desv estandar 
    if metodo ==1:
        lower_std = y_train.mean()-(y_train.std()*multiplo_IQR)
        upper_std = y_train.mean()+(y_train.std()*multiplo_IQR)
    
    ### Metodos del boxplot
    elif metodo ==2:
        ### Vamos eliminar los outliers aqui
        IQR = y_train.quantile(0.75) - y_train.quantile(0.25)
        lower_std = y_train.quantile(0.25) - (multiplo_IQR*IQR)
        upper_std = y_train.quantile(0.75) + (multiplo_IQR*IQR)
        max_label = y_train.max()

    ## Crearemos los cuts sin los outliers y luego agregaremos el max_label
    y_train_sin_out = y_train[(y_train<=upper_std) & (y_train>=lower_std)].dropna()

    return y_train_sin_out



def get_non_biased_mean(df,col,cant_cuts=10):
    import mis_funciones as mf
    import pandas as pd
    import numpy as np

    col_qcut = f"{col}_qcut" ### Creamos el nombre de la columna qcut
    
    ## Creamos los rangos por cuantiles
    df[col_qcut] = pd.qcut(df[col],cant_cuts,duplicates='drop')
    
    ## Creamos a min_min y min para asi poder los registros maaas pequenos
    min_min_label = df[col].min() - 0.00001 ##
    min_label = df[col].min()
    cuts_series = df[col_qcut]

    ##Convertimos los rangos creados en una lista 
    cuts = pd.DataFrame(cuts_series.cat.categories.right)
    cuts = [min_min_label] + [min_label] + list(cuts.iloc[:,0]) 
    #cuts = [np.round(c,2) for c in cuts]

    
    ## Lo llevamos a una variable para luego agrupar y obtener los porcentajes(tamanos) de cada grupo y sus medias
    df['group_cut']  = pd.cut(df[col],cuts)
    medias = df.groupby('group_cut')[col].apply(lambda x: mf.get_col_sin_out(x,multiplo_IQR=3.5).mean())
    porcents = df.groupby('group_cut')[col].apply(lambda x: len(x)/len(df))

    return medias,porcents


def proyectar_non_biased_mean(clientes_nuevos,medias,porcents):
    total = 0
    for m,p in zip(medias,porcents):
        res = (clientes_nuevos * p) * m
        total+= res

    return total



############### FUNCIONES PARA BINARY CLASS

def resumir_variables(df,tab_diff_filt,targ,diff=0.05):
    var_target = targ+" |"

    df[targ+"_alto_true"] = 0
    df[targ+"_bajo_true"] = 0

    for t in tab_diff_filt.index:
        if var_target in t:
            val = t.split(" | ")[1]
            val_true = tab_diff_filt.loc[t, True]
            prior_true = tab_diff_filt.loc[t,"Prior_proportion_True"]
            if (val_true - prior_true) > diff:
                #print(t, val_true)
                df[targ+"_alto_true"] = np.where(df[targ]==val,1,df[targ+"_alto_true"])
            elif (val_true - prior_true) < diff:
                #print(t, val_true)
                df[targ+"_bajo_true"] = np.where(df[targ]==val,1,df[targ+"_bajo_true"])
                
                
    return df



def prepare_tabs(df,del_cols,target,porcentaje_signifi=0.1,
                                     diferen_minima=0.03,show_table=0,output_tables=1,
                                     inc_num_cols=1,qcut_len = 8,kmean_cut=1,k_means_cols=[],date_feats=1,tipo_target=1
                                     ,nivel_confianza=0.85):
    import mis_funciones as mf
    import pandas as pd
    import numpy as np
    
    df2 = df[set(df.columns)-set(del_cols)]
    
    out = mf.feature_frecuency_binary_class(df2,target,porcentaje_signifi=porcentaje_signifi,
                                         diferen_minima=diferen_minima,show_table=show_table,output_tables=output_tables,
                                         inc_num_cols=inc_num_cols,qcut_len = qcut_len,signi_pval=0.01,kmean_cut=kmean_cut,date_feats=date_feats) 
    df_tabs = out[0]
    tabs_target = out[1]
    
    
    tab_diff = get_diff_table(tabs_target,error=0,tipo_target=tipo_target)
    tab_diff["diff_porcen"] = (tab_diff["diff_prior"]/tab_diff["Prior_proportion_True"])*100
    tab_diff = tab_diff[["diff_porcen"]+list(set(tab_diff.columns)-set(list("diff_porcen")))]
    filt = tab_diff["diff_abs"] <1
    filt2 = (tab_diff["diff_abs"] <1) & (tab_diff["nivel_confianza"] >nivel_confianza)
    tab_diff_filt = tab_diff[filt2]
    tab_diff_filt = tab_diff_filt[['diff_porcen',True,'Prior_proportion_True','Total_abs','nivel_confianza','intertvalo_confianza_True','diff_abs','Total_rel','diff_prior_abs', 'diff','diff_prior',False]]
    
    
    
    return (df_tabs,tab_diff_filt)




def show_best_props(df,target,del_cols,comp_var="",con_df=0,porcentaje_signifi=0.1,
                                     diferen_minima=0.03,show_table=0,output_tables=1,
                                     inc_num_cols=1,qcut_len = 8,kmean_cut=1,k_means_cols=[],date_feats=1
                                     ,nivel_confianza=0.85):
    
    import mis_funciones as mf
    import pandas as pd
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')
    
    if df[target].nunique()>2:
        tipo_target = 2
    elif df[target].nunique()==2:
        tipo_target = 1
    
    out = mf.prepare_tabs(df,del_cols,target,porcentaje_signifi=porcentaje_signifi,
                 diferen_minima=diferen_minima,show_table=show_table,output_tables=output_tables,
                 inc_num_cols=inc_num_cols,qcut_len = qcut_len,kmean_cut=kmean_cut,
                       k_means_cols=k_means_cols,date_feats=date_feats,tipo_target=tipo_target,nivel_confianza=nivel_confianza)
    
    
    df_tabs = out[0]
    tab_diff_filt = out[1]
    

    if comp_var != "":
        targ = comp_var
        diff = diferen_minima
        df = resumir_variables(df,tab_diff_filt,targ,diff)
        
        ## Volvemos a calcular pero con el valor_alto_true
        target_alto = targ+"_alto_true"
        out2 = prepare_tabs(df,del_cols,target_alto,porcentaje_signifi=porcentaje_signifi,
                 diferen_minima=diferen_minima,show_table=show_table,output_tables=output_tables,
                 inc_num_cols=inc_num_cols,qcut_len = qcut_len)
        
        df_col_alto = out2[0]
        tabs_col_alto = out2[1]
        
        
        ## Para que los findings no esten sesgados por la variable target
        tab_diff_filt_alto = tabs_col_alto[~tabs_col_alto.index.isin(tab_diff_filt.index)]
        
        
        ## Volvemos a calcular pero con el valor_bajo_true
        target_bajo = targ+"_bajo_true"
        out3 = prepare_tabs(df,del_cols,target_bajo,porcentaje_signifi=porcentaje_signifi,
                 diferen_minima=diferen_minima,show_table=show_table,output_tables=output_tables,
                 inc_num_cols=inc_num_cols,qcut_len = qcut_len)
        
        df_col_bajo = out3[0]
        tabs_col_bajo = out3[1]
        
        ## Para que los findings no esten sesgados por la variable target
        tab_diff_filt_bajo = tabs_col_bajo[~tabs_col_bajo.index.isin(tab_diff_filt.index)]
        
        
        ## Para no redundar en los findings que ya aparecen en tab_diff_filt_alto
        tab_diff_filt_bajo = tab_diff_filt_bajo[~tab_diff_filt_bajo.index.isin(tab_diff_filt_alto.index)]
        
        
        ##Por alguna razon el comp_target se convierte en string
        df_col_alto[target_alto] = df_col_alto[target_alto].astype(int)
        df_col_bajo[target_bajo] = df_col_bajo[target_bajo].astype(int)

        if con_df:
            return (df_col_alto,tab_diff_filt_alto,df_col_bajo,tab_diff_filt_bajo)
        else:
            return (tab_diff_filt_alto,tab_diff_filt_bajo)

    
    if con_df:
        return (df_tabs,tab_diff_filt)
    else:
        return tab_diff_filt
    
    
    
    
def find_bad_cuts(tab_diff_filt,df,target):
    import numpy as np
    import pandas as pd
    import mis_funciones as mf
    import warnings
    warnings.filterwarnings('ignore')
    cols_rep = []
    vals_rep = []
    prior_tab = tab_diff_filt.iloc[-1,:]
    for i in tab_diff_filt.index[:-1]:
        col = i.split(" | ")[0]
        val = i.split(" | ")[1]
        if ("_kcut" in col) or ("_qcut" in col):
            #print(col,val)
            cols_rep.append(col)
            vals_rep.append(val)


    df_cols_rep = pd.DataFrame({"cols":cols_rep,"vals":vals_rep})

    df_cols_rep_freq = df_cols_rep["cols"].value_counts()

    #Ya tenemos las columnas que se repiten
    cols = df_cols_rep_freq[df_cols_rep_freq>1].index

    for c in cols:
        #print(c)
        df_comp = mf.compare_var_dist(df,c,target,rel=1).iloc[:-1,:].reset_index()

        df_comp["ini_interv"] = df_comp[c].apply(lambda x: str(x).split(", ")[0].replace("(","")).astype(float)
        df_comp["ini_interv_round"] = np.round(df_comp["ini_interv"])
        df_comp = df_comp.sort_values("ini_interv")
        df_comp["True_diff"] = df_comp[True].diff()
        df_comp["True_diff2"] = df_comp[True].diff(-1)
        df_comp = df_comp.reset_index()
        #display(df_comp)

        filt = df_cols_rep["cols"]==c
        for v in df_cols_rep[filt]["vals"].values:
            #print(v)
            v_ini = np.round(float(v.split(", ")[0].replace("(","")))
            filt2 = ((np.abs(df_comp["ini_interv_round"] - v_ini) <= 0.001))
            #display(df_comp)
            if ( (len(df_comp[filt2])>0) and (np.abs(df_comp[filt2]["True_diff"]).values[0] <= 0.015)):
                #continue
                print("True_diff ",c,v)
                df_founded = pd.concat([df_comp.loc[df_comp[filt2].index-1],df_comp[filt2]])
                ini_interv = str(df_founded[c].iloc[0]).split(", ")[0].replace("(","").replace("(","")
                final_interv = str(df_founded[c].iloc[1]).split(", ")[1].replace("]","").replace("(","")

                if ini_interv != final_interv:  ## Agregamos las filas merged
                    tab_diff_filt = tab_diff_filt.rename(index={f"{c} | {v}":f"{c} | ({ini_interv}, {final_interv}]"})
                else:  ## Eliminamos las filas repetidas
                    #tab_diff_filt.drop(f"{c} | {v}",axis=0,inplace=True)
                    continue
                
            if ((len(df_comp[filt2])>0) and (np.abs(df_comp[filt2]["True_diff2"]).values[0] <= 0.005) ):
                #continue
                print("True_diff2 ",c,v)
                df_founded = pd.concat([df_comp.loc[df_comp[filt2].index+1],df_comp[filt2]])
                ini_interv = str(df_founded[c].iloc[0]).split(", ")[0].replace("(","").replace("(","")
                final_interv = str(df_founded[c].iloc[1]).split(", ")[1].replace("]","").replace("(","")
                
                if ini_interv != final_interv:  ## Agregamos las filas merged
                    tab_diff_filt = tab_diff_filt.rename(index={f"{c} | {v}":f"{c} | ({ini_interv}, {final_interv}]"})
                else:  ## Eliminamos las filas repetidas
                    #tab_diff_filt.drop(f"{c} | {v}",axis=0,inplace=True)
                    continue
    
    #tab_diff_filt = pd.concat([tab_diff_filt,prior_tab],axis=0)
    tab_diff_filt_cleaned = tab_diff_filt
    
    return tab_diff_filt_cleaned

    
    
def find_bad_cuts(tab_diff_filt,df,target):
    import numpy as np
    import pandas as pd
    import mis_funciones as mf
    import warnings
    warnings.filterwarnings('ignore')
    cols_rep = []
    vals_rep = []
    prior_tab = tab_diff_filt.iloc[-1,:]
    for i in tab_diff_filt.index[:-1]:
        col = i.split(" | ")[0]
        val = i.split(" | ")[1]
        if ("_kcut" in col) or ("_qcut" in col):
            #print(col,val)
            cols_rep.append(col)
            vals_rep.append(val)


    df_cols_rep = pd.DataFrame({"cols":cols_rep,"vals":vals_rep})

    df_cols_rep_freq = df_cols_rep["cols"].value_counts()

    #Ya tenemos las columnas que se repiten
    cols = df_cols_rep_freq[df_cols_rep_freq>1].index

    for c in cols:
        #print(c)
        df_comp = mf.compare_var_dist(df,c,target,rel=1).iloc[:-1,:].reset_index()

        df_comp["ini_interv"] = df_comp[c].apply(lambda x: str(x).split(", ")[0].replace("(","")).astype(float)
        df_comp["ini_interv_round"] = np.round(df_comp["ini_interv"])
        df_comp = df_comp.sort_values("ini_interv")
        df_comp["True_diff"] = df_comp[True].diff()
        df_comp["True_diff2"] = df_comp[True].diff(-1)
        df_comp = df_comp.reset_index()
        #display(df_comp)

        filt = df_cols_rep["cols"]==c
        for v in df_cols_rep[filt]["vals"].values:
            #print(v)
            v_ini = np.round(float(v.split(", ")[0].replace("(","")))
            filt2 = ((np.abs(df_comp["ini_interv_round"] - v_ini) <= 0.001))
            #display(df_comp)
            if ( (len(df_comp[filt2])>0) and (np.abs(df_comp[filt2]["True_diff"]).values[0] <= 0.015)):
                #continue
                print("True_diff ",c,v)
                df_founded = pd.concat([df_comp.loc[df_comp[filt2].index-1],df_comp[filt2]])
                ini_interv = str(df_founded[c].iloc[0]).split(", ")[0].replace("(","").replace("(","")
                final_interv = str(df_founded[c].iloc[1]).split(", ")[1].replace("]","").replace("(","")

                if ini_interv != final_interv:  ## Agregamos las filas merged
                    tab_diff_filt = tab_diff_filt.rename(index={f"{c} | {v}":f"{c} | ({ini_interv}, {final_interv}]"})
                else:  ## Eliminamos las filas repetidas
                    #tab_diff_filt.drop(f"{c} | {v}",axis=0,inplace=True)
                    continue
                
            if ((len(df_comp[filt2])>0) and (np.abs(df_comp[filt2]["True_diff2"]).values[0] <= 0.005) ):
                #continue
                print("True_diff2 ",c,v)
                df_founded = pd.concat([df_comp.loc[df_comp[filt2].index+1],df_comp[filt2]])
                ini_interv = str(df_founded[c].iloc[0]).split(", ")[0].replace("(","").replace("(","")
                final_interv = str(df_founded[c].iloc[1]).split(", ")[1].replace("]","").replace("(","")
                
                if ini_interv != final_interv:  ## Agregamos las filas merged
                    tab_diff_filt = tab_diff_filt.rename(index={f"{c} | {v}":f"{c} | ({ini_interv}, {final_interv}]"})
                else:  ## Eliminamos las filas repetidas
                    #tab_diff_filt.drop(f"{c} | {v}",axis=0,inplace=True)
                    continue
    
    #tab_diff_filt = pd.concat([tab_diff_filt,prior_tab],axis=0)
    tab_diff_filt_cleaned = tab_diff_filt
    
    return tab_diff_filt_cleaned


########################################################################



def reducir_dimensionalidad_col_cat(df,group,cat_cols,target_col,sample,quantiles):
    """Funcion para resumir la informacion de una variable categorica muy amplia en base a una numerica"""
    import pandas as pd
    import numpy as np
    ## Eliminamos los outliers de las distribuciones
    col = target_col
    IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
    lower_std = df[col].quantile(0.25) - (3.5*IQR)
    upper_std = df[col].quantile(0.75) + (3.5*IQR)
    max_label = df[col].max()
    min_label = df[col].min()
    df_merged = df[(df[col]<=upper_std) & (df[col]>=lower_std)]
    
    #Por si el df no es unico en group, haremos un groupby, tomaremos el ultimo registro de cada grupo
    #Por lo que asegurate de que esten organizados
    df_merged_gr = df_merged.groupby(group).tail(1)
    
    for i,c in enumerate(cat_cols):
        
        ### Aqui hacemos esto para solo seleccionar los values que tengan la cant de registros que se necesita del sample
        #selec_values = df_merged_gr[c].value_counts()[df_merged_gr[c].value_counts()>=sample].index
        #df_merged_gr = df_merged_gr[df_merged_gr[c].isin(selec_values)]
        
        df_merged_gr_col_target_total = df_merged_gr.groupby(c).agg(
                TARGET_MEAN=(target_col,"mean"),
                CANTIDAD= (group,"count"))



        df_merged_gr_col_target = df_merged_gr_col_target_total[df_merged_gr_col_target_total.CANTIDAD>sample].sort_values(["TARGET_MEAN",'CANTIDAD'],ascending=False)

        print(c,"Hay ",len(df_merged_gr_col_target)," de ",len(df_merged_gr_col_target_total))
        
        display(df_merged_gr_col_target.head())
        display(df_merged_gr_col_target.tail())

        cuts = pd.DataFrame([pd.qcut(df_merged_gr_col_target.TARGET_MEAN,q=quantiles[i]).cat.categories.right])
        cuts = [min_label] + list(cuts.iloc[0,:]) 
        cuts = [np.round(c,2) for c in cuts]
        
        print(cuts)
        
        #labels = [f'ESTRATO {n} de {quantiles[i]}' for n in range(1,quantiles[i]+1)]
        target_cut = pd.cut(df_merged_gr_col_target.TARGET_MEAN,bins=cuts)
        
        target_cut = target_cut.reset_index()
        
        df_join = pd.merge(df,target_cut,how='left',left_on=c,right_on=c)
        
        nuevo_len = len(df_join['TARGET_MEAN'].dropna()) ## Donde el target mean es NAN es porque no encontro join
        anterior_len = len(df[c].dropna()) ## Con lo que se hizo join
        print('Nuevo len ',nuevo_len,'Anterior len ',anterior_len)
        
        print('Se perdio un ',np.round((1-nuevo_len/anterior_len),4)*100,'%')
        
        new_col_name = f'{c}_{target_col}_grouped'
        df_join = df_join.rename(columns={'TARGET_MEAN':new_col_name})
        #display(df_join[new_col_name])
        df_join[new_col_name] = df_join[new_col_name].astype(str).str.replace(" ","").str.replace("  ","").str.replace("\t","")
        
        df = df_join
        
    return df






def get_patrones_temporales(data_func,
                            min_registros,
                            max_registros,
                            cambios_porc,
                            num_cols,
                            group,
                            fecha,
                            lags,
                            ult_reg_n):


    """
### Algoritmo para extraer features/patrones de series de tiempo

En base a una variable fecha (mes_ano)
Y en base a una variable grupo (NUMERO_TARJETA)

Se analizaran cada una de las variables numericas para extraer los patrones significativamente dicriminantes o mas comunes entre todos los registros

"""

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import mis_funciones as mf
    import json
    import datetime
    from tqdm.notebook import trange, tqdm


    ##### PATRONES DE VARIABLES CONTINUAS

    def pat_se_mantiene_igual(x,*args):
        cam = args[0][0]
        l = args[0][1]
        m = args[0][2]
        ult_tipo = args[0][3]

        if (len(x)>m) & (m>0):
            if ult_tipo == 'ultimos_reg':
                x = x.iloc[-m:]
            elif ult_tipo == 'antes_ultimos_reg':
                x = x.iloc[-m*2:-m]
        elif (m==0):
            pass
        elif (len(x)<=m):
            return 0  ## Solo tener en cuenta los clientes que tengan al menos esa cant de registros

        x_ch = x.pct_change(l).fillna(0).iloc[l:] # Calculamos el porcentaje de cambio en cada x

        x = x.iloc[l:] # Los primeros lag registros siempre seran nan
        x_new = np.where((abs(x_ch) <= cam),1,0) ## Aqui calculamos si el cambio estuvo dentro del intervalo
        return np.round(x_new.sum()/len(x_new),1)  # Aqui calculamos el score de los valores que entraron en el rango

    def pat_se_mantiene_subiendo(x,*args):
        cam = args[0][0]
        l = args[0][1]
        m = args[0][2]
        ult_tipo = args[0][3]

        if (len(x)>m) & (m>0):
            if ult_tipo == 'ultimos_reg':
                x = x.iloc[-m:]
            elif ult_tipo == 'antes_ultimos_reg':
                x = x.iloc[-m*2:-m]
        elif (m==0):
            pass
        elif (len(x)<=m):
            return 0  ## Solo tener en cuenta los clientes que tengan al menos esa cant de registros

        x_ch = x.pct_change(l).fillna(0).iloc[l:] # Calculamos el porcentaje de cambio en cada x

        if (x_ch.sum()==0): ## Si nunca hubo cambios return false
            return 0

        x = x.iloc[l:] # Los primeros lag registros siempre seran nan
        x_new = np.where((x_ch >= cam) | ( (x_ch==0) & (x > x.mean()) ),1,0) ## Aqui le estamos diciendo, que es true si el cambio siempre es mayor O el cambio es igual a 0 PERO el valor es mayor a la media, esto es para tener en cuenta cuando x llega a un techo
        return np.round(x_new.sum()/len(x_new),1)  # Aqui calculamos el score de los valores que entraron en el rango


    def pat_se_mantiene_bajando(x,*args):
        cam = args[0][0]
        l = args[0][1]
        m = args[0][2]
        ult_tipo = args[0][3]

        if (len(x)>m) & (m>0):
            if ult_tipo == 'ultimos_reg':
                x = x.iloc[-m:]
            elif ult_tipo == 'antes_ultimos_reg':
                x = x.iloc[-m*2:-m]
        elif (m==0):
            pass
        elif (len(x)<=m):
            return 0  ## Solo tener en cuenta los clientes que tengan al menos esa cant de registros

        x_ch = x.pct_change(l).fillna(0).iloc[l:] # Calculamos el porcentaje de cambio en cada x

        if (x_ch.sum()==0): ## Si nunca hubo cambios return false
            return 0

        x = x.iloc[l:] # Los primeros lag registros siempre seran nan
        x_new = np.where((x_ch <= cam) | ( (x_ch==0) & (x < x.mean()) ),1,0) ## Aqui le estamos diciendo, que es true si el cambio siempre es mayor O el cambio es igual a 0 PERO el valor es mayor a la media, esto es para tener en cuenta cuando x llega a un techo
        return np.round(x_new.sum()/len(x_new),1)  # Aqui calculamos el score de los valores que entraron en el rango

    ##### PATRONES DE VARIABLES BOOLEANAS
    def pat_se_mantiene_true(x,*args):
        cam = args[0][0]
        l = args[0][1]
        m = args[0][2]
        ult_tipo = args[0][3]


        if (len(x)>m) & (m>0):
            if ult_tipo == 'ultimos_reg':
                x = x.iloc[-m:]
            elif ult_tipo == 'antes_ultimos_reg':
                x = x.iloc[-m*2:-m]
        elif (m==0):
            pass
        elif (len(x)<=m):
            return 0  ## Solo tener en cuenta los clientes que tengan al menos esa cant de registros

        x_new = np.where((x==1),1,0) ## Aqui calculamos si el cambio estuvo dentro del intervalo
        return np.round(x_new.sum()/len(x_new),1)  # Aqui calculamos el score de los valores que entraron en el rango

    def pat_se_mantiene_false(x,*args):
        cam = args[0][0]
        l = args[0][1]
        m = args[0][2]
        ult_tipo = args[0][3]

        if (len(x)>m) & (m>0):
            if ult_tipo == 'ultimos_reg':
                x = x.iloc[-m:]
            elif ult_tipo == 'antes_ultimos_reg':
                x = x.iloc[-m*2:-m]
        elif (m==0):
            pass
        elif (len(x)<=m):
            return 0  ## Solo tener en cuenta los clientes que tengan al menos esa cant de registros

        x_new = np.where((x==0),1,0) ## Aqui calculamos si el cambio estuvo dentro del intervalo
        return np.round(x_new.sum()/len(x_new),1)  # Aqui calculamos el score de los valores que entraron en el rango

    def pat_erafalse_ahora_se_mantiene_true(x,*args):
        cam = args[0][0]
        l = args[0][1]
        m = args[0][2]
        ult_tipo = args[0][3]
        x2 = x

        if ((len(x)*2)>m) & (m>0):
            if ult_tipo == 'ultimos_reg':
                x = x.iloc[-m:]
                x2 = x.iloc[-m*2:-m]
            elif ult_tipo == 'antes_ultimos_reg':
                return 0 ## Este patron no aplica para esto
        elif (m==0):
            pass
        elif (len(x)<=m):
            return 0  ## Solo tener en cuenta los clientes que tengan al menos esa cant de registros

        x_new = np.where(((x2==0) & (x==1)),1,0) ## Aqui calculamos si el cambio estuvo dentro del intervalo
        return np.round(x_new.sum()/len(x_new),1)  # Aqui calculamos el score de los valores que entraron en el rango


    def pat_eratrue_ahora_se_mantiene_false(x,*args):
        cam = args[0][0]
        l = args[0][1]
        m = args[0][2]
        ult_tipo = args[0][3]
        x2 = x

        if ((len(x)*2)>m) & (m>0):
            if ult_tipo == 'ultimos_reg':
                x = x.iloc[-m:]
                x2 = x.iloc[-m*2:-m]
            elif ult_tipo == 'antes_ultimos_reg':
                return 0 ## Este patron no aplica para esto
        elif (m==0):
            pass
        elif (len(x)<=m):
            return 0  ## Solo tener en cuenta los clientes que tengan al menos esa cant de registros

        x_new = np.where(((x2==1) & (x==0)),1,0) ## Aqui calculamos si el cambio estuvo dentro del intervalo
        return np.round(x_new.sum()/len(x_new),1)  # Aqui calculamos el score de los valores que entraron en el rango


    
    #data_func = data_juan[data_juan.SE_FINANCIO==1]
    #data_func = data_juan
   
    discrete_cols = [k for k,v in zip(data_func[num_cols].columns,data_func[num_cols].nunique().values) if v <= 12]
    ## Solo elegimos los que tengan minimo x registros o maximo x registros
    data_frec = data_func[group].value_counts()
    group_elegido = data_frec[(data_frec>=min_registros) & (data_frec<=max_registros)]
    data_func = data_func[data_func[group].isin(group_elegido.index)]

    ## Organizamos
    data_func = data_func.sort_values([group,fecha])

    ## Calculamos el cambio en cada interevalo de tiempo
    # dicc_tipo_col = {}
    # for c in num_cols:
    #     #print(c)
    #     for l in lags:
    #         pct_col_name = c+"_ch_lag"+str(l)
    #         data_func[pct_col_name] = data_func.groupby(group)[c].pct_change(l).fillna(0)

    #         ## Guardaremos esto aqui porque segun el tipo de col se calcula diferente
    #         if c in discrete_cols:
    #             dicc_tipo_col[pct_col_name] = 'discreta'
    #         else:
    #             dicc_tipo_col[pct_col_name] = 'continua'


    ## Definimos los tipos de patrones
    list_patrones_cont = [pat_se_mantiene_igual,pat_se_mantiene_subiendo,pat_se_mantiene_bajando]
    list_patrones_bool = [pat_se_mantiene_true,pat_se_mantiene_false,pat_erafalse_ahora_se_mantiene_true,pat_eratrue_ahora_se_mantiene_false]

    list_pat = list_patrones_cont + list_patrones_bool
    list_tipo = ["cont" for c in range(len(list_patrones_cont))] +  ["disc" for c in range(len(list_patrones_bool))] 



    ### Agrupamos por group
    data_func_gr_merged = data_func.groupby(group).mean().reset_index().iloc[:,0:1]

    ## Calculamos donde hubo y no hubo patrones
    for pat,pat_t in zip(list_pat,list_tipo):

        for c in tqdm(num_cols):

            for l in lags:

                for ult in ult_reg_n:

                    for ult_tipo in ["ultimos_reg","antes_ultimos_reg"]:

                        if (c in discrete_cols) and (pat_t=="disc"):
                            cam = 1
                            pct_col_name_cam = c+"_ch_lag"+str(l)+f"_{str(pat).split()[1]}_{ult_tipo}{ult}_{cam}"
                            data_func_gr_pat = data_func.groupby(group)[c].apply(pat,(cam,l,ult,ult_tipo)).reset_index()
                            data_func_gr_pat.columns = [group,pct_col_name_cam]
                            data_func_gr_merged = pd.merge(data_func_gr_merged,data_func_gr_pat,how='inner',left_on=group,right_on=group)
                        elif (c not in discrete_cols) and (pat_t=="cont"):
                            for cam in cambios_porc:
                                pct_col_name_cam = c+"_ch_lag"+str(l)+f"_{str(pat).split()[1]}_{ult_tipo}{ult}_{int(cam*100)}"
                                data_func_gr_pat = data_func.groupby(group)[c].apply(pat,(cam,l,ult,ult_tipo)).reset_index()
                                data_func_gr_pat.columns = [group,pct_col_name_cam]
                                data_func_gr_merged = pd.merge(data_func_gr_merged,data_func_gr_pat,how='inner',left_on=group,right_on=group)

    return data_func_gr_merged


def mayoria_valor_70(x):
    val = x.value_counts(normalize=True).iloc[0:1]
    if val.values >= 0.7:
        return val.index[0]
    else:
        return np.nan




def get_best_cluster_variables(df,target,min_impureza,min_len=500):
    
    """
    target = Variable que se usara para medir la impureza por cada grupo
    min_impureza =  El minimo de impureza para mostrar los valores, Por ejemplo el 80% de la impureza del baseline
    min_len = Minimo de registros para poder calcularle el gini
    """
    
    import numpy as np
    import pandas as pd

    numeric_cols = set(df._get_numeric_data().columns)
    all_cols = set(df.columns) 
    categorical_cols = list((all_cols - numeric_cols) )

    def gini(array):
        
        """Calculate the Gini coefficient of a numpy array."""
        array = array.flatten()
        if np.amin(array) < 0:
            # Values cannot be negative:
            array -= np.amin(array)
        # Values cannot be 0:
        array += 0.0000001
        # Values must be sorted:
        array = np.sort(array)
        # Index per array element:
        index = np.arange(1,array.shape[0]+1)
        # Number of array elements:
        n = array.shape[0]
        # Gini coefficient:
        return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

    def gini_groupby(x):
        if len(x)>min_len:
            return gini(x.dropna().to_numpy())

    
    num_c = target
    df[num_c] = df[num_c].astype(float)

    print('Baseline')
    baseline = gini(df[num_c].dropna().to_numpy())
    print(baseline)
    print()

    scores = [baseline]
    features = ['Baseline']
    #valores = ['Baseline']
    print('Grupos mas puros\n')
    for c in categorical_cols:

        if ((df[c].nunique()<50) & 
        (df[c].nunique()>1) &
        (all(df.groupby(c)[num_c].count().values>0))):

            gini_results = df.groupby(c)[num_c].apply(gini_groupby).dropna()
            gini_results_filt = gini_results[gini_results<(baseline*min_impureza)]
            scor = gini_results.mean()
            #print(scor)
            if len(gini_results_filt)>0:
                #print(c)
                print(gini_results_filt)
                print()
            scores.append(scor)
            features.append(c)


    df_gini = pd.DataFrame([features,scores]).T
    df_gini.columns = ['Feature','Gini']
    df_gini = df_gini.sort_values('Gini')
    df_gini['Baseline'] = df_gini[df_gini.Feature=='Baseline'].Gini.values[0]

    return df_gini



def get_df_tab(df_tab,target,feature,tipo_target):
    import numpy as np
    import pandas as pd
    import mis_funciones as mf
    """
    tipo_target = 1 es para targets categoricos
    tipo_target = 2 es para targets numericos
    """
    if tipo_target == 1:
        df_type_crossed = pd.crosstab(index = df_tab[target],columns = df_tab[feature],margins = True)

        legen = list(df_type_crossed.index)
        legen[-1] = "Total_binary"
        df_type_crossed.index = legen

        tipos = list(df_type_crossed.columns)
        tipos[-1] = "Prior_proportion"
        df_type_crossed.columns = tipos

        df_type_crossed_rel = df_type_crossed/df_type_crossed.loc["Total_binary"]
        df_type_crossed_rel.index = [False,True,"Total_rel"]
        df_type_crossed_rel

        df_type_crossed_totals = pd.DataFrame(df_type_crossed.iloc[-1,:]).T
        df_type_crossed_totals.index = ["Total_abs"]
        df_type_crossed_totals = pd.concat([df_type_crossed_rel,df_type_crossed_totals],axis=0)

        df_type_crossed_totals.index.name=feature

    elif tipo_target == 2:
        df_mean = pd.DataFrame(df_tab.groupby(feature)[target].apply(lambda x: mf.get_col_sin_out(x).mean())).T
        df_mean.index = [False]
        df_mean2 = pd.DataFrame(df_tab.groupby(feature)[target].apply(lambda x: mf.get_col_sin_out(x).mean())).T
        df_mean2.index = [True]
        df_total_rel = pd.DataFrame(df_tab.groupby(feature)[target].apply(lambda x: len(x)/len(df_tab))).T
        df_total_rel.index = ['Total_rel']
        df_total_abs = pd.DataFrame(df_tab.groupby(feature)[target].apply(lambda x: len(x))).T
        df_total_abs.index = ['Total_abs']

        df_type_crossed_totals = pd.concat([df_mean,df_mean2,df_total_rel,df_total_abs])

        df_prior = pd.DataFrame([mf.get_col_sin_out(df_tab[target]).mean(),mf.get_col_sin_out(df_tab[target]).mean(),1,len(df_tab)],columns=['Prior_proportion'])
        df_prior.index = df_type_crossed_totals.index

        df_type_crossed_totals = pd.concat([df_type_crossed_totals,df_prior],axis=1)
        df_type_crossed_totals.index.name=feature


    return df_type_crossed_totals


def calc_priors_tab(df,target,tipo_target):
    import numpy as np
    import pandas as pd
    import mis_funciones as mf
    """
    tipo_target = 1 es para targets categoricos
    tipo_target = 2 es para targets numericos
    """
    if tipo_target == 1:
        tabss = pd.crosstab(index = df[target],columns = 'count',margins = True)
        tabss2 = pd.DataFrame((tabss.T/tabss.loc["All","All"]).T["count"])
        tabss2.index = [False,True,"Total_rel"]
        tabss3 = pd.DataFrame({"Total_abs":[tabss.loc["All"].values[0]]})
        tabss3.index = ["Total_abs"]
        tabss3.columns = ["count"]
        tab_deafult = pd.concat([tabss2,tabss3])
        tab_deafult.columns = ["Prior_proportion"]

    elif tipo_target == 2:
        tab_deafult = pd.DataFrame([mf.get_col_sin_out(df[target]).mean(),mf.get_col_sin_out(df[target]).mean(),1,len(df)],columns=['Prior_proportion'])
        tab_deafult.index = [False,True,"Total_rel","Total_abs"]
        tab_deafult.loc[False] = df[target].std() ## Para convenientemente guardar la desviacion estandar aqui
        
    
    return tab_deafult




def leyenda_target_enc(df,df_enc):
    """
    Para obtener que significan cada uno de los valores del Target encoding
    """
    import pandas as pd
    import numpy as np
    dfs = []
    for c in df.columns:
        data_test = pd.concat([df.loc[idx][c],df_enc.loc[idx][c]],axis=1)
        data_test.columns = ['Val_name',"Enc"]
        data_test_gr = data_test.groupby('Val_name').Enc.apply(lambda x: x.head(1).values[0] if all(x.value_counts(normalize=True)==1) else 'Error').to_frame()
        #data_test_gr.index = [f"{c}_{i}" for i in data_test_gr.index]
        data_test_gr['Col'] = c
        data_test_gr = data_test_gr.reset_index()[['Col','Val_name','Enc']]
        dfs.append(data_test_gr)
        
    return pd.concat(dfs)


    
def detetec_outliers_rows(df):
    import numpy as np
    import pandas as pd
    import mis_funciones as mf
    from pyod.models.copod import COPOD
    
    train,test = mf.organizamos_df(df)

    # train the COPOD detector
    clf = COPOD()
    clf.fit(train)

    # get outlier scores
    y_train_scores = clf.decision_scores_  # raw outlier scores on the train data
    y_test_scores = clf.decision_function(test)  # predict raw outlier scores on test

    train['Outlier_score'] = y_train_scores
    test['Outlier_score'] = y_test_scores
    
    df_out = pd.concat([train,test])
    
    df_out = df_out.rename(columns={0:'Outlier_score'})
    
    return df_out.sort_values('Outlier_score',ascending=False)


def get_correos_validos(df,col_name_email='EMAIL'):
    df_email = df[df[col_name_email].str.match("\A\S+@\S+\Z")==True]
    
    print(f'De {len(df)} correos solo {len(df_email)} tenian correos validos')
    
    return df_email


def get_tel_validos(df,col_name_tel='TELEFONO'):
    import re
    df[col_name_tel] = df[col_name_tel].str.replace('-',"")
    df[col_name_tel] = df[col_name_tel].apply(lambda x: re.sub('^1','',x))

    df_tel = df[df[col_name_tel].apply(lambda x: len(x)==10)]

    print(f'De {len(df)} telefonos solo {len(df_tel)} tenian telefonos validos')
    
    return df_tel



def del_corr_cols(df,target,tol_corr=0.98,drop_frac=0.1,notdrop=[],pref_cols=[]):
    
    from category_encoders import TargetEncoder
    import pandas as pd
    import mis_funciones as mf
    import numpy as np
    
    
    df = df.reset_index(drop=True)
    df_backup = df.copy()

    ### Para convertir a nulls lo que digan 'nan'
    for c in df.columns:
        if str(df[c].dtype) == 'object':
            df[c] = np.where((df[c]=='nan') | (df[c]=='NAN') | (df[c]=='NaN') | (df[c]=='Nan'),np.nan,df[c])


    ### Se vuelve a calcular las cat_cols
    numeric_cols = set(df._get_numeric_data().columns)
    all_cols = set(df.columns) 
    categorical_cols = list((all_cols - numeric_cols) )


    ########################## encodeamos en base a un target



    ## Se entrena con train_target
    #display(train_target[target])
    trains = []
    tests = []
    for c in categorical_cols:
        train_target,test_target = mf.organizamos_df(df[[c,target]].dropna())
        enc = TargetEncoder(cols=[c],min_samples_leaf=50)
        train_target[c] = enc.fit_transform(train_target[c], train_target[target])
        train_target[c].head()
        print(c)
        # Y con test_target solo se predice
        test_target[c] = enc.transform(test_target[c])
        #display(c,test_target[c].mean())

        df.loc[train_target.index,c] = train_target
        df.loc[test_target.index,c] = test_target







    ############################## Calculamos las correlaciones

    #df = pd.concat([train_target,test_target])
    df_corrscores = df.corr()

    mapeo = df_corrscores.unstack().apply(lambda x: x >= tol_corr)
    correlaciones_corr = df_corrscores.unstack()[mapeo]



    #### Organizamos
    correlaciones_corr = correlaciones_corr.reset_index()
    correlaciones_corr.columns = ['col1','col2','score']
    correlaciones_corr = correlaciones_corr.sort_values('score',ascending=False)
    correlaciones_corr = correlaciones_corr.drop_duplicates()

    
    ### Le damos preferencia a algunas columnas
    col_corrs = list(correlaciones_corr.col1.value_counts().index) #Con las que nos quedaremos
    for c in pref_cols:
        col_corrs.remove(c)
    col_corrs = pref_cols + col_corrs
    
    ### Anotamos las columnas redudantes
    dropcols_corr_master = [] ## Las que borraremos
    drop_dicc = {} ## El diccionario donde guardaremos que las corrs borradas
    for col in col_corrs:
        if (col not in dropcols_corr_master):
            #print(col)
            dropcols_corr = correlaciones_corr[correlaciones_corr.col1==col].col2.values
            dropcols_corr = list(dropcols_corr)
            dropcols_corr.remove(col)
            drop_dicc[col] = list(set(dropcols_corr) - set(notdrop))
            dropcols_corr_master = dropcols_corr_master + dropcols_corr

    dropcols_corr_master = list(set(dropcols_corr_master) - set(notdrop) )     
    df = df.drop(dropcols_corr_master,axis=1)
    
    categorical_cols2 = list(set(df_backup.columns).intersection(set(categorical_cols)))
    #print(categorical_cols2)
    #display(df_backup[categorical_cols2])
    df[categorical_cols2] = df_backup[categorical_cols2]
    
    return df,drop_dicc,categorical_cols,dropcols_corr_master




def data_prep_get_variables_influencia(df,
                                       target,
                                       rutas_data_feature_store=[],
                                       cod_id_data_feature_store=[],
                                       cod_id_this_df=[],
                                       tol_corr=0.98,
                                       drop_frac=0.1,
                                      notdrop=[],
                                      pref_cols=[],
                                      include_qcuts=0,
                                      qcut_list=[3,5,8],
                                      print_it=1):
    
    import pandas as pd
    import numpy as np
    import mis_funciones as mf
    from itertools import combinations
    from tqdm.notebook import trange, tqdm


    ### Para convertir a nulls lo que digan 'nan'
    for c in df.columns:
        if str(df[c].dtype) == 'object':
            df[c] = np.where((df[c]=='nan') | (df[c]=='NAN') | (df[c]=='NaN') | (df[c]=='Nan'),np.nan,df[c])

    ## Borramos cols que no aplican
    dropnacols = list(df.notna().sum()[df.notna().sum() <=len(df)*drop_frac].index)
    uniquecols = list(df.nunique()[df.nunique()==1].index)
    df = df.drop(dropnacols+uniquecols,axis=1)
    
    print('Se borraron ',dropnacols,uniquecols)
    
    #print(dropnacols,dropnacols)

    ## Convertimos a string todos los nans
    #for c in df.columns:
        #df[c] = df[c].fillna('NaN')



    ### Creamos los qcuts
    if include_qcuts:
        numeric_cols = set(df._get_numeric_data().columns)
        for c in numeric_cols:
            if target != c:
                for qc in qcut_list:
                    df[f"{c}_qcut{str(qc)}"] = pd.qcut(df[c],qc,duplicates='drop')



    ### Para convertir en object los que sean category
    for c in df.columns:
        if str(df[c].dtype)=='category':
            df[c] = df[c].astype(str)

    ##### Hacemos join con la data del feature store
    df_copy = df.copy()
    dfs = []
    dfs_join = []
    if (len(rutas_data_feature_store)>0):
        ## Aqui cargamos los df de las rutas
        for ruta in rutas_data_feature_store:
            df_ruta = pd.read_csv(ruta)
            dfs.append(df_ruta)

        ## Aqui hacemos los join como tal
        for i,df_cod in enumerate(dfs):
            df_join = pd.merge(df,df_cod,how='left',left_on=cod_id_this_df[i],right_on=cod_id_data_feature_store[i])
            dfs_join.append(df_join)

        ## Unimos todos los df
        df = pd.concat(dfs_join)

        ## Aqui borramos las cols que dicen Unnamed (Osea las columnas repetidas)
        drop_cols_unnnamed = [c for c in df.columns if 'Unnamed' in c]
        df = df.drop(drop_cols_unnnamed,axis=1)


    ## Para borrar las cols con solo 1 valor
    uno_cols =  df.nunique()[df.nunique() == 1].index
    df = df.drop(uno_cols,axis=1)

    ## Para convertir en cat las columnas con menos de 6 valores por si acaso
    bin_cols =  df.nunique()[df.nunique() <= 6].index
    for b in bin_cols:
        if b != target:
            df[b] = df[b].astype(str)
            
    
    
    #return df
    ####################  del_corr_cols
    
    ### Ahora vamos a eliminar las columnas que estan correlacionadas entre si, osea que son redundantes
    tol_corr=tol_corr ## Tolerancia de la redundancia
    drop_frac=drop_frac  ## Las columnas con solo el x% de sus filas no sean nans seran borradas
    notdrop= notdrop + [target] # Las columnas que no queremos tenga en cuenta
    pref_cols= pref_cols # Las columnas con las que queremos haya preferencia si va a eligir entre varias a eliminar
    
    
    out = del_corr_cols(df,target,tol_corr=tol_corr,drop_frac=drop_frac,notdrop=notdrop,pref_cols=pref_cols)
    df = out[0]
    drop_dicc = out[1] ## Diccionario que dice cuales columnas fueron eliminadas y el motivo
    categorical_cols = out[2] 
    dropcols_corr_master = out[3] ## Listado de columnas eliminadas
    
    if print_it:
        print('drop_dicc')
        print(drop_dicc)
    
    
    ### Para convertir a nulls lo que digan 'nan'
    for c in df.columns:
        if str(df[c].dtype) == 'object':
            df[c] = np.where((df[c]=='nan') | (df[c]=='NAN') | (df[c]=='NaN') | (df[c]=='Nan'),np.nan,df[c])

    #return df
    ####################  kcuts
    
    ### Aqui calculamos las columnas que son numericas y categoricas
    numeric_cols = set(df._get_numeric_data().columns)

    ## Las columnas con menos de 10 nunique le pondremos como cat
    numeric_cols = (numeric_cols - set(df[numeric_cols].nunique()[df[numeric_cols].nunique()<10].index )) - set([target])
    all_cols = set(df.columns) 
    categorical_cols = (all_cols - numeric_cols) 

    ### Para si ver cuales columnas le calcularemos el k_cut
    num_kcols = numeric_cols


    ## En caso de que el df sea muy grande se recomienda calcular los k_cuts con una muestra estadisticamente significativa
    ## Esto son los parametros para calcular como tomar esa muestra
    create_mini_df = 0
    if len(df) >30000: 
        create_mini_df = 1
    mini_df_size = np.round((30000/len(df)),2)
    if mini_df_size > 0.3:
        mini_df_size = 0.3

    ### Calculamos los kcuts
    for c in num_kcols:
        print(c)
        try:
            df[c+"_kcut"] = mf.kmeans_bins(df,c,multiplo_IQR=3.5,create_mini_df=create_mini_df,mini_df_size=mini_df_size,mini_df_signi=0.01,print_it=print_it)
        except Exception as e:
            print('En calculo k_cut Error con ',c,e)
            df = df.drop(c,axis=1)

    ## Eliminamos las columnas que ya tenemos los kcuts
    df = df[set(df.columns) - set(num_kcols)]

    ## Eliminamos las columnas que solo sean nan
    drop_na_cols = df.isnull().sum()[df.isnull().sum()==len(df)].index
    df = df.drop(drop_na_cols,axis=1)

    



    return df



def get_multidim_prop(data,col_master,col_group,col_agg,func_agg,metodo=1):
    
    """
    El metodo 1 hace un ranking y escoge los registros que pasen la media de col_group[1:] respecto a sus variaciones
    con col_master por lo que 
    ES IMPORTANTE que col_group[1:] sean variables de fecha NO UNICAS como por ejemplo los mes_num de diferentes anios o los dias
    de diferentes meses
    Por ejemplo comparar las llamadas de cancelacion de servicios en los diferentes dias del mes
    
    El metodo 2 hace una comparacion de las distrubuciones de como col_group[1:] es normalmente y como col_master es
    dependiendo de sus niveles
    Por ejemplo la distribucion de los clasicos en comparacion a los clasicos que llaman para una TC
    
    EN ambos metodos se pueden usar combinaciones para encontrar insights mas ocultos,
    Por ejemplo en el metodo 1, comparar los 1 de enero de todos los años
    
    O por ejemplo en el metodo 2 comparar la distribucion esperada de los clasicos hombres con la que que tienen los 
    clasicos hombres que llaman para una tarjeta de debito
    """
    
    import pandas as pd
    import numpy as np
    import mis_funciones as mf

    #data_gr = data.groupby(col_group)[col_agg].apply(func_agg)/data.groupby([col_master])[col_agg].apply(func_agg)
    data_gr = data[col_group].value_counts(normalize=True)
    data_gr.name = col_agg
    
    if metodo==1:
        data_gr_prior = data_gr.reset_index().groupby([col_master])[col_agg].mean().reset_index()
        data_gr_prior.columns = [col_master,'Prior']
        
        data_gr_skew = data_gr.reset_index().groupby([col_master])[col_agg].skew().reset_index()
        data_gr_skew.columns = [col_master,'Skew']
        
        data_gr2 = pd.merge(data_gr.reset_index(),data_gr_prior,how='inner',left_on=col_master,right_on=col_master)
        data_gr2 = pd.merge(data_gr2,data_gr_skew,how='inner',left_on=col_master,right_on=col_master)

        data_gr2 = pd.merge(data_gr2,data.reset_index().groupby(col_master)[col_agg].apply(func_agg).reset_index(),how='inner',left_on=col_master,right_on=col_master)
    
    elif metodo==2:
        data_gr_prior = data.reset_index()[col_group[1:]].value_counts(normalize=True).reset_index()
        data_gr_prior.columns = col_group[1:]+['Prior']
        
        data_gr_skew = data_gr.reset_index().groupby([col_master])[col_agg].skew().reset_index()
        data_gr_skew.columns = [col_master,'Skew']
        
        data_gr2 = pd.merge(data_gr.reset_index(),data_gr_prior,how='inner',left_on=col_group[1:],right_on=col_group[1:])
        data_gr2 = pd.merge(data_gr2,data_gr_skew,how='inner',left_on=col_master,right_on=col_master)
        
        data_gr2 = pd.merge(data_gr2,data.reset_index().groupby(col_master)[col_agg].apply(func_agg).reset_index(),how='inner',left_on=col_master,right_on=col_master)


 
    
    data_gr2.columns = col_group+['Prop','Prior','Skew','Total abs']
    data_gr2 = data_gr2.drop('Skew',axis=1)

    data_gr2['diff_abs'] = np.abs(data_gr2.Prop - data_gr2.Prior)
    data_gr2['diff'] = data_gr2.Prop - data_gr2.Prior
    data_gr2['Row_abs'] = data_gr2['Total abs'] * data_gr2['Prop']
    
    return data_gr2













def binary_class_multidim(df,target,col_group_dims,prop_signi,min_sample):
    
    import pandas as pd
    import numpy as np

    for c in df.columns:
        if str(df[c].dtype)=='category':
            df[c] = df[c].astype(str)

    col_group = [target]+col_group_dims
    series_list = [df[c] for c in col_group_dims]

    ### Aqui calculamos la frecuencia relativa marginal del grupo
    d1 = pd.crosstab(index = df[target],columns = series_list,margins = False,normalize='columns')
    for s in series_list:
        d1 = d1.stack()
    d1_2 = d1.copy()
    d1 = d1.reset_index()
    
    #display(d1)
    ## Aqui el prior
    
    ## Calculamos el prior con los valores que no son nan
    len_cols = len(col_group_dims)
    filt = df[col_group_dims].notna().sum(axis=1)==len_cols
    
    d2 = df[filt][target].value_counts(normalize=True)
    d2.name = 'All'
    d2.index.name = target
    d2

    ## aqui el diff
    d3 = d1_2-d2
    d3 = d3.reset_index()
    d3 = d3.rename(columns={0:'diff'})
    #display(d3)

    ## Aqui el diff abs
    d35 = d3
    d35['diff_abs'] = np.abs(d35['diff'])
    d35 = d35[d35.diff_abs>=prop_signi]
    

    #display(d35)
    ## Aqui el sample size
    d4 = pd.crosstab(index = df[target],columns = series_list)
    for s in series_list:
        d4 = d4.stack()
    d4 = d4.reset_index()

    d4 = d4.rename(columns={0:'size_sample'})
    d4 = d4.groupby(col_group_dims).size_sample.sum().reset_index()

    ## Empezamos a hacer los join
    d_merged = d35.merge(d2.reset_index())
    d_merged

    #d4[target] = d4[target].astype(int)
    d_merged[target] = d_merged[target].astype(int)

    d_merged2 = d_merged.merge(d4[col_group_dims+['size_sample']])
    d_merged2

    ## Aqui algunos arreglos
    d5 = d1.reset_index()
    d5 = d5.rename(columns={0:'Prop'})

    d_merged3 = d_merged2.merge(d5[col_group+['Prop']])
    d_merged3['Row_abs'] = np.round(d_merged3.Prop * d_merged3.size_sample)
    d_merged3

    ### Algunos renames y enviamos output
    d_merged3 = d_merged3.rename(columns={'All':'Prior','size_sample':'Total abs'})
    d_merged3 = d_merged3[[target, 'Prop', 'Prior',  'Total abs', 'diff_abs', 'diff','Row_abs']+col_group_dims]
    
    d_merged3 = d_merged3[d_merged3['Total abs']>=min_sample]
    
    return d_merged3
    
    
#binary_class_multidim(df,'Legendary',['Type 1'],0,0)



def stack_tab_results(tab,df,diff_prop=0.03):
    #
    
    import pandas as pd
    import numpy as np


    #tab_test = tab_filt2[(tab_filt2.Col_master_val=='Solicitudes de Servicios-Aumento o Disminucion de Limite')]
    #tab = tab[tab.diff_abs>diff_prop]
    max_len = tab.Col_dim2_val.dropna().str.split('\|-\|').apply(lambda x: len(x) if x!=np.nan else 0).max()
    #print(tab.Col_dim2_val)
    ser_list = []
    ser_list_names = []
    #print(max_len)
    #display(tab)
    for l in range(0,max_len):
        ser_list.append(tab.Col_dim2_val.dropna().str.split('\|-\|').apply(lambda x: x[l] if len(x)>l else np.nan))
        ser_list_names.append(tab.Col_dim2.dropna().str.split('\|-\|').apply(lambda x: x[l] if len(x)>l else np.nan))
    
    
    ## Dtypes
    df_dtypes = pd.DataFrame([c for c in df.columns],[df[c].dtype for c in df.columns]).reset_index()
    df_dtypes.columns = ['dtype','Col_names']
    dtypes_list = [pd.merge(s,df_dtypes,how='inner',left_on='Col_dim2',right_on='Col_names') for s in ser_list_names]
    
    ser_concat = pd.concat(ser_list,axis=0)
    ser_concat_names = pd.concat(ser_list_names,axis=0)
    dtypes_concat = pd.concat(dtypes_list,axis=0)
   
    ser_concat_axis_1 = pd.concat(ser_list,axis=1)
    ser_concat_axis_1_names = pd.concat(ser_list_names,axis=1)
    dtypes_concat_axis_1 = pd.concat(dtypes_list,axis=1)
    
    ser_concat.columns = ["col"+str(l) for l in range(0,max_len)]
    ser_concat_names.columns = ["col"+str(l) for l in range(0,max_len)]
    #dtypes_concat.columns = ["col"+str(l) for l in range(0,max_len)]
    
    ser_concat_axis_1.columns = ["col"+str(l) for l in range(0,max_len)]
    ser_concat_axis_1_names.columns = ["col"+str(l) for l in range(0,max_len)]
    #dtypes_concat_axis_1.columns = ["col"+str(l) for l in range(0,max_len)]
    
    ser_concat = ser_concat_names + '---'+ser_concat
    ser_concat_axis_1 = ser_concat_axis_1_names + '---'+ser_concat_axis_1
    
    idx_ser = []
    for s in ser_concat.value_counts().index:
        idx_ser.append(ser_concat[~ser_concat.isin(idx_ser)].value_counts().index[0])
    
    return idx_ser,ser_concat_axis_1,ser_concat



def get_dtype(x):
    if x=='True':
        x = 1
    elif x=='False':
        x = 0
    try:
        x = float(x)
    except:
        x = str(x)
        
    return x


def get_dtype(x):
    if x=='True':
        x = 1
    elif x=='False':
        x = 0
    try:
        x = float(x)
    except:
        x = str(x)
        
    return x


 



def clean_comb_tab_results(tab,df,target,signo,nivel_confianza,error=0,limit=0,top1=0):
    
    import pandas as pd
    import numpy as np
    import mis_funciones as mf
    from tqdm.notebook import trange, tqdm
    
    tab = tab[tab.Col_dim2_val.notna()]
    tab = tab[tab.Col_master_val==1]
    
    
    if len(tab)==0:
        print('No existen correlaciones o el size_sample el muy grande o el nivel de confianza de las correlaciones es muy pequeno')
        return False
    
    if df[target].nunique()>2:
        tipo_target = 2
    elif df[target].nunique()==2:
        tipo_target = 1
    
    max_len = tab.Col_dim2_val.dropna().str.split('\|-\|').apply(lambda x: len(x) if x!=np.nan else 0).max()
    
    ### Para tomar el tab creado cuando es sin combinaciones
    col_names = ['Aumento_porcentaje','Media_del_grupo','Media_esperada',
                                    'sample_size','nivel_confianza','intertvalo_nivel_confianza']

    #tab_data_coldim2_out_iloc = tab_1var.iloc[:,[0,2,3,4,5,6]]
    #tab_data_coldim2_out_iloc.columns = col_names

#     if signo==True:
#         tab_data_coldim2_out_iloc = tab_data_coldim2_out_iloc[tab_data_coldim2_out_iloc['Aumento_porcentaje']>0]

#     else:
#         tab_data_coldim2_out_iloc = tab_data_coldim2_out_iloc[tab_data_coldim2_out_iloc['Aumento_porcentaje']<0]

    if limit==0:
        tab_filt2 = tab.sort_values('diff_abs',ascending=False).reset_index(drop=True).reset_index()
    else:
        tab_filt2 = tab.sort_values('diff_abs',ascending=False).reset_index(drop=True).reset_index().head(limit)
    #tab_filt2 = tab_filt2[tab_filt2.Col_master_val==signo]


    #display(tab_filt2)
    ##### Aqui extraemos los cols y vals
    idx_ser,ser_concat_axis_1,ser_concat = stack_tab_results(tab_filt2,df)

    #cols_tab = ["Col_master","Col_master_val","Prop","Prior",'Total abs','diff_abs']


    ##### #Aqui calculamos los cant_cols
    ser_concat_axis_1['cant_cols'] = ser_concat_axis_1.notna().sum(axis=1)
    tab_filt2['cant_cols'] = ser_concat_axis_1['cant_cols']

    #print('1')
    ##### Aqui los anexamos al tab
    cols = []
    for n in range(max_len):

        tab_filt2['col_name'+str(n)] = ser_concat_axis_1['col'+str(n)].apply(lambda x: str(x).split("---")[0] if (len(str(x).split("---"))>n) else np.nan)
        tab_filt2['col_val'+str(n)] = ser_concat_axis_1['col'+str(n)].apply(lambda x: str(x).split("---")[1] if (len(str(x).split("---"))>n) else np.nan)
        col_name = 'col'+str(n)
        cols.append(col_name)
        tab_filt2[col_name] = ser_concat_axis_1['col'+str(n)]


    ###### Aqui hacemos un stack y merge de los results
    tab_stacked = tab_filt2[cols].stack().reset_index()
    tab_stacked.columns = ['index','col','val']
    tab_stacked_merg = pd.merge(tab_filt2,tab_stacked,how='left',left_on='index',right_on='index')[['index','cant_cols','Prop','diff_abs','col','val']]

    #print('2')
    ###### Aqui calculamos el multiplo de algunas metricas
    multiplo = 1.5
    tab_stacked_merg['diff_abs_multiplo'] = multiplo *np.round(tab_stacked_merg["diff_abs"]/multiplo,2)
    tab_filt2['diff_abs_multiplo'] = multiplo *np.round(tab_filt2["diff_abs"]/multiplo,2)
    tab_stacked_merg['prop_multiplo'] = multiplo *np.round(tab_stacked_merg["Prop"]/multiplo,2)


    
    #idxs = tab_stacked_merg.groupby(['diff_abs_multiplo']).apply(lambda x: x.sort_values(['diff_abs_multiplo','cant_cols'],ascending=[False,True]).iloc[0]['index'])


    ###### Aqui le quitamos los caracteres especiales al nombre de los columns
    for c in cols:
        #print(c)
        df.columns = [mf.remove_special_characters(c) for c in df.columns]
        tab_filt2.loc[tab_filt2[c].notna(),c] = tab_filt2[c].dropna().apply(mf.remove_special_characters)

    #print('3')
    ##### Aqui hacemos el mesh de los cols con sus vals
    tab_filt2 = tab_filt2.fillna('nan')
    tab_filt2['cols_mesh'] = ""
    for c in cols:
        tab_filt2['cols_mesh'] = tab_filt2['cols_mesh']+"|"+tab_filt2[c]

    tab_filt2['cols_mesh'] = tab_filt2['cols_mesh'].dropna().apply(lambda x: x[1:] )
    
    tab_filt2['cols_mesh'] = tab_filt2.cols_mesh.str.replace("\|nan","")
    
    #display(tab_filt2)
    idx_list = tab_filt2.index
    tab_filt2_2 = tab_filt2.copy()
    
    
    #print('4')
    ### Aqui obtenemos los mejores rows por cada grupo, cada grupo corresponde si un col_val corresponde por un or
    choosen_idx = []
    drop_idx = []
    print(idx_list)
    for idx in tqdm(idx_list):
        mapas = []
        if idx not in drop_idx+choosen_idx:
            #print(idx)
            #print(tab_filt2.loc[idx].cols_mesh)
            for c in cols:

                mapeo = tab_filt2_2[c].str.contains(tab_filt2_2.loc[idx].cols_mesh)
                mapas.append(mapeo)

            df_map = pd.DataFrame(mapas).apply(lambda x: any(x))

            #if signo==1:
            tab_gr = tab_filt2_2[df_map].sort_values(['diff_abs_multiplo','cant_cols'],ascending=[False,True])
            #display(tab_gr)
            
            #else:
                #tab_gr = tab_filt2_2[df_map].sort_values(['diff_abs_multiplo','cant_cols'],ascending=[True,True])
                
                
            if top1:
                idx_choosen = tab_gr.iloc[0]['index']
                choosen_idx.append(idx_choosen)
                drop_new = tab_gr.iloc[1:]['index'].to_list()
            else:
                #idx_choosen = tab_gr.groupby('cant_cols').head(1)['index'].to_list()
                idx_choosen = tab_gr['index'].to_list()
                choosen_idx = choosen_idx + idx_choosen
                drop_new = list(set(tab_gr['index']) - set(idx_choosen))

            drop_idx = drop_idx + drop_new
            tab_filt2_2 = tab_filt2_2.drop(drop_new)
    
    
    #print(choosen_idx)
    ## Aqui ya tenemos los mejores indices
    tab_final = tab_filt2[tab_filt2['index'].isin(choosen_idx)].drop_duplicates()

    #display(tab_final[tab_final.cant_cols==1])

    
    #print('5')
    ### Aqui convertimos en df los resultados
    combs_names = []
    tab_combs_values = []
    for idx in tab_final.index:
        comb_name = tab_final.loc[idx].Col_dim2 +"---"+tab_final.loc[idx].Col_dim2_val
        cols_len = len(comb_name.split('---')[0].split('|-|'))
        #print(tab_final.loc[idx].Col_dim2)
        #print(comb_name)
        df_filt = df.copy()
        #print()
        for n in range(cols_len):
            col_name = tab_final.loc[idx]['col_name'+str(n)]
            col_name  =  mf.remove_special_characters(col_name)
            col_val = get_dtype(tab_final.loc[idx]['col_val'+str(n)])
            #print(n,col_name,col_val)
            if col_name!='nan':
                try:
                    df_filt[col_name] = df_filt[col_name].apply(get_dtype)
                    df_filt = df_filt[df_filt[col_name]==col_val]
                except:
                    continue
        
        #display(df_filt)
        sample_size = len(df_filt)

        if sample_size <=0:
            continue


        #display(prior1)
        #display(df_filt)
        #prior1 = df[df_filt.notna().sum(axis=1)==][target].value_counts(normalize=True).loc[1]

        #display()


        #print(col_name)
        ## Si no hay registros == signo
        try:
            prop = df_filt[target].value_counts(normalize=True).loc[1]
            prior1 = df[df[col_name].notna()][target].value_counts(normalize=True).loc[1]

        except:
            continue

        diff_prop = prop-prior1
        aumento = (diff_prop/prior1)*100

        if tipo_target==2:
            p = t_prior["Prior_proportion_True"].iloc[0] ## Media de la poblacion
            poblacion_std = t_prior["Prior_proportion_False"].iloc[0] ## En la version para conitnuas el std se guardo en Prior_proportion_False
            if error==0: #Si el error no fue definido
                error=p*0.1 # Que sea entre un -20% y 20% de la media del prior True
            t_prior["nivel_confianza"] = t_prior["Total_abs"].apply(mf.get_nc_de_media,args=[poblacion_std,error])
            t_prior["intertvalo_confianza_True"] = t_prior[True].apply(mf.get_ic,args=[p,error])
        elif tipo_target==1:
            p = prior1
            if error==0: #Si el error no fue definido
                error=p*0.1 # Que sea entre un -20% y 20% de la proporcion del prior True
            nc = mf.get_nc(sample_size,p,error) 
            ic = mf.get_ic(prop,p,error) 


        tab_combs_values.append([aumento,prop,prior1,sample_size,nc,ic])
        combs_names.append(comb_name)


    #print('6')
    
    #### Finalmente lo concatenemos con el tab sin combinaciones
    tab_combs = pd.DataFrame(tab_combs_values,columns=col_names,index=combs_names)
    #tab_data_coldim2_out_iloc.columns = col_names

    #tab_concated = pd.concat([tab_data_coldim2_out_iloc,tab_combs]).sort_values(['Aumento_porcentaje','nivel_confianza'],ascending=False)

    tab_concated = tab_combs
    tab_concated = tab_concated.groupby('Aumento_porcentaje').head(1)
    tab_concated = tab_concated[tab_concated.nivel_confianza>=nivel_confianza]
    return tab_concated








def get_variables_influencia(df,
                             target,
                             cols_agg=[],
                             func_agg_usar="len",
                             notusecols=[],
                            max_combs_n = 2,
                            size_sample = 500,
                            prop_signi = 0.03,
                            solo_clean = 0,
                            tab = '',
                            signo = True,
                            error=0,
                            nivel_confianza = 0.75,
                            top1=0,
                            show_variables_dis=0):
                            
    

    
    """
    cols_agg = [list-like] Son las variables que se agregaran, prferiblemente que sean una sola
    y que sea una variable que no tenga nans 
    
    func_agg_usar = La funcion de agregado a usar, 
    para calcular las proporciones usar "len"
    para calcular cantidad de valores unicos "nunique"
    para calcular la suma "sum"
    para calcular la media "mean"
    para calcular la dispersion "std"
    
    notusecols = No usar estas columnas en las combinaciones
    
    max_combs_n = Hace referencia al numero maximo de variables por combinacion.
    
    signo = Si queremos ver las correlaciones positivas o negativas
    
    size_sample = Tamano de la muestra minimo para hacer la inferencia

    prop_signi = Minimo de diferencia que debe haber entre la proporcion/media obtenida y lo esperado
    
    error = Es el error del intervalo de confianza,Si el error no fue definido sera entre un -20% y 20% de la media del prior True
    
    nivel_confianza = Es el nivel de confianza con que se hace la inferencia
    """
    
    
    import datetime
    import pandas as pd
    import numpy as np
    import mis_funciones as mf
    from itertools import combinations
    from tqdm.notebook import trange, tqdm
    
    if solo_clean==0:
        ### Definimos las diferentes formas de calcular los agregados
        if func_agg_usar== "nunique":
            func_agg_usar = lambda x: x.nunique()

        elif func_agg_usar== "sum":
            func_agg_usar = lambda x: x.sum()

        elif func_agg_usar== "mean":
            func_agg_usar = lambda x: mf.get_col_sin_out(x).mean()

        elif func_agg_usar== "len":
            func_agg_usar = lambda x: len(x)

        elif func_agg_usar== "std":
            func_agg_usar = lambda x: x.std()


        ## Eliminamos las columnas que tengas valores demasiados dispersos
        drop_cols = []
        for c in df.columns:
            if df[c].value_counts().max() < 50: ## Este numero fue sacado de forma totalmente empirica
                drop_cols.append(c)


        ## Eliminamos las columnas que no usaremos
        cols_clientes =  list(set(df.columns) - set([target] +notusecols + drop_cols))

        ## Hacemos las combinaciones
        comb_bin_cols = []
        comb_cols = cols_clientes

        for r in tqdm(range(1,max_combs_n+1)):

            comb = combinations(comb_cols,r)
            for i in list(comb):
                comb_bin_cols.append(list(i))

        print("Cantidad de combinaciones: ",len(comb_bin_cols))


        ### Definmos los lineamientos de la funcion
        dicc = {}
        dicc['cols_dim'] = [target]

        dicc['cols_agg'] = cols_agg

        dicc['cols_dim2'] = comb_bin_cols

        dicc['aggs_dicc'] = {c:[func_agg_usar] for c in cols_agg} ## La funcion_agg que le pasaremos a cada col_agg



        #### Empezamos a calcular cada combinacion


        
        dfs = []

        for c in tqdm(dicc['cols_dim']):

            for c2 in tqdm(dicc['cols_agg']):

                for col_dim2 in dicc['cols_dim2']:

                    for agg in dicc['aggs_dicc'][c2]:

                        if (c != col_dim2) & (len(df[col_dim2].value_counts()) > 0):
                            #print(col_dim2)
                            data = df
                            col_master = c
                            col_group = [c] + col_dim2
                            col_agg = c2
                            func_agg = agg
                            metodo = 2

                            #data_out = get_multidim_prop(data,col_master,col_group,col_agg,func_agg,metodo)


                            target = col_master
                            col_group_dims = col_dim2
                            prop_signi_bin = 0
                            min_sample_bin = 0
                            
                            
                            data_out = binary_class_multidim(data,target,col_group_dims,prop_signi_bin,min_sample_bin)

                                
                            #display(data_out)
                            ### Para combinar las columnas que son multiples
                            data_sum_cols = pd.DataFrame()
                            col_comb_name = "|-|".join(col_group[1:])
                            data_sum_cols[col_comb_name] = ""
                            for i,dfc in enumerate(col_group[1:]):
                                if i==0:
                                    data_sum_cols[col_comb_name] = data_out[dfc].astype(str)
                                else:
                                    data_sum_cols[col_comb_name] = data_sum_cols[col_comb_name].astype(str) + '|-|' + data_out[dfc].astype(str)

                            data_out = data_out.drop(col_group[1:],axis=1)
                            data_out = pd.concat([data_out,data_sum_cols],axis=1)
                            #data_out2 = pd.concat([data_out2,data_sum_cols],axis=1)  

                            #data_out = pd.concat([data_out,data_out2]).reset_index()


                            #filt = (data_out[col_master]==data_out[col_master])


                            filt = (data_out['Total abs']>size_sample) & (data_out['diff_abs']>prop_signi)


                            #filt2 = (data_out[col_master]==data_out[col_master])
                            #filt2 = (data_test2['Categoria']=='Lineas Aereas')
                            #filt2 = (data_out[col_group[-1]].isin([2,3]))

                            data_out_filt = data_out[filt].sort_values('diff_abs',ascending=False)



                            data_out_filt = data_out_filt.rename(columns={c:"Col_master_val",col_comb_name:"Col_dim2_val"})

                            data_out_filt['Col_master'] = c
                            data_out_filt['Col_dim2'] = col_comb_name
                            data_out_filt['Col_agg'] = c2



                            dfs.append(data_out_filt)
        
        if len(dfs)==0:
            print('El dataset es muy pequeno o ninguna variable tiene correlacion con el target')
            return False
        
        data_out_filt_concated = pd.concat(dfs)
        data_out_filt_concated = data_out_filt_concated.sort_values('diff_abs',ascending=False)
        new_ord = ['Col_master','Col_master_val','Col_dim2','Col_dim2_val','Col_agg','Prop','Prior',
                   'Total abs','diff_abs','diff','Row_abs']
        data_out_filt_concated = data_out_filt_concated[new_ord]

        ## Guardamos un backup por si acaso
        x = datetime.datetime.now()
        data_out_filt_concated.to_csv(f'data_out_filt_concated_{target}_{x.day}-{x.month}-{x.year}.csv')
        tab = data_out_filt_concated
        
    
    ### Definimos el tipo target
    if df[target].nunique()>2:
        tipo_target = 2
    elif df[target].nunique()==2:
        tipo_target = 1
    
    ## Ahora hacemos el cleaning de las combinaciones
    signo = signo ## Positivos o negativos
    tab = tab
    tab_combs = clean_comb_tab_results(tab,df,target,signo,nivel_confianza,error=0,top1=top1).sort_values('Aumento_porcentaje',ascending=signo!=True)
    

    if show_variables_dis==0:
        return tab_combs
        
        
    else:
        for c in df.columns:
            if df[c].dtype.name=='category':
                df[c] = df[c].astype(str)
        
        tab_final = tab_combs
        colnames = [c2.split('|-|') for c2 in [c.split('---')[0] for c in tab_final.index]]
        colvalues = [c2.split('|-|') for c2 in [c.split('---')[1] for c in tab_final.index]]
        colvalues = [list(map(mf.get_dtype,c)) for c in colvalues]
        
        for i,col in enumerate(colnames):
            print(f"{col} {colvalues[i]}")
            data_table = df[df[target]==1][col].value_counts(normalize=True).to_frame()
            data_table.columns = [f"proporcion data target" for c in data_table.columns]
            
            data_table2 = df[df[target]==0][col].value_counts(normalize=True).to_frame()
            data_table2.columns = [f"proporcion data compare" for c in data_table2.columns]
            
            data_table_concated = pd.concat([data_table,data_table2],axis=1)
            
            display(data_table_concated.loc[colvalues[i]])


def max_min_scaler(x,max_x,min_x):
    return (x-min_x)/(max_x-min_x)






def compare_data(df_target,
                 df_compare_list,
                 rutas_data_feature_store = [],
                 cod_id_data_feature_store = [],
                 cod_id_this_df = [],
                 max_combs_n=2,
                 size_sample=500,
                 prop_signi = 0.1,
                 solo_clean = 0,
                 tab = "",
                 signo = True,
                 error=0,
                 nivel_confianza = 0.85,
                 tol_corr=0.98,
                 drop_frac=0.1,
                 notdrop=[],
                 pref_cols=[]):

    ## Creamos la variable target
    target = 'target_compare'
    df_target[target]=1
    for d in df_compare_list:
        d[target] = 0

    ## tomamos solo las columnas comunes
    dfs = [df_target]+df_compare_list
    common_cols = set(dfs[0].columns)
    for i in range(1,len(dfs)):
        common_cols = common_cols.intersection(dfs[i].columns)


    # Concatenamos todos los df
    df_compare_list = [d[common_cols] for d in df_compare_list]
    df = pd.concat([df_target]+df_compare_list)

    
    ## Preparamos la data
    df_prepared = mf.data_prep_get_variables_influencia(df,
                                           target,
                                           rutas_data_feature_store=rutas_data_feature_store,
                                           cod_id_data_feature_store=cod_id_data_feature_store,
                                           cod_id_this_df=cod_id_this_df,
                                           tol_corr=tol_corr,
                                           drop_frac=drop_frac,
                                          notdrop=notdrop,
                                          pref_cols=pref_cols)

    df = df_prepared
    
    
    ## Empezamos a comparar
    ## Importante poner en 1 a 'show_variables_dis' para que muestre las comparaciones
    
    tab_final = mf.get_variables_influencia(df,
                                 target,
                                 cols_agg=['target_compare'],
                                 func_agg_usar="len",
                                 notusecols=[],
                                max_combs_n = max_combs_n,
                                size_sample = size_sample,
                                prop_signi = prop_signi,
                                solo_clean = solo_clean,
                                tab = tab,
                                signo = signo,
                                error=error,
                                nivel_confianza = nivel_confianza,
                                show_variables_dis=1)





def get_cambios_dist_multiple_target(df,cols_dim,cols_agg,cols_dim2,size_sample,prop_signi):

    import pandas as pd
    import numpy as np
    from tqdm.notebook import trange, tqdm


    func_len = lambda x: len(x)

    dicc = {}
    dicc['cols_dim'] = cols_dim
    dicc['cols_agg'] = cols_agg
    dicc['cols_dim2'] = cols_dim2
    dicc['aggs_dicc'] = {cols_agg[0]:[func_len]}
    dicc['cols_dim'] = list(set(dicc['cols_dim']) - set(dicc['cols_dim2']))

    dfs = []

    for c in tqdm(dicc['cols_dim']):

        for c2 in tqdm(dicc['cols_agg']):

            for col_f in dicc['cols_dim2']:

                for agg in dicc['aggs_dicc'][c2]:

                    data = df
                    col_master = c
                    col_group = [c,col_f]
                    col_agg = c2
                    func_agg = agg
                    metodo = 1

                    #data_out = get_multidim_prop(data,col_master,col_group,col_agg,func_agg,metodo)

                    prop_signi_bin = 0
                    min_sample_bin = 0

                    #filt = (data_out[col_master]==data_out[col_master])



                    #filt2 = (data_out[col_master]==data_out[col_master])
                    #filt2 = (data_test2['Categoria']=='Lineas Aereas')
                    #filt2 = (data_out[col_group[-1]].isin([2,3]))

                    data_out = mf.binary_class_multidim(data,col_master,col_group[1:],prop_signi_bin,min_sample_bin)

                    filt = (data_out['Total abs']>size_sample) & (data_out['diff_abs']>prop_signi) & (data_out['diff']>0)

                    data_out_filt = data_out[filt].sort_values('diff_abs',ascending=False)

                    data_out_filt = data_out_filt.rename(columns={c:"Col_master_val",col_f:"Col_fecha_val"})

                    data_out_filt['Col_master'] = c
                    data_out_filt['Col_fecha'] = col_f
                    data_out_filt['Col_agg'] = c2


                    dfs.append(data_out_filt)

    data_out_filt_concated = pd.concat(dfs)
    data_out_filt_concated = data_out_filt_concated.sort_values('diff_abs',ascending=False)
    new_ord = ['Col_master','Col_master_val','Col_fecha','Col_fecha_val','Col_agg','Prop','Prior',
               'Total abs','diff_abs','diff','Row_abs']
    data_out_filt_concated = data_out_filt_concated[new_ord]

    return data_out_filt_concated




def truncate_outliers(df,col,std_n=4,min_notna=0.3,drop_na=1,skew_validation=0,skew_limit=3.5):
    ### Metodo de la desviacion estandar
    ### Teniendo en cuenta el teorema de Chebischev que dice que toda 
    ### distribucion no importa cual debe tener el 94% de su data
    
    import numpy as np
    df_out = df.copy()
    
    if (len(df_out[col].dropna())/len(df_out))<min_notna: ## Entonces no hay na que hacer
        if drop_na:
            return df_out.drop(col,axis=1)
        else:
            return df_out
    
    skew = 9999
    while skew >skew_limit: ## Va a iterar hasta que el skew sea igual o menor al limite
        lower_std = df_out[col].mean() - (df_out[col].std()*std_n)
        upper_std = df_out[col].mean() + (df_out[col].std()*std_n)

        df_out.loc[df_out[col]<lower_std,col] = df_out[df_out[col]>=lower_std][col].min() ## Para tomar el mas cercano luego del corte
        df_out.loc[df_out[col]>upper_std,col] = df_out[df_out[col]<=upper_std][col].min() ## Para tomar el mas cercano luego del corte
        
        skew = np.abs(df_out[col].skew())
        #print(col,skew)
        
        if skew_validation==0: ## Si el skew_validation no esta activado solo itera 1 vez
            return df_out
        
    return df_out


def row_to_cols(x,col):
    df_temp = x[[col]].T
    df_temp.columns = [n for n in range(len(df_temp.columns))]
    return df_temp