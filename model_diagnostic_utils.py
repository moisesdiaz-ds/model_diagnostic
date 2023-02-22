import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import pickle
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
import datetime

warnings.filterwarnings("ignore")
from scipy.stats import ks_2samp
from upsetplot import plot as plot_upset
from tqdm.notebook import trange, tqdm
import shap


import sys

import miscellaneous_functions as mf

import track_model_utils

from sklearn.calibration import calibration_curve
    


def get_confusion_matrix(umbral_elegido,X_test,y_test,features,chosen_model,target):
    df = pd.DataFrame()
    df[target] = y_test[target]
    df['pred_proba'] = chosen_model.predict_proba(X_test[features])[:,1]
    df['pred'] = df['pred_proba']>=umbral_elegido

    df['result'] = ""
    df['result'] = np.where((df[target]==1)&(df['pred']==1),'true_positive',df['result'])
    df['result'] = np.where((df[target]==0)&(df['pred']==1),'false_positive',df['result'])
    df['result'] = np.where((df[target]==0)&(df['pred']==0),'true_negative',df['result'])
    df['result'] = np.where((df[target]==1)&(df['pred']==0),'false_negative',df['result'])
    
    print('--- Absoluto')
    display(np.round(df['result'].value_counts(normalize=False).sort_index(ascending=False),4))
    print()
    print('--- Relativo')
    display(np.round(df['result'].value_counts(normalize=True).sort_index(ascending=False),4))
    print()
    
def model_metrics(df_results,X_test,y_test,features,chosen_model,features2,chosen_model2,target,model_id,model_id2,model_name,model_name2,umbral_elegido,umbral_elegido2):
    # Tomamos las metricas que se guardaron al momento de que se 
    # guardo el modelo
    print(f'=== {model_name}')
    print(df_results['metrics'].set_index('Model').loc[model_id])
    print()
    print(f'== Confusion matrix')
    get_confusion_matrix(umbral_elegido,X_test,y_test,features,chosen_model,target)

    print(f'=== {model_name2}')
    print(df_results['metrics'].set_index('Model').loc[model_id2])
    print()
    print(f'== Confusion matrix')
    get_confusion_matrix(umbral_elegido2,X_test,y_test,features2,chosen_model2,target)
    


def model_feature_importance_gini_index(features,
                                        chosen_model,
                                        features2,
                                        chosen_model2,
                                        model_name,
                                        model_name2,
                                       limit_imp):
    
    
    try:

        if 'xgb' in str(chosen_model).split('(')[0].lower():
            dict_imp = chosen_model.get_booster().get_score(importance_type='gain')
            dict_imp = { c:[dict_imp[c]] for c in dict_imp.keys()}
            df_feature_importance = pd.DataFrame(dict_imp).T.sort_values(0,ascending=False).head(limit_imp)
        else:
            df_feature_importance = pd.DataFrame([features,
                 chosen_model.feature_importances_]).T.set_index(0).sort_values(1,ascending=False).head(limit_imp)

        if 'xgb' in str(chosen_model2).split('(')[0].lower():
            dict_imp2 = chosen_model2.get_booster().get_score(importance_type='gain')
            dict_imp2 = { c:[dict_imp2[c]] for c in dict_imp2.keys()}
            df_feature_importance2 = pd.DataFrame(dict_imp2).T.sort_values(0,ascending=False).head(limit_imp)
        else:
            df_feature_importance2 = pd.DataFrame([features,
                 chosen_model2.feature_importances_]).T.set_index(0).sort_values(1,ascending=False).head(limit_imp)

        
        print(f'=== {model_name}')
        df_feature_importance.plot(kind='barh',figsize=(5,5))
        plt.show()

        print()

        print(f'=== {model_name2}')
        df_feature_importance2.plot(kind='barh',figsize=(5,5))
        plt.show()
        
    except Exception as e:
        print("ERROR ",e)
        pass

    
    
def model_calibration_curve(X_test,y_test,features,chosen_model,model_name):
    

    probs = chosen_model.predict_proba(X_test[features])[:,1]

    # reliability diagram
    x, y = calibration_curve(y_test, probs, n_bins=10)

    # Plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')

    # Plot model's calibration curve
    plt.plot(y, x, marker = '.', label = model_name)

    leg = plt.legend(loc = 'upper left')
    plt.xlabel('Average Predicted Probability in each bin')
    plt.ylabel('Ratio of positives')
    plt.show()
    

def model_compare_data_dist(features,
                            X_train,
                            data_for_comparison):
    
    # El 1-p-value maximo para considerar iguales a 2 distribuciones
    # Usando la prueba kolmogorov
    prob_igualdad_compare_dist = 0.85
    
    # Iteramos en cada feature
    for c in features:
        print(c)
        col = c

        try:
            df_training = X_train.reset_index(drop=True).copy()
            df_pilot_old = data_for_comparison.reset_index(drop=True).copy()

            for n in range(1):
                # Le eliminamos los outliers de todos los valores que
                # sean 4.5 veces mayores o menores al rango intercuartilico
                df_training[col] = mf.get_col_sin_out(df_training[col],metodo=2,multiplo_IQR=4.5)
                df_pilot_old[col] = mf.get_col_sin_out(df_pilot_old[col],metodo=2,multiplo_IQR=4.5)

            plt.subplot(1,2,2)

            plt.subplot(121)
            print(f'Media Data Training: {df_training[col].mean()}',"\n",f'Media Data comparison: {df_pilot_old[col].mean()}')
            df_training[col].hist(figsize=(6,2))


            plt.subplot(122)
            df_pilot_old[col].hist(figsize=(6,2))
            
            plt.subplots_adjust(wspace=0.4)
            plt.show()


            ks = np.round(ks_2samp(df_training[col], df_pilot_old[col])[1],15)
            if ks<=prob_igualdad_compare_dist:
                print(f'Con {col}, Hay un {ks} de proababilidades de que sean iguales')
            else:
                print(f'Con {col} SON IGUALES')

        except Exception as e:
            print(f"ERROR CON {col}",e)
        print("\n-----------------------------\n")
        


def model_performance_by_segment(
    features,
    X_train,
    y_train,
    X_test,
    y_test,
    chosen_model,
    chosen_model2,
    model_name,
    model_name2,
    features2,
    target,
):

    # Unificamos la data training y testing
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    df = pd.concat([train, test])

    # Esto es para llevar a categoricas todas las variables
    limit_n = 10  # Limit unique values
    # (Si la variable tiene menos de 10 valores, se considerara catetgorica)

    numeric_cols = features
    segmen_cols = []
    for c in numeric_cols:
        if target != c:
            if train[c].nunique() > limit_n:
                col_name = f"{c}_qcut6"
                train[col_name] = pd.qcut(train[c], 6, duplicates="drop")
            else:
                col_name = f"{c}_str"
                train[col_name] = train[c].astype(str)
            segmen_cols.append(col_name)

    sample_size = 200  # El sample minimo para tomar en cuenta esa audiencia
    dfs = []  # Donde guardaremos todos los df
    tol = 0.01  # El minimo de diferencia que debe haber para considerar que hubo un cambio
    model = chosen_model  # El modelo a analizar
    model2 = chosen_model2  # Un segundo modelo para comparar
    features_model = features  # Los features del primer modelo
    features_model2 = features2  # Los features del segundo modelo

    dicc_features = {}
    for i, c in enumerate(segmen_cols):

        dicc_c = {}
        dicc_c2 = {}
        df2 = train.sort_values(c).copy()
        df2[c] = df2[c].astype(str)

        # Aqui filtramos los variables que al menos uno de sus valores no tenga el min_sample_size
        if (c != target) & (df2[c].value_counts().max() > sample_size):
            for v in (df2[c].dropna().unique()):
                # Aqui iteramos en cada valor de la variable
                df_forpred = df2[df2[c] == v]
                if (len(df_forpred) >= sample_size):
                    # Si tiene el min_sample_size
                    if (df_forpred[target].nunique() > 1):
                        # Si la variable tiene mas de un valor
                        # Calculamos el performance del modelo para esa audiencia (bureau)
                        preds_p = pd.Series(
                            model.predict_proba(df_forpred[features_model])[
                                :, 1
                            ]
                        )
                        score = np.round(
                            roc_auc_score(df_forpred[target], preds_p), 5
                        )

                        # Calculamos el performance del modelo para esa audiencia (behavior)
                        preds_p2 = pd.Series(
                            model2.predict_proba(df_forpred[features_model2])[
                                :, 1
                            ]
                        )
                        score2 = np.round(
                            roc_auc_score(df_forpred[target], preds_p2), 5
                        )

                        # Guardamos en un dict los scores, en donde la llave es el valor de la variable de turno
                        # print(f'{v} AUC score: ',score)
                        dicc_c[v] = [score]
                        dicc_c2[v] = [score2]

                    # Si tiene solo un mismo valor entonces simplemente se le agrega
                    # el unico valor que tiene como score
                    # Ej: Si todos son target=1 entonces su score obviamente es 1
                    else:
                        unique_val = df_forpred[target].unique()[0]
                        # print(f'{v} AUC score: ',unique_val)
                        dicc_c[v] = [unique_val]
                        dicc_c2[v] = [unique_val]
            # print()

            # Creamos los df con los scores
            df_c1 = pd.DataFrame(dicc_c).T
            df_c1.columns = ["Values"]
            df_c1.index.name = c
            df_c1 = df_c1[(df_c1.reset_index().iloc[:, 0] != "nan").values]
            df_c1["Modelo"] = model_name

            df_c2 = pd.DataFrame(dicc_c2).T
            df_c2.columns = ["Values"]
            df_c2.index.name = c
            df_c2 = df_c2[(df_c2.reset_index().iloc[:, 0] != "nan").values]
            df_c2["Modelo"] = model_name2

            df_c = pd.concat([df_c1, df_c2])
            # Lo guardamos en el dicc de features

            # Calculamos las metricas que nos interesan
            if "qcut" in c:  # Solo si es numerica
                lin_posi_df_c1, lin_posi_df_c2 = (
                    df_c1.Values.diff().dropna() >= -tol
                ).mean(), (df_c2.Values.diff().dropna() >= -tol).mean()
                lin_nega_df_c1, lin_nega_df_c2 = (
                    df_c1.Values.diff().dropna() <= tol
                ).mean(), (df_c2.Values.diff().dropna() <= tol).mean()

                Linealidad_positiva = np.mean([lin_posi_df_c1, lin_posi_df_c2])
                Linealidad_negativa = np.mean([lin_nega_df_c1, lin_nega_df_c2])
            else:
                Linealidad_positiva = 0
                Linealidad_negativa = 0

            Max_diff1 = df_c1.Values.max() - df_c1.Values.min()
            Max_diff2 = df_c2.Values.max() - df_c2.Values.min()
            Mean_diff = np.mean(
                [
                    np.abs(df_c1.Values.diff()).mean(),
                    np.abs(df_c2.Values.mean()).max(),
                ]
            )
            dicc_features[c] = [
                Linealidad_positiva,
                Linealidad_negativa,
                Max_diff1,
                Max_diff2,
                Mean_diff,
                int(i),
            ]

            # Lo agregamos a la lista de dfs
            # Estos dfs seran llamados segun su posicion
            # Porque la posicion en df_features es equivalente a la posicion en dfs
            dfs.append(df_c)

    df_features = pd.DataFrame(
        dicc_features,
        index=[
            "Linealidad_positiva",
            "Linealidad_negativa",
            "Max_diff_model1",
            "Max_diff_model2",
            "Mean_diff",
            "Posicion",
        ],
    )
    df_features = df_features.T

    # Features que muestran el menor incremento en poder predictivo
    # segun aumentan sus valores

    df_top5_max_diff = df_features.sort_values(
        "Max_diff_model1", ascending=False
    ).head(15)

    for p in df_top5_max_diff.Posicion[:]:
        p = int(p)
        df_p = dfs[p]
        # display(dfs[int(p)])
        df_p = pd.concat(
            [
                df_p[df_p.Modelo == model_name]["Values"],
                df_p[df_p.Modelo == model_name2]["Values"],
            ],
            axis=1,
        )
        df_p.columns = [model_name, model_name2]
        df_p.plot(figsize=(15, 3))
        col = df_p.index.name
        plt.title(
            f"Features con mayor diferencia de perfomance segun sus valores {col}"
        )
        plt.grid(True, alpha=0.3)
        plt.xticks([n for n in range(len(df_p.index))], df_p.index)
        plt.show()
        print()

    return df_features, dfs



def model_profiling_weight_of_evidence(
    features,
    X_train,
    y_train,
    X_test,
    y_test,
    chosen_model,
    umbral_elegido,
    target,
):

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    df = pd.concat([train, test])

    probas = chosen_model.predict_proba(df[features])[:, 1]
    preds = probas >= umbral_elegido

    df["True_negative"] = (preds == 0) & (df[target] == 0)
    df["False_negative"] = (preds == 0) & (df[target] == 1)

    df["True_positive"] = (preds == 1) & (df[target] == 1)
    df["False_positive"] = (preds == 1) & (df[target] == 0)

    df_positives = df[df[target] == 1]

    display(df_positives["False_negative"].value_counts())

    for c in df_positives.columns:
        if (str(df_positives[c].dtype) == "category") or (
            "interval" in str(df_positives[c].dtype)
        ):
            df_positives[c] = df_positives[c].astype(str)

    out = mf.del_corr_cols(
        df_positives,
        "False_negative",
        tol_corr=0.85,
        drop_frac=0.1,
        notdrop=[],
        pref_cols=[],
    )
    drop_dicc = out[1]
    # ^ Diccionario que dice cuales columnas fueron eliminadas y el motivo
    dropcols_corr_master = out[3]  # Listado de columnas eliminadas

    dropcols = list(set(dropcols_corr_master) - set(features))
    df_positives = df_positives.drop(dropcols, axis=1)

    # Solo tomamos las columnas numericas
    numeric_cols = set(df_positives._get_numeric_data().columns)
    all_cols = set(df_positives.columns)
    categorical_cols = all_cols - numeric_cols
    len(categorical_cols)

    # numeric_cols = Model_behavior_numeric_data().columns
    num_qcut = 6
    for c in numeric_cols:
        if target != c:
            df_positives[f"{c}_qcut{num_qcut}"] = pd.qcut(
                df_positives[c], num_qcut, duplicates="drop"
            )

    # Falsos negativos - weight of evidence
    df_positives["cod_index"] = 0
    tab_final = mf.get_variables_influencia(
        df_positives,
        "False_negative",
        cols_agg=["cod_index"],
        func_agg_usar="len",
        notusecols=[],
        max_combs_n=1,
        size_sample=300,
        prop_signi=0.03,
        solo_clean=0,
        tab="",
        signo=True,
        error=0,
        nivel_confianza=0.75,
        top1=0,
    )

    display(tab_final.head(20))

    return tab_final, df_positives


def model_profiling_data(df_positives, tab_final):
    # PROFILING
    # Solo tomamos las columnas numericas
    numeric_cols = set(df_positives._get_numeric_data().columns)
    all_cols = set(df_positives.columns)
    categorical_cols = all_cols - numeric_cols
    len(categorical_cols)

    dicc = {}
    for c in categorical_cols:
        col = c
        df_positives[col] = df_positives[col].astype(str)
        d1 = df_positives[df_positives["False_negative"] == 1][
            col
        ].value_counts(normalize=True, dropna=False)

        try:
            d1 = d1.drop(np.nan)
        except:
            pass
        if len(d1[d1 > 0.50]) > 0:
            d1 = d1[d1 > 0.50]
            # print(c)
            # display(d1[d1>0.8])

            d2 = df_positives[col].value_counts(normalize=True, dropna=False)

            diff = (d2 - d1).dropna()

            for d in diff.index:
                sample_size_true = (
                    df_positives[df_positives["False_negative"] == 1][col]
                    .value_counts(normalize=False, dropna=False)
                    .loc[d]
                )
                dicc[diff.name + "_" + d] = [
                    diff.name,
                    d,
                    diff.loc[d],
                    sample_size_true,
                ]

    df_diff_dist = pd.DataFrame(dicc).T
    df_diff_dist.columns = ["Col", "Value", "Diff", "Target_size"]
    df_diff_dist["Diff_abs"] = np.abs(df_diff_dist.Diff)
    df_diff_dist = df_diff_dist.sort_values(
        ["Diff_abs", "Target_size"], ascending=False
    )
    # df_diff_dist.head(20)

    return df_diff_dist


def model_profiling_upset_plot(df_diff_dist, df_positives):
    # UPSET PLOT
    # Obtenemos listas de listas de las combinaciones
    limit = 6
    cols = cols = list(df_diff_dist["Col"].iloc[0:limit].to_list())
    values = list(df_diff_dist["Value"].iloc[0:limit].to_list())
    increases = list(df_diff_dist["Diff_abs"].iloc[0:limit].to_list())

    df_bool = pd.DataFrame()
    for c, v in zip(cols, values):
        df_bool[f"{c}_{v}"] = df_positives[c] == v

    df_bool = pd.get_dummies(df_bool)

    plt.figure(figsize=(20, 4))
    plot_upset(df_bool.value_counts().sort_values(), sort_by="cardinality")
    # plt.savefig('Upset_plot_profile_overrights.png', format="png",dpi=200)
    plt.show()


def model_profiling_scatter_plot(df_diff_dist, df_positives):

    # SCATTER
    # df_diff_dist = df_diff_dist[~df_diff_dist['Col'].str.contains('MewewewtwetORA')]

    df_diff_dist_gr = df_diff_dist.groupby("Diff").apply(
        lambda x: str(list(x.index))
    )
    df_diff_dist_gr = df_diff_dist.merge(df_diff_dist_gr.reset_index())
    df_diff_dist_gr = df_diff_dist_gr.rename(columns={0: "Names"})

    plt.figure(figsize=(15, 10))
    # Preparing dataset
    min_diff = 0.05
    x = df_diff_dist_gr[df_diff_dist_gr["Diff_abs"] > min_diff]["Target_size"]
    y = df_diff_dist_gr[df_diff_dist_gr["Diff_abs"] > min_diff]["Diff_abs"]
    text = df_diff_dist_gr[df_diff_dist_gr["Diff_abs"] > min_diff][
        "Names"
    ].apply(lambda x: x[:150] + "..." if len(x) > 150 else x)
    text = text.apply(lambda x: "\n".join(x.split("',")))
    plt.scatter(x, y)

    # Loop for annotation of all points
    for i in range(len(x)):
        plt.annotate(text[i], (x[i] + 0.1, y[i]))

    plt.ylabel("Porcentaje de differencia con el prior")
    plt.xlabel("Tamaño del target")

    # plt.savefig('Scatter_plot_profile_overrights.png', format="png",dpi=200)
    plt.show()

    return df_diff_dist_gr


def model_profiling_prop_false_negatives(
    df_diff_dist_gr, X_train, y_train, X_test, y_test, df_positives, target
):

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    df = pd.concat([train, test])

    cols = df_diff_dist_gr.Col.unique()

    dfs_false_negatives = []
    for c in cols:
        try:
            c = c.replace("_qcut6", "")
            # Todos los clientes agrupados por si son default o no
            df_gr = df.groupby(target)[[c.replace("_qcut6", "")]].mean().T
            df_gr.columns = ["No defaults", "Defaults"]

            # Defaults agrupados por si son falsos negativos o no
            df_gr2 = (
                df_positives.groupby("False_negative")[
                    [c.replace("_qcut6", "")]
                ]
                .mean()
                .T
            )
            df_gr2.columns = ["True negatives", "False negatives"]

            df_gr3 = pd.concat([df_gr, df_gr2], axis=1)
            dfs_false_negatives.append(df_gr3)
        except:
            print("Error con ", c)

    df_gr_false_negatives = pd.concat(dfs_false_negatives)
    df_gr_false_negatives.index.name = "Column mean by"

    display(df_gr_false_negatives.head(20))

    return df_gr_false_negatives


def model_profiling_false_negatives(
    features,
    X_train,
    y_train,
    X_test,
    y_test,
    chosen_model,
    umbral_elegido,
    target,
):

    # Tab final
    tab_final, df_positives = model_profiling_weight_of_evidence(
        features,
        X_train,
        y_train,
        X_test,
        y_test,
        chosen_model,
        umbral_elegido,
        target,
    )

    # Get the data
    df_diff_dist = model_profiling_data(df_positives, tab_final)

    # Upset plot
    model_profiling_upset_plot(df_diff_dist, df_positives)

    # Scatter plot
    df_diff_dist_gr = model_profiling_scatter_plot(df_diff_dist, df_positives)

    # Proportions false negatives
    df_gr_false_negatives = model_profiling_prop_false_negatives(
        df_diff_dist_gr, X_train, y_train, X_test, y_test, df_positives, target
    )

    return tab_final, df_positives, df_diff_dist_gr, df_gr_false_negatives




def model_shap_analysis(features,
                         X_train,
                         y_train,
                         X_test,
                         y_test,
                         chosen_model,
                         target,
                        umbral,
                        porcentaje_df_sample = 0.5,
            existing_shap_values = None,
            existing_df_shap_values = None,
            show_global_explainer = 1,
            show_partial_dependence_plot = 1,
            show_local_explainer = 1,
            truncate_out = 1,
            skew_validation_out = 0,
            use_normal_shap = 0,
            sample_size = 5,
            specific_rows_local_shap = [],
            ruta_save_img_shaplocal = '',
            show_plot = 1
                ):
    

    
    
    ### Functions 
    def get_shap_local(row,row_name,X,tree_shap_obj,X_ratio_info,features,shap_values):
        """
        Show shap values by an observation using a waterfall plot

        row: Row number of the dataframe you want to plot
        X: The dataframe
        tree_shap_obj: tree_shap_obj returned by treeshap
        X_ratio_info: Dataframe about the numerator and denominator of every ratio variable
        """

        shap_pred = 0 #Prediction
        shap_pred += tree_shap_obj.expected_value #Intercept
        cols_shap_local = [] # Where to save the row data
        vals_shap_local = [] # Where to save the row shap data
        dict_features = {} # Where to save col and shap data by column
        for i,c in enumerate(features):

            sh_val = shap_values[row,i]
            data_val = X.iloc[row,i]
            #print(c,sh_val)
            dict_features[c] = [sh_val,data_val]


        df_feats = pd.DataFrame(dict_features)

        df_feats = df_feats.T.reset_index()
        df_feats.columns = ['Feature','shap_val','data_val']
        df_feats['shap_val_abs'] =  np.abs(df_feats.shap_val)
        df_feats = df_feats.sort_values('shap_val_abs',ascending=True)


        # In order to also have the intercept (expected/prior) prediction
        df_expected = pd.DataFrame({'expected_values':[tree_shap_obj.expected_value,0,tree_shap_obj.expected_value]}).T.reset_index()
        df_expected.columns = df_feats.columns

        # In order to create some blank spaces on the plot
        df_dummy = pd.DataFrame({'':[0,0,0,0]}).T
        df_dummy.columns = df_feats.columns
        df_dummy = pd.concat([df_dummy]*3)

        df_feats = pd.concat([df_expected,df_feats,df_dummy])



        def waterfall(series):
            pred = np.round(series.sum(),3) # The prediction its the sum of all shap data
            if show_plot:
                print('Prediction: ',pred)


            # Code to artificially create an waterfall plot
            df = pd.DataFrame({'pos':np.maximum(series,0),'neg':np.minimum(series,0)})
            blank = series.cumsum().shift(1).fillna(0)
            df.plot(kind='barh', stacked=True, left=blank, color=['b','r'],figsize=(9,5))
            step = blank.reset_index(drop=True).repeat(3).shift(-1)
            step[1::3] = np.nan
            plt.plot(step.values, step.index,'k')
            plt.grid(True,alpha=0.4)

            # Code to add some padding on the plot relative to the waterfall shape 
            last_x = max(series.cumsum())
            padding = 0.07 - np.abs((last_x-pred))
            if padding<0:
                padding=0
            xvals = np.round(np.linspace(0,last_x+padding,round(len(series)/1.5)),2)
            plt.xticks(xvals)

            plt.annotate(pred, (pred,len(series)-2.5),fontsize=13)
            plt.legend(loc='center left')
            if (ruta_save_img_shaplocal!="") and (ruta_save_img_shaplocal!=None):
                plt.savefig(f'{ruta_save_img_shaplocal}/{row_name}.jpg', bbox_inches='tight')
            if show_plot:
                plt.show()

        # Code to organize row data
        series = df_feats.set_index('Feature').shap_val
        series.index = [f"{str(np.round(v,2))} = {i}" for i,v in zip(df_feats.Feature,df_feats.data_val)]
        series.index = [i.replace("0.0 = expected_values",'expected_probabilty').replace("0.0 = 0",'') for i in list(series.index)]

        # Show the data info about the columns that create every ratio_variable
        if show_plot:
            print(X_ratio_info.iloc[row])
            print()

        waterfall(series)

        return series





    ### Code starts
    if existing_shap_values==False or existing_shap_values==None:

        print('\n====== Creating shap values')
        train = pd.concat([X_train,y_train],axis=1)
        test = pd.concat([X_test,y_test],axis=1)
        df = pd.concat([train,test])

        if porcentaje_df_sample<1:
            print('\nSampling dataset')
            df_sample = mf.extract_mini_dataset(df[features+[target]],porcentaje=porcentaje_df_sample,significancia=0.10,print_it=0)

            X,y,df = df_sample[features],df_sample[target].astype(bool),df_sample.merge(df)
        else:
            X,y,df = df[features],df[target].astype(bool),df

        ## Obtener las columnas que son de ratio del modelo
        ratio_cols = [c for c in X.columns if c.startswith('ratio_')]
        ratio_cols_taken = [c for c in X_train.columns if any([c in c2 for c2 in ratio_cols])]
        ratio_cols_taken = [c for c in ratio_cols_taken if 'ratio_' not in c]
        X_ratio_info = df[ratio_cols_taken]

        # Le truncamos los outliers
        # Todo lo que este mas a alla de 4 desviaciones estandar
        # Ademas iterara la funcion indeifinadamente hasta que el skew sean menor
        # A 3
        X2 = X.copy()
        if truncate_out:
            print('\nTruncating outliers')

            for c in X.columns:
                X2 = mf.truncate_outliers(X2,c,std_n=4,min_notna=0.2,drop_na=1,skew_validation=skew_validation_out,skew_limit=3)

        X = X2


        #display(X.isnull().sum())


        #########  Check if tree_based model
        if  (
            ((("max_depth" in dir(chosen_model)) and ('max_leaf_nodes' in dir(chosen_model))) or
             (("max_depth" in dir(chosen_model)) and ('get_booster' in dir(chosen_model))) or
             (("max_depth" in dir(chosen_model)) and ('max_leaves' in dir(chosen_model)))) 
        and (use_normal_shap==0)
        ):
            print('\n====== Creating TREE shap values')
            tree_shap_obj = shap.TreeExplainer(chosen_model,X,
                                model_output='probability',
                                feature_pertubation='tree_path_dependent')

            shap_values = tree_shap_obj.shap_values(X)

            shap_values = shap.Explanation(values=shap_values, data=np.array(X), feature_names=list(X_train.columns), base_values=np.array([tree_shap_obj.expected_value]*len(X)))


            try: # Cuando es randomforest esto es una lista
                tree_shap_obj.expected_value = tree_shap_obj.expected_value[1]
            except:
                pass

        else:
            print('\nCreating NORMAL shap values')
            # build a clustering of the features based on shared information about y
            clustering = shap.utils.hclust(X, y)

            # above we implicitly used shap.maskers.Independent by passing a raw dataframe as the masker
            # now we explicitly use a Partition masker that uses the clustering we just computed
            masker = shap.maskers.Partition(X, clustering=clustering)


            # build an Exact explainer and explain the model predictions on the given dataset
            explainer = shap.explainers.Exact(chosen_model.predict_proba, masker)
            shap_values = explainer(X)

            # get just the explanations for the positive class
            shap_values = shap_values[...,1]

            ### Guardamos
            now = datetime.datetime.now()
            today = f"{now.day}-{now.month}-{now.year}"

            pickle.dump(X, open(f"data/df_shap_values_beha_{today}.dmp", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(shap_values, open(f"data/shap_values_beha_{today}.dmp", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    else:

        print('\n======== Loading Shap values')
        f = open(existing_shap_values, 'rb')
        shap_values = pickle.load(f)

        X = pd.read_pickle(existing_df_shap_values)


    #Validacion de tipo de shap_values #Cuando es randomforest esto es una lista
    if isinstance(shap_values,list):
        shap_values = shap_values[1]


    #########  Check if tree_based model
    if  (
        ((("max_depth" in dir(chosen_model)) and ('max_leaf_nodes' in dir(chosen_model))) or
         (("max_depth" in dir(chosen_model)) and ('get_booster' in dir(chosen_model))) or
         (("max_depth" in dir(chosen_model)) and ('max_leaves' in dir(chosen_model)))) 
    and (use_normal_shap==0)
    ):

        if show_global_explainer:
            print('\n======== TREE show_global_explainer')
            shap.summary_plot(shap_values, X)
            plt.show()

        if show_partial_dependence_plot:
            print('\n======== show_partial_dependence_plot')
            ## Creamos un diccionario para ordenar todo
            dicc_order_feat = {}
            for i,v in enumerate(shap_values.values[1]):
                #print(list(X.columns)[i],i)
                dicc_order_feat[list(X.columns)[i]] = i

            ## Ordenamos por importancia
            importances = chosen_model.feature_importances_
            sorted_indices = np.argsort(importances)[::-1]
            cant_features_show = 50
            feats_ordered = X.columns[sorted_indices[:cant_features_show]]

            ## Ploteamos
            cols_non_corr = []
            linealidad_dicc = {}
            for i,f in enumerate(feats_ordered):

                df_shap_corr = pd.DataFrame()
                df_shap_corr['shap_values'] = shap_values.values[:,dicc_order_feat[f]]
                df_shap_corr['shap_data'] = shap_values.data[:,dicc_order_feat[f]]

                if (len(df_shap_corr.corr().stack())>0) & (df_shap_corr.shap_values.nunique()>1):
                    linealidad = df_shap_corr.corr().iloc[:,0].fillna(0)[df_shap_corr.corr().iloc[:,0].fillna(0)<1].values[0]
                else:
                    linealidad = 0

                print(f,linealidad)
                shap.plots.scatter(shap_values[:,dicc_order_feat[f]])
                plt.show()

                if linealidad>0:
                    linealidad_dicc[f] = 1
                else:
                    linealidad_dicc[f] = -1

                if np.abs(linealidad)<0.65:
                    cols_non_corr.append(f)


        if show_local_explainer:
            #display(X.isnull().sum())
            shap_values = shap_values.values
            print('\n======== show_local_explainer_plot by type of prediction')

            if len(specific_rows_local_shap)>0:
                ## Esta tecnica es para obtener la posicion de cada uno de los indices que se le pasaron al df
                X_reseted = X.copy()
                X_reseted.index.name = "any_name_index"
                X_reseted = X_reseted.reset_index(drop=False)
                specific_rows_local_shap_iloc = list(X_reseted[X_reseted['any_name_index'].isin(specific_rows_local_shap)].index)
                del X_reseted
            else:
                specific_rows_local_shap_iloc = []

            ## Solo eliminara el index si no son coherentes
            try:
                xy = pd.concat([X,y],axis=1)
            except:
                print('index deleted')
                xy = pd.concat([X.reset_index(drop=True),y.reset_index(drop=True)],axis=1)

            xy['pred_proba'] = chosen_model.predict_proba(X[features])[:,1]
            xy['pred'] = xy['pred_proba']>=umbral
            q_labels = ['quantile_25','quantile_50','quantile_75','quantile_100']
            xy['pred_q'] =  pd.cut(xy['pred'],[0,0.25,0.5,0.75,1],q_labels)

            xy['true_positive'] = np.where((xy[target]==1)&(xy['pred']==1),1,0)
            xy['false_positive'] = np.where((xy[target]==0)&(xy['pred']==1),1,0)
            xy['true_negative'] = np.where((xy[target]==0)&(xy['pred']==0),1,0)
            xy['false_negative'] = np.where((xy[target]==1)&(xy['pred']==0),1,0)

            xy['pred_q'] =  pd.cut(xy['pred_proba'],[0,0.25,0.5,0.75,1])
            q_labels = ['pred_q_'+str(c) for c in list(xy['pred_q'].unique())]
            xy = pd.get_dummies(xy,columns=['pred_q'])


            #cols_pred = ["true_positive","false_positive","true_negative","false_negative"]
            cols_pred = []
            cols_pred = cols_pred + q_labels
            dfs = []
            dict_local_shap_rows = {}
            dict_local_shap_rows_qlabels = {}
            for c in cols_pred:
                if show_plot:
                    print("### ",c)
                    print()

                # Esto es en caso de que el sample_size sea mayor al numero de observaciones
                if sample_size>sum(xy[c]==1):
                    sample_size_arg = sum(xy[c]==1)
                else:
                    sample_size_arg = sample_size

                ## Se va a verificar si se tomara una muestra de todos los rows que califiquen o de rows especificos de la variable specific_rows_local_shap_iloc
                if len(specific_rows_local_shap_iloc)==0:
                    d = xy[xy[c]==1].sample(sample_size_arg,replace=False)
                    idx_rows = list(d.index)
                    idx_rows_nums = [i for i,name in enumerate(list(d.index))]
                else: 
                    xy_specific = xy.iloc[specific_rows_local_shap_iloc][xy[c]==1]
                    if len(xy_specific)>0:
                        d = xy_specific
                        idx_rows = list(d.index)
                        idx_rows_nums = [i for i,name in enumerate(list(d.index))]
                    else:
                        idx_rows = []
                        idx_rows_nums = []

                for i in idx_rows_nums:
                    if show_plot:
                        print(f'=== {c} | Registro #: ',idx_rows[i])

                    #return i,X,tree_shap_obj,X_ratio_info,features,shap_values

                    series_result_shap_local = get_shap_local(i,idx_rows[i],X,tree_shap_obj,X_ratio_info,features,shap_values)

                    if c in q_labels:
                        dict_local_shap_rows_qlabels[idx_rows[i]] = [series_result_shap_local,X_ratio_info,c]
                    else:
                        dict_local_shap_rows[idx_rows[i]] = [series_result_shap_local,X_ratio_info,c]

    else:
        if show_global_explainer:
            print('\n======== show_global_explainer')
            shap.summary_plot(shap_values, X)

        if show_partial_dependence_plot:
            print('\n======== show_partial_dependence_plot')
            ## Creamos un diccionario para ordenar todo
            dicc_order_feat = {}
            for i,v in enumerate(shap_values.values[1]):
                #print(list(X.columns)[i],i)
                dicc_order_feat[list(X.columns)[i]] = i

            ## Ordenamos por importancia
            importances = chosen_model.feature_importances_
            sorted_indices = np.argsort(importances)[::-1]
            cant_features_show = 50
            feats_ordered = X.columns[sorted_indices[:cant_features_show]]

            ## Ploteamos
            cols_non_corr = []
            linealidad_dicc = {}
            for i,f in enumerate(feats_ordered):

                df_shap_corr = pd.DataFrame()
                df_shap_corr['shap_values'] = shap_values.values[:,dicc_order_feat[f]]
                df_shap_corr['shap_data'] = shap_values.data[:,dicc_order_feat[f]]

                if (len(df_shap_corr.corr().stack())>0) & (df_shap_corr.shap_values.nunique()>1):
                    linealidad = df_shap_corr.corr().iloc[:,0].fillna(0)[df_shap_corr.corr().iloc[:,0].fillna(0)<1].values[0]
                else:
                    linealidad = 0

                print(f,linealidad)
                shap.plots.scatter(shap_values[:,dicc_order_feat[f]])
                plt.show()

                if linealidad>0:
                    linealidad_dicc[f] = 1
                else:
                    linealidad_dicc[f] = -1

                if np.abs(linealidad)<0.65:
                    cols_non_corr.append(f)

        tree_shap_obj = ""

    return shap_values,X,y,tree_shap_obj,dict_local_shap_rows,dict_local_shap_rows_qlabels








def model_feature_importance_by_perm(
    features, X_train, y_train, X_test, y_test, iterations=5, perc_draw=1
):
    scores = {}
    scores_list = []
    cols_sets_list = []
    all_cols = features

    for i in range(iterations):
        for n, col in enumerate(tqdm(all_cols)):

            draw = round(perc_draw * len(X_train[all_cols]))
            draw_test = round(perc_draw * len(X_test[all_cols]))

            index_chosen_train = np.random.choice(
                np.array(X_train[all_cols].index), draw
            )
            index_chosen_test = np.random.choice(
                np.array(X_test[all_cols].index), draw_test
            )

            # Sin la nueva columna
            X_train_func = X_train.loc[index_chosen_train, all_cols].copy()
            X_test_func = X_test.loc[index_chosen_test, all_cols].copy()
            y_train_func = y_train.loc[index_chosen_train].copy()
            y_test_func = y_test.loc[index_chosen_test].copy()

            clf = XGBClassifier(
                gamma=1.5,
                min_child_weight=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=0,
            )
            clf.fit(X_train_func, y_train_func)
            score = (
                roc_auc_score(y_test_func, clf.predict_proba(X_test_func)[:, 1])
                * 2
                - 1
            )

            # Con la nueva columna
            # chosen_cols_perm = set(all_cols) -set([col])
            X_train_func = X_train.loc[index_chosen_train, all_cols].copy()
            X_test_func = X_test.loc[index_chosen_test, all_cols].copy()
            y_train_func = y_train.loc[index_chosen_train].copy()
            y_test_func = y_test.loc[index_chosen_test].copy()

            X_train_func[col] = (
                X_train_func[col].sample(frac=1, replace=False).values
            )
            X_test_func[col] = (
                X_test_func[col].sample(frac=1, replace=False).values
            )

            clf = XGBClassifier(
                gamma=1.5,
                min_child_weight=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=0,
            )
            clf.fit(X_train_func, y_train_func)
            without_col_score = (
                roc_auc_score(y_test_func, clf.predict_proba(X_test_func)[:, 1])
                * 2
                - 1
            )
            diff_gini = score - without_col_score
            scores[col] = diff_gini

            scores_list.append(scores)

    df_scores = pd.DataFrame(scores_list).mean().sort_values()

    return df_scores


def model_diagnostic(X_train,
                    y_train,
                    X_test,
                    y_test,
                    target,
                    dir_results_files,
                    dir_models,
                    model_id,
                    model_id2,
                    umbral_elegido,
                    umbral_elegido2,
                    limit_imp,
                    model_name,
                    model_name2,
                    data_for_comparison,
                    porcentaje_df_sample=0.7,
                    existing_shap_values=None,
                    existing_df_shap_values=None,
                    existing_shap_values_df_compare=None,
                    existing_df_shap_values_df_compare=None,
                    truncate_out = 1,
                    skew_validation_out_shap = 0,
                    use_normal_shap=0,
                    sample_size_local_shap = 20,
                    return_model_metrics=1,
                    return_feature_importance_gini=1,
                    return_model_calibration_curve=1,
                    return_compare_data_dist=1,
                    return_performance_by_segment=0,
                    return_profiling_false_negatives=0,
                    return_shap=0,
                    return_shap_df_compare=0,
                    return_feature_importance_by_perm=0,
                    ):
    
    
    """
    Get a diagnostic of the model, how was trained and how is perfoming.
    Args:
        X_train: Data the model used for the training phase
        
        y_train: Target data the model used for the training phase
        
        X_test: Data the model used for the test phase
        
        y_test: Target data the model used for the test phase
        
        dir_results_files: Directory from the tracking models package 
        where the metrics, stats and other files are stored.
        
        dir_models: Directory from the tracking models package 
        where the models are stored.
        
        model_id: Model id you want to diagnose
        
        model_id2: Model id you want to compare
        
        umbral_elegido: Chosen threshold for making the predictions
        
        limit_imp: Qty of features youn want to see on the 
        feature importance section 
        
        model_name: Name how you identify the model (Can be anything) 
        
        model_name2: Name how you identify the 2nd model (Can be anything) 
        
        data_for_comparison: Data you want to use for comparing with
        the training data distribution 
        
        return_model_metrics: If you want to show the model metrics
        
        return_feature_importance_gini: If you want to show the 
        feature importance using gini method
        
        return_compare_data_dist: If you want to show features distribution
        compared with other data
        
        return_performance_by_segment: If you want to show the model
        performance by segment
        
        return_profiling_false_negatives: If you want to show 
        the model metrics false negatives profiling
        
        return_shap: If you want to show the model shap analysis
        
        return_feature_importance_permutation: If you want to show the 
        model feature importance by permutation
        

    Returns:
        Dictionary with the most important insights
    """
    
    if isinstance(y_test,pd.Series):
        y_test = y_test.to_frame()

    if isinstance(y_train,pd.Series):
        y_train = y_train.to_frame()

    
    ### LOAD DEL MODELO
    # Si ya existe no volveremos a intanciar y cargar el modelo y demas datos
    if 'model_results' not in locals():
        # Instanciamos la clase
        model_results = track_model_utils.ClassModelResults()
        df_results = model_results.get_model_results(dir_results_files)

        # Escogemos el modelo 
        dict_results = track_model_utils.load_model(model_id,dir_models)
        chosen_model = dict_results["chosen_model"]

        dict_results = track_model_utils.load_model(model_id2,dir_models)
        chosen_model2 = dict_results["chosen_model"]

        # Features del modelo (No tomar el ultimo valor)
        features = list(df_results['features_train_cols'].set_index('Model').loc[model_id].dropna().values)[:-1]
        target = list(df_results['features_train_cols'].set_index('Model').loc[model_id].dropna().values)[-1]

        # Features del modelo 2 (No tomar el ultimo valor)
        features2 = list(df_results['features_train_cols'].set_index('Model').loc[model_id2].dropna().values)[:-1]
        target2 = list(df_results['features_train_cols'].set_index('Model').loc[model_id2].dropna().values)[-1]
        
    dicc_return = {}
    
    if return_model_metrics:
        print('\n===== Mostrando Las metricas del modelo\n')
        model_metrics(df_results,X_test,y_test,features,chosen_model,features2,chosen_model2,target,model_id,model_id2,model_name,model_name2,umbral_elegido,umbral_elegido2)
    
    if return_feature_importance_gini:
        print('\n===== Mostrando el feature importance by gini\n')
        model_feature_importance_gini_index(features,
                                                chosen_model,
                                                features2,
                                                chosen_model2,
                                                model_name,
                                                model_name2,
                                               limit_imp)   
        
        
    if return_model_calibration_curve:
        print('\n===== Mostrando la calibracion del modelo\n')
        for n in range(1,3):
            if n==1:
                n=''
            chosen_model_n = locals()[f'chosen_model{n}']
            features_n = locals()[f'features{n}']
            model_name_n = locals()[f'model_name{n}']
            model_calibration_curve(X_test,y_test,features_n,chosen_model_n,model_name_n)
            
            
    if return_compare_data_dist:
        print('\n===== Comparando la distribucion de los features\n')
        model_compare_data_dist(
                            features,
                            X_train,
                            data_for_comparison)
        
    if return_performance_by_segment:
        print('\n===== Mostrando el performance del modelo por segmento\n')
        df_features,dfs = model_performance_by_segment(features,
                                         X_train,
                                         y_train,
                                         X_test,
                                         y_test,
                                         chosen_model,
                                         chosen_model2,
                                         model_name,
                                         model_name2,
                                         features2,
                                         target)
        
        dicc_return['df_features'] = df_features
        dicc_return['dfs_perf_segment'] = dfs
        
        

    
    if return_profiling_false_negatives:
        print('\n===== Mostrando el profiling de los falsos negativos\n')
        tab_final,df_positives,df_diff_dist_gr,df_gr_false_negatives = model_profiling_false_negatives(features,
                     X_train,
                     y_train,
                     X_test,
                     y_test,
                     chosen_model,
                     umbral_elegido,
                     target
                   )
        
        dicc_return['df_diff_dist_gr'] = df_diff_dist_gr
        dicc_return['tab_final'] = tab_final
        dicc_return['df_positives'] = df_positives
        dicc_return['df_gr_false_negatives'] = df_gr_false_negatives
        
    if return_shap:
        shap_values,X,y,tree_shap_obj,dict_local_shap_rows = model_shap_analysis(features,
                         X_train,
                         y_train,
                         X_test,
                         y_test,
                         chosen_model,
                         target,
                        umbral = umbral_elegido,
                        porcentaje_df_sample = porcentaje_df_sample,
            existing_shap_values = existing_shap_values,
            existing_df_shap_values = existing_df_shap_values,
            show_global_explainer = 1,
            show_partial_dependence_plot = 1,
            truncate_out = truncate_out,
            skew_validation_out=skew_validation_out_shap,
            use_normal_shap=use_normal_shap,
            sample_size = sample_size_local_shap
                       )
        
        dicc_return['shap_values'] = shap_values
        dicc_return['shap_values_df'] = X
        dicc_return['shap_values_df_y'] = y
        dicc_return['tree_shap_obj'] = tree_shap_obj
        
    if return_shap_df_compare:
        shap_values_compare,X_compare,y_compare,tree_shap_obj_compare = model_shap_analysis(features,
                         data_for_comparison,
                         pd.DataFrame(),
                         pd.DataFrame(),
                         pd.DataFrame(),
                         chosen_model,
                         target,
                        umbral = umbral_elegido,
                        porcentaje_df_sample = porcentaje_df_sample,
        existing_shap_values = existing_shap_values_df_compare,
        existing_df_shap_values = existing_df_shap_values_df_compare,
            show_global_explainer = 1,
            show_partial_dependence_plot = 1,
            truncate_out = truncate_out,
            skew_validation_out=skew_validation_out_shap,
            use_normal_shap=use_normal_shap,
            sample_size = sample_size_local_shap
                       )
        
        dicc_return['shap_values_compare'] = shap_values_compare
        dicc_return['shap_values_df_compare'] = X_compare
        dicc_return['shap_values_df_y_compare'] = y_compare
        dicc_return['tree_shap_obj_compare'] = tree_shap_obj_compare
        
        
    if return_feature_importance_by_perm:
        df_scores_perm= model_feature_importance_by_perm(features,
                         X_train,
                         y_train,
                         X_test,
                         y_test,
                         iterations =5,
                         perc_draw = 1
                         )
        
        dicc_return['df_scores_perm'] = df_scores_perm
        
    return dicc_return