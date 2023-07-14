import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import collections
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr, kendalltau, kruskal, chi2_contingency, shapiro, anderson, kstest, normaltest
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import axes3d

def missing_cells(df):
    '''Calcule le nombre de cellules manquantes sur le data set total'''
    return df.isna().sum().sum()

def missing_cells_perc(df):
    '''Calcule le pourcentage de cellules manquantes sur le data set total'''
    return df.isna().sum().sum()/(df.size)

def missing_general(df):
    '''Donne un aperçu général du nombre de données manquantes dans le data frame'''
    print('Nombre total de cellules manquantes :',missing_cells(df))
    print('Nombre de cellules manquantes en % : {:.2%}'.format(missing_cells_perc(df)))
    
def valeurs_manquantes(df):
    '''Prend un data frame en entrée et créer en sortie un dataframe contenant le nombre de valeurs manquantes
    et leur pourcentage pour chaque variables. '''
    tab_missing = pd.DataFrame(columns = ['Variable', 'Missing values', 'Missing (%)'])
    tab_missing['Variable'] = df.columns
    missing_val = list()
    missing_perc = list()
    
    for var in df.columns:
        nb_miss = missing_cells(df[var])
        missing_val.append(nb_miss)
        perc_miss = missing_cells_perc(df[var])
        missing_perc.append(perc_miss)
        
    tab_missing['Missing values'] = list(missing_val)
    tab_missing['Missing (%)'] = list(missing_perc)
    return tab_missing

def data_duplicated(df):
    '''Retourne le nombres de lignes identiques.'''
    return df.duplicated(keep=False).sum()

def row_duplicated(df,col):
    '''Retourne le nombre de doublons de la variables col.'''
    return df.duplicated(subset = col, keep='first').sum()

def duplicated(df,col):
    '''Supprime les doublons de la variable col du data frame df.'''
    print('Le nombre de lignes dupliquées en fonction de {} est : {}'.format(col, row_duplicated(df,col)))
    df.drop_duplicates(subset = col, keep = 'first', inplace = True)
    print('Suppression des duplicatas effectuée.')
    print('La nouvelle taille du data frame est :', df.shape)

def pays_france(df):
    '''Réduction du dataframe au territoire français.'''
    data_country = df['countries_fr'].str.contains('Franc|Réunion|Nouvelle-Calédonie|Martinique|Guadeloupe|Polynésie française|Mayotte|Guyane|Wallis et Futuna|Saint-Pierre-et-Miquelon', 
                                                    case = False)
    return df.loc[data_country==True]

def barplot_missing(df):
    '''Affiche le barplot présentant le pourcentage de données manquantes par variable.'''
    proportion_nan = df.isna().sum().divide(df.shape[0]/100).sort_values(ascending=False)
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 30))
    ax = sns.barplot(y = proportion_nan.index, x=proportion_nan.values)
    plt.title('Pourcentage de données manquantes par variable', size=15)
    plt.show()
    
def bar_missing(df):
    '''Affiche le barplot présentant le nombre de données présentes par variable.'''
    msno.bar(df)
    plt.title('Nombre de données présentes par variable', size=15)
    plt.show()
    
def graph_missing(df):
    '''Affiche le graphe présentant le pourcentage de données manquantes par variable.'''
    tab = valeurs_manquantes(df).sort_values(by=['Missing (%)'])
    plt.figure(figsize=(30, 15))
    plt.plot(tab['Variable'],tab['Missing (%)'])
    plt.xticks(rotation=90)
    plt.xlabel('Variable')
    plt.ylabel('% Missing')
    plt.title('Pourcentage de données manquantes')
    plt.show()

def drop_columns_empty(df,lim):
    '''Prend en entrée un data frame et un seuil de remplissage de données.
    Supprime chaque variable ayant un pourcentage de données manquantes supérieur à celui renseigné. 
    Donne en sortie le data frame filtré avec les colonnes à garder.'''
    
    tab = valeurs_manquantes(df)
    columns_keep = list()
    for row in tab.iterrows():
        if float(row[1]['Missing (%)'])>float(lim):
            print('Suppression de la variable {} avec % de valeurs manquantes {}'.format(row[1]['Variable'],
                                                                                         round(float(row[1]['Missing (%)']),2)))
            
        else :
            columns_keep.append(row[1]['Variable'])
    
    return df[columns_keep]    
    
def boxplot(df,ylim):
    '''Affiche le boxplot des variables.'''
    fig = plt.figure(figsize=(20, 15))
    ax = plt.axes()
    plt.xticks(rotation=90)
    ax.set_ylim(ylim)
    sns.boxplot(data=df)
    plt.title('Boxplot des variables', size=15)    

def multi_boxplot(df):
    ''' Affiche indépendamment tous les boxplots des variables sélectionnées'''
    fig, axs = plt.subplots(4,3,figsize=(20,20))
    axs = axs.ravel()
    
    for i,col in enumerate(df.columns):
        sns.boxplot(x=df[col], ax=axs[i])
    fig.suptitle('Boxplot pour chaque variable quantitative')
    plt.show()

def nettoyage_100g(dataframe,list_col):
    '''Pour chaque variable, on remplace par NaN chaque valeur non comprise dans l'intervalle [min;max]'''
    df = dataframe.copy()
    for var in range(len(list_col)):
        var_nom = list_col[var][0]
        var_min = list_col[var][1]
        var_max = list_col[var][2]
        count_av = df[var_nom].count()
        df[var_nom] = df[var_nom].apply(lambda x: np.nan if (x < var_min or x > var_max) else x)
        count_ap = df[var_nom].count()
        
        print('{} : Nombre de valeurs remplacées : {} ; % de valeurs remplacées : {}%.\n'.format(var_nom,
              str(count_av - count_ap), (count_av - count_ap) / count_av * 100))
    return df        
   
def incoherences(df,col):
    '''Retourne les valeurs incohérentes des variables corrélées.
    C'est-à-dire par exemple , les variables telles que fat_100g est supérieur à la somme des autres variables 
    liées au gras dont elle est composée.
    '''
    data_inco = df.loc[df[col[0]]<df[col[1]]]
    print(data_inco.shape)
    return data_inco

def transformation_NaN(df,index,var):
    ''' Transforme en NaN la variable indiquée pour les index sélectionnés.'''
    taille = df.loc[index].shape[0]
    df.loc[index,var]=np.nan
    print('Transformation effectuée sur: {} observations'.format(taille))
    
def incoherence_somme(df):
    '''Retourne les valeurs incohérentes telles que la somme de chacune des composantes est supérieure à 100g.'''
    data_somme = df.loc[df['fat_100g']+ df['carbohydrates_100g']+ df['proteins_100g']+df['salt_100g']>100]
    print(data_somme.shape)
    return data_somme

def drop_lignes(df,index):
    '''Supprime les lignes des index donnés en argument'''
    df.drop(index, axis=0, inplace = True, errors = 'ignore')
    print('Suppression effectuée')
    
def val_nutri_absentes(df):
    '''Retourne les index des lignes ne possédant aucune valeur nutritionnelle.'''
    
    all_na = df[df["energy_100g"].isna()
                & df["proteins_100g"].isna()
                & df["sugars_100g"].isna()
                & df["fat_100g"].isna()
                & df["saturated-fat_100g"].isna()
                & df["carbohydrates_100g"].isna()
                & df["salt_100g"].isna()
                & df["fiber_100g"].isna()
                    ]
    
    print("Nombre d'observations ayant toutes les variables nutritionnelles absentes: ",all_na.shape[0])
    return all_na.index

def imput_simple(df,col,strat,valeur):
    '''Imput les colonnes sélectionnées selon la stratégie précisée (mean,median,most_frequent,constant)'''
    #df1 = df.copy()
    
    if strat == 'constant':
        imput = SimpleImputer(strategy=strat, fill_value=valeur)
    else :
        imput = SimpleImputer(strategy=strat)
    
    df[col] = imput.fit_transform(df[col])
    print("Valeurs avec lesquelles ont été complétées la ou les colonnes:")
    print(imput.statistics_)
       
    return df

def iterative_imput(df,min_value,max_value):
    '''Imput les colonnes sélectionnées avec un IterativeImputer.'''
    imputer = IterativeImputer(min_value=min_value, max_value=max_value)
                               #,keep_empty_features=True
                              
    imputed = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(imputed, columns=df.columns)

    return df_imputed

def knn_imputer(df):
    '''Imput les colonnes sélectionnées à l'aide d'un KNNImputer.'''
    scaler = StandardScaler().fit(df)
    scaled = scaler.transform(df)

    imputer = KNNImputer(n_neighbors=5)

    knn = imputer.fit_transform(scaled)
    filled_data = scaler.inverse_transform(knn)

    df_filled = pd.DataFrame(filled_data, columns=df.columns, index=df.index)
    
    return df_filled

def calcul_energie(df,col):
    '''Calcule la variable energy_100g à l'aide des coefficients de conversion de chaque valeur nutritionnelle.'''
    dic_vn = {'fat_100g':37, 'saturated-fat_100g':0, 'carbohydrates_100g':17, 'sugars_100g':0, 'fiber_100g':8,
              'proteins_100g':17, 'salt_100g':0,'sodium_100g':0}
        
    df1 = df[col].copy()    
    calcul_energy = df1.dot(pd.Series(dic_vn))

    return calcul_energy

def remplacement_colonnes(df,col):
    '''Remplace les valeurs manquantes de la 1ère variable passé en paramètre avec celles de la 2ème variable passée en paramètre. Le remplacement est effectué dans la variable directement.'''
    index = df.loc[df[col[0]].isna()==True].index
    df.loc[index,col[0]] = df.loc[index,col[1]]

def nutrigrade(df):
    '''Détermine le nutrigrade en fonction du nutriscore.'''
    df['nutri_calcul'] = pd.cut(df['nutrition-score-fr_100g'],[-15,-1,2,10,18,40], 
                                labels=list('abcde'), include_lowest=True).astype('object')
    return df['nutri_calcul'] 

def nettoyage_string(df,col):
    '''Nettoie une colonne de string.'''
    df[col] = df[col].str.replace('en:','')
    df[col] = df[col].str.replace('de:','')
    df[col] = df[col].str.replace('-',' ')
    df[col] = df[col].str.lower()
    print('Nettoyage effectué')
    
def occurrences_traces(df,col,nb_occur):
    '''Détermine et affiche dans un barplot les nb_occur mots ayant le plus d'occurrences dans la colonne traces.'''
    data = pd.DataFrame(df[col])
    data = data.assign(traces_sep=data[col].str.split(',')).explode('traces_sep')
    data = pd.DataFrame(data['traces_sep'])
    nettoyage_string(data,'traces_sep')
    trace_occur = collections.Counter(data['traces_sep'].tolist()).most_common(nb_occur)
    df_trace_occur = pd.DataFrame(trace_occur, 
                                  columns = ["Mot", "Nombre d'occurrences"]).sort_values(by="Nombre d'occurrences", 
                                                                                        ascending=False)
    

    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 10))
    sns.barplot(y = 'Mot', x="Nombre d'occurrences", data = df_trace_occur)
    plt.title("{} mots avec le plus d'occurrences dans les traces".format(nb_occur), size=20)
    plt.show()

    return df_trace_occur

def filtre_traces(df,col,df_traces):
    '''Filtre la variable trace en ne gardant que les nb_occur mots.'''
    df1 = df.copy()
    df_traces = df_traces.drop(0,axis=0)
    liste_traces = df_traces['Mot'].tolist()
    traces_filtre = df1[col].str.contains('|'.join(liste_traces),case = False)
    df1['traces'] = df1.loc[traces_filtre==True]['traces_fr']
    return df1

''' Notebook exploration '''

def distribution(df,colonnes):
    ''' Affiche les histogrammes pour chaque variable renseignée.'''
    fig, axs = plt.subplots(4,3,figsize=(20,20))
    axs = axs.ravel()

    for i, col in enumerate(colonnes):
        sns.histplot(data=df, x=col, bins=30, kde=True, ax=axs[i])        
    fig.suptitle('Distribution pour chaque variable quantitative')
    plt.show()
    
def indicateurs(df,colonnes):
    '''Calcul les coefficient d'asymétrie et d'aplatissement pour chaque variable renseignée.'''
    for col in colonnes:
        print("Étude de {} :".format(col))
        print("Coefficient d'asymétrie de la variable {} : {}".format(col,round(df[col].skew(),3)))
        print("Coefficient d'aplatissement de la variable {} : {}".format(col,round(df[col].kurtosis(),3)))
        print("-"*70)
        
def qq_plot(df,colonnes):
    ''' Affiche le diagramme quantile-quantile entre chacune des variables renseignées et une loi normale '''
    fig = plt.figure(figsize=(15,15))
    
    for i,col in enumerate(colonnes,1):
        ax = fig.add_subplot(5,2,i)
        sm.qqplot(df[col], fit=True, line="45", ax=ax)
        ax.set_title("qq-plot entre la variable {} et une loi normale".format(col))
        
    plt.tight_layout(pad = 4)
    fig.suptitle("Diagramme quantile-quantile")
    plt.show()
        
def test_normalite(df, colonnes,level):
    ''' Calcul les différents tests de normalité pour chacune des variables passées en paramètres'''
    for col in colonnes:
        print("Tests de normalité pour la variable {}.".format(col))
        tests = [shapiro, anderson, normaltest, kstest]
        index = ['Shapiro Wilk','Anderson-Darling',"K2 de D'Agostino",'Kolmogorov-Smirnov']
        tab_result = pd.DataFrame(columns=['Stat','p-value','Resultat'], index = index)
    
        for i, fc in enumerate(tests):
            if fc==anderson:
                result = fc(df[col])
                tab_result.loc[index[i],'Stat'] = result.statistic
                if result.statistic < result.critical_values[2]:
                    tab_result.loc[index[i],'Resultat'] = 'H0'
                if result.statistic > result.critical_values[2]:
                    tab_result.loc[index[i],'Resultat'] = 'H1'
                    
            elif fc==kstest:
                stat, p = fc(df[col],cdf='norm')
                tab_result.loc[index[i],'Stat'] = stat
                tab_result.loc[index[i],'p-value'] = p
                if p < level:
                    tab_result.loc[index[i],'Resultat'] = 'H1'
                if p > level:
                    tab_result.loc[index[i],'Resultat'] = 'H0'
            
            else :
                stat, p = fc(df[col])
                tab_result.loc[index[i],'Stat'] = stat
                tab_result.loc[index[i],'p-value'] = p
                if p < level:
                    tab_result.loc[index[i],'Resultat'] = 'H1'
                if p > level:
                    tab_result.loc[index[i],'Resultat'] = 'H0'
    
        print(tab_result)
        print("-"*70)
        
def bar_plot(df,colonnes):
    ''' Affiche les bar plots pour chaque variable renseignée.'''
    fig = plt.figure(figsize=(15,15))
    for i, col in enumerate(colonnes,1):
        ax = fig.add_subplot(2,2,i)
        plt.title(col)
        count = df[col].value_counts()
        count.plot(kind="bar", ax=ax)
        plt.xticks(rotation=90, ha='right')
        ax.set_title(col)
    plt.tight_layout(pad = 4)
    fig.suptitle('Bar plots pour chaque variable qualitative')
    plt.show()
    
def occurrence_mot(df,colonnes,nb_occur):
    '''Affiche un barplot avec les 15 mots ayant le plus d'occurrences pour chacune des variables renseignées.'''
    for col in colonnes :
        occur = collections.Counter(df[col].tolist()).most_common(nb_occur)
        df_occur = pd.DataFrame(occur, 
                                columns = ["Mot", "Nombre d'occurrences"]).sort_values(by="Nombre d'occurrences", 
                                                                                        ascending=False)

        sns.set(style="whitegrid")
        plt.figure(figsize=(15, 10))
        sns.barplot(y = 'Mot', x="Nombre d'occurrences", data = df_occur)
        plt.title("{} mots avec le plus d'occurrences dans {}".format(nb_occur,col), size=20)
        plt.show()

def pie_plot(df,colonnes):
    '''Affiche un pie plot présentant la répartition de la variable renseignée.'''
    for col in colonnes :
        labels = list(df[col].value_counts().sort_index().index.astype(str))
        count = df[col].value_counts().sort_index()
        
        plt.figure(figsize=(10, 10))
        plt.pie(count,autopct='%1.2f%%')
        plt.title('Répartition de {}'.format(col), size = 20)
        plt.legend(labels)
        plt.show()

def scatter_plot(df,colonnes,var_comparaison, largeur, longueur):
    ''' Affiche le scatter plot des variables quantitatives.'''
    fig = plt.figure(figsize=(15,15))
    for i,col in enumerate(colonnes,1):
        X = df[[var_comparaison]]
        Y = df[col]
        X = X.copy()
        X['intercept'] = 1.
        result = sm.OLS(Y, X).fit()
        a,b = result.params[var_comparaison],result.params['intercept']
        equa = "y = " + str(round(a,2)) + " x + " + str(round(b,0))

        ax = fig.add_subplot(longueur,largeur,i)
        plt.scatter(x=df[var_comparaison], y=df[col])        
        plt.plot(range(-15,41),[a*x+b for x in range(-15,41)],label=equa,color='red')
        ax.set_xlabel(xlabel=var_comparaison)
        ax.set_ylabel(ylabel=col)
        plt.legend()
    plt.tight_layout(pad = 4)
    fig.suptitle("Scatter plot des variables quantitatives")
    plt.show()
    
def pair_plot(df,nom,var_quali=None):
    ''' Affiche le pair plot des variables quantitatives.'''
    sns.pairplot(data=df,hue=var_quali)
    plt.savefig(nom)
    plt.show()
      
def pair_plot_reg(df,colonnes,nom):
    ''' Affiche le pair plot des variables quantitatives ainsi que la regression.'''
    sns.pairplot(data=df[colonnes],kind='reg', diag_kind='kde', plot_kws={'line_kws':{'color':'red'}})
    plt.savefig(nom)
    plt.show()

def heat_map(df_corr):
    '''Affiche la heatmap '''
    plt.figure(figsize=(15,10))
    sns.heatmap(df_corr, annot=True, linewidth=.5)
    plt.title("Heatmap")

def tests_corr(df,colonnes,var_comparaison):
    ''' Calcul les différents tests de corrélation pour chacun des couples de variables passés en paramètres'''
    for col in colonnes:
        print("Tests de corrélation pour la variable {} par rapport à la variable {}.".format(col,var_comparaison))
        tests = [pearsonr, spearmanr, kendalltau]
        index = ['Pearson', 'Spearman', 'Kendall']
        tab_result = pd.DataFrame(columns=['Stat','p-value'], index = index)
    
        for i, fc in enumerate(tests):
            stat, p = fc(df[col],df[var_comparaison])
            tab_result.loc[index[i],'Stat'] = stat
            tab_result.loc[index[i],'p-value'] = p
        display(tab_result)
        print("-"*100)
        
def boxplot_relation(df,colonnes,var_comparaison,longueur,largeur, nom, ordre=None,outliers=True,option=False):
    '''Affiche les boxplot des colonnes en fonctions de var_comparaison.'''
    fig = plt.figure(figsize=(30,30))
    for i,col in enumerate(colonnes,1):
        ax = fig.add_subplot(longueur,largeur,i)
        sns.boxplot(x=df[var_comparaison],y=df[col], ax=ax, order=ordre, showfliers = outliers)
        if option:
            plt.xticks(rotation=90, ha='right')
    fig.suptitle('Boxplot pour chaque variable quantitative en fonction de {}'.format(var_comparaison))
    plt.tight_layout(pad = 4)
    plt.savefig(nom)
    plt.show()
    
def violinplot_relation(df,colonnes,var_comparaison,longueur,largeur, ordre=None,outliers=True,option=False):
    '''Affiche les violinplot des colonnes en fonctions de var_comparaison.'''
    fig = plt.figure(figsize=(30,30))
    for i,col in enumerate(colonnes,1):
        ax = fig.add_subplot(longueur,largeur,i)
        sns.violinplot(x=df[var_comparaison],y=df[col], ax=ax, order=ordre, showfliers = outliers)
        if option:
            plt.xticks(rotation=90, ha='right')
    fig.suptitle('Violin plot pour chaque variable quantitative en fonction de {}'.format(var_comparaison))
    plt.tight_layout(pad = 4)
    plt.show()
    
def Kruskall_Wallis_test(df,colonnes, var_comparaison,level,test):
    '''Test alternatif de l'ANOVA: Kruskall-Wallis.'''
    
    for col in colonnes:
        group = df.groupby(var_comparaison)[col]
        ser = [gr[1].values for gr in group]

        if test=='Kruskall-Wallis':
            print("Test de Kruskall-Wallis pour la variable {} par rapport à la variable {}.".format(col,var_comparaison))
            stat, p = kruskal(*ser)
            print('stat={:.3f}, p-value={:.5f}'.format(stat, p))
        if test == 'ANOVA':
            print("Test de l'ANOVA pour la variable {} par rapport à la variable {}.".format(col,var_comparaison))
            stat, p = f_oneway(*ser)
            print('stat={:.3f}, p-value={:.5f}'.format(stat, p))

        
        if p < level:
            print('H1: rejet de H0')
        else:
            print('H0')
        print("-"*100)
        
def contingence_tab(df,var1,var2):
    data_crosstab = pd.crosstab(df[var1],df[var2], margins = False)
    return data_crosstab

def test_chi2(df, var1, var2, alpha=0.05):
    '''Test de Chi-2 pour 2 variables qualitatives.'''
    print("Test d'indépendance Chi-2 entre {} et {}".format(var1,var2))
    tab_cont = contingence_tab(df, var1, var2)
    results = chi2_contingency(tab_cont)
    print("stat = {}\np-value = {}\ndegrees of freedom = {}".format(results[0], results[1], results[2]))
    if results[1] <= alpha:
        print('Variables non indépendantes (H0 rejetée) car p = {} <= alpha = {}'.format(results[1], alpha))
    else:
        print('H0 non rejetée car p = {} > alpha = {}'.format(results[1], alpha))
    print("-"*70)
    
def eboulis(pca):
    '''Réalise un éboulis de valeurs propres'''
    scree = pca.explained_variance_ratio_*100
    scree_cum = scree.cumsum()
    
    plt.figure(figsize=(10,5))
    plt.bar(np.arange(len(scree)) + 1, scree)
    plt.plot(np.arange(len(scree)) + 1, scree_cum,c="red",marker='o')
    #plt.axhline(100/10,0, color='black')
    plt.xlabel("Rang de l'axe d'inertie")
    plt.ylabel("Pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)
    
def scree_plot(pca):
    '''Réalise un scree plot'''
    scree = pca.explained_variance_
    
    plt.figure(figsize=(12,8))
    plt.plot(range(1,len(scree)+1),scree,marker='.',mec='r',mew=2)
    plt.title('Scree Plot',fontsize=15)
    plt.ylabel("Valeurs propres",fontsize=12)
    plt.xlabel("Composantes",fontsize=12)
    plt.show()
    
def correlation_graph(pca, x, y, features) : 
    '''Affiche le graphe des correlations'''
    
    fig, ax = plt.subplots(figsize=(10, 9))
 
    for i in range(0, pca.components_.shape[1]):

        ax.arrow(0,0, pca.components_[x, i],  pca.components_[y, i], head_width=0.05, head_length=0.07, color ='grey')

        plt.text(pca.components_[x, i] + 0.05, pca.components_[y, i] + 0.05, features[i])
                
    # Affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    # Nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
    plt.ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))

    plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1))

    # Le cercle 
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))

    # Axes et display
    plt.axis('equal')
    plt.show(block=False)
    
def plan_factoriel(X_projected, x, y , pca, labels = None):
    '''Affiche la projection des individus'''
    X_arr = np.array(X_projected)
    fig, ax = plt.subplots(1, 1, figsize=[10,8])
 
    # Les points 
    plt.scatter(X_arr[:, x], X_arr[:, y], alpha=1)

    v1 = str(round(100*pca.explained_variance_ratio_[x]))  + " %"
    v2 = str(round(100*pca.explained_variance_ratio_[y]))  + " %"
    ax.set_xlabel(f'F{x+1} {v1}')
    ax.set_ylabel(f'F{y+1} {v2}')

   # Valeur x max et y max
    x_max = np.abs(X_arr[:, x]).max() *1.1
    y_max = np.abs(X_arr[:, y]).max() *1.1

    # On borne x et y 
    ax.set_xlim(left=-x_max, right=x_max)
    ax.set_ylim(bottom= -y_max, top=y_max)

    # Affichage des lignes horizontales et verticales
    plt.plot([-x_max, x_max], [0, 0], color='grey', alpha=0.5)
    plt.plot([0,0], [-y_max, y_max], color='grey', alpha=0.5)

    # Affichage des labels des points
    if labels is not None : 
        for i,(_x,_y) in enumerate(X_arr[:,[x,y]]):
            plt.text(_x, _y+0.05, labels[i], fontsize='14', ha='center',va='center') 

    # Titre et display
    plt.title(f"Projection des individus (sur F{x+1} et F{y+1})")
    plt.show()
    
def graph_3D(X_projected,composantes):
    x = X_projected[:,composantes[0]]
    y = X_projected[:,composantes[1]]
    z = X_projected[:,composantes[2]]
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, c=x, marker='o', alpha=0.8)
    plt.title("Représentation  3D")
    ax.set_xlabel('PC'+str(composantes[0]))
    ax.set_ylabel('PC'+str(composantes[1]))
    ax.set_zlabel('PC'+str(composantes[2]))
    plt.legend
    plt.show()
    

