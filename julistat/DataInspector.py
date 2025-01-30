import sys, gc, warnings, pandas as pd, numpy as np, shap
from joblib import dump, load, Parallel, delayed
from Quietly import Quietly
from datetime import datetime
import statsmodels.formula.api as smf
from itertools import combinations
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2

warnings.filterwarnings("ignore", category=RuntimeWarning)

class DataInspector():
    def __init__(self):
        self.parallel = False
        self.categorical_var = None
        self.varlist = []
        self.target = None
        self.data = None
        

    def entropy(self, data:pd.DataFrame='', target:str='', varlist:list=[], parallel=False):
        """A function for calculations like entropy and correlation direction,
        requiring data, a target variable, and a list of variables. It may be 
        executed in parallel.

        1. Data: Data frame with the information necessary to process.
        2. Target: Objective variable or variable to be explained, must contain ones or zeros.
        3. List with the variables to which the entropy will be calculated.
        4. Boolean to indicate whether it will work in parallel or not.
        """

        #Initializing with predefined variables or those loaded in the function
        data = self.data if not data else data
        target = self.target if not target else target
        varlist = self.varlist if not varlist else varlist
        parallel = self.parallel if not parallel else parallel
        #Ending early
        if not isinstance(data, pd.DataFrame) or data.empty or not target or not varlist:
            print("Incomplete information for calculations")
            return
        #Initializing variables
        info = {}
        variables = []
        cuttings = []
        left_entropies = []     
        right_entropies = []
        weighted_entropies = []
        directions = []
        vifs = []

        prev_length = 0

        #Calculations for each variable
        start_time = datetime.now()
        amount = len(varlist)
        count = 0
        for var in varlist:
            count += 1
            #Filtering to remove nulls
            filtered_data = data.loc[:, [target, var]].dropna(subset=[var])
            if filtered_data.shape[0] == 0:
                print(var, ": Without useful information to continue with the analysis")
                continue
            #Printing message
            current_time = datetime.now().strftime("%H:%M:%S")
            message = f"[{current_time}] Processing variable: {count}/{amount}. {var}"
            padding = " "*max(prev_length - len(message), 0)
            sys.stdout.write(f"\r{message}{padding}")
            sys.stdout.flush()
            prev_length = len(message)
            #Convert to numpy arrays
            X1 = filtered_data.drop(target, axis=1).values
            y = filtered_data[target].values
            
            #Calculating entropia
            clf = []
            clf = DecisionTreeClassifier( criterion='entropy' , max_depth = 1 )
            clf.fit(X1, y)
            #Calculating entropy
            cutting = clf.tree_.threshold[0]
            population = clf.tree_.n_node_samples[0]
            left_entropy = clf.tree_.impurity[1]
            right_entropy = clf.tree_.impurity[2]
            left_sample = clf.tree_.n_node_samples[1]
            right_sample = clf.tree_.n_node_samples[2]
            weighted_entropy = left_entropy*(left_sample/population) + \
                               right_entropy*(right_sample/population)
            #Calculating VIF
            X2 = data[varlist]
            X2 = X2.fillna(X2.mean())
            vif = variance_inflation_factor(X2, X2.columns.get_loc(var))
            #Calculating direction
            formula = f"{target} ~ {var}"
            with Quietly():
                modelo = smf.logit( formula = formula, data = filtered_data ).fit()
            direction = "Decreases" if modelo.params[var] <0 else "Increase"

            #Storing information
            variables += [var]
            cuttings += [cutting]
            left_entropies += [left_entropy]
            right_entropies += [right_entropy]
            weighted_entropies += [weighted_entropy]
            directions += [direction]
            vifs += [vif]

            #Cleaning
            del X1, X2, y, filtered_data, clf, weighted_entropy, vif, modelo
            gc.collect()

        #Storing information
        info['Variable'] = variables
        info['Cutting'] = cuttings
        info['Left Entropy'] = left_entropies
        info['Right Entropy'] = right_entropies
        info['Weighted Entropy'] = weighted_entropies
        info['Direction'] = directions
        info['VIF'] = vifs

        end_time = datetime.now()
        total_time = end_time - start_time
        #Printing end message
        current_time = datetime.now().strftime("%H:%M:%S")
        message = f"Processed variables: {count}/{amount}. Time {total_time}"
        padding = " "*max(prev_length - len(message), 0)
        prev_length = len(message)
        sys.stdout.write(f"\r{message}{padding}")
        sys.stdout.flush()
        return pd.DataFrame(info)

    
    def groupAnalizer(self, data:pd.DataFrame='', target:str=''
                      , varlist:list=[], categorical_var:str=''
                      , combinatory=False, parallel=False):
        """Function to determine and study possible groupings. 
        For this purpose, changes in the slope are studied. 
        Allows you to study independent groups or combinations.
        The group must arrive in a categorical variable and not in a dummy variable.
        
        1. Data: Data frame with the information necessary to process.
        2. Target: Objective variable or variable to be explained, must contain ones or zeros.
        3. List with the variables to which the AIC will be calculated.
        4. Categorical variable. The analysis by group will be carried out for each category indicated in the variable.
        5. Boolean to indicate if you want to review the combination of categories
        6. Boolean to indicate whether it will work in parallel or not.
        """

        #Initializing with predefined variables or those loaded in the function
        data = self.data if not data else data
        target = self.target if not target else target
        varlist = self.varlist if not varlist else varlist
        categorical_var = self.categorical_var if not categorical_var else categorical_var
        parallel = self.parallel if not parallel else parallel
        
        #Ending early
        if not isinstance(data, pd.DataFrame) or data.empty or not target or not varlist:
            print("Incomplete information for calculations")
            return

        #Formatting category variable
        data[categorical_var] = data[categorical_var].map(str)
        data[categorical_var] = data[categorical_var].fillna("indet.")

        #First info
        counts = data[categorical_var].value_counts()
        #Most frequent category
        most_freq_cat = counts.idxmax()
        #Determining the groups
        groups = counts.index.tolist()
        groups.remove(most_freq_cat)
        
        #Generate combinations if combinatory is True
        if combinatory:
            all_combinations = [list(combinations(groups, r)) for r in range(1, len(groups) + 1)]
            all_combinations = [item for sublist in all_combinations for item in sublist]
        else:
            all_combinations = [(group, ) for group in groups]  #Single categories

        #Initializing variables
        variables = []
        pvalues_var = []
        aic_base = []
        aics_var = []

        prev_length = 0
        #Calculations for each variable
        start_time = datetime.now()
        amount = len(varlist)
        count = 0
        for var in varlist:
            count += 1
            #Filtering to remove nulls
            filtered_data = data.loc[:, [target, var, categorical_var]].dropna(subset=[var])
            if filtered_data.shape[0] == 0:
                print(var, ": Without useful information to continue with the analysis")
                continue
            #Printing message
            current_time = datetime.now().strftime("%H:%M:%S")
            message = f"[{current_time}] Processing variable: {count}/{amount}. {var}"
            padding = " "*max(prev_length - len(message), 0)
            sys.stdout.write(f"\r{message}{padding}")
            sys.stdout.flush()
            prev_length = len(message)
            
            #Starting iteration for group variable.
            pvalues = []; aics = []
            
            #Calculating structural changes
            try:
                #Reduced model
                reduced_formula = f"{target} ~ {var}"
                reduced_model = smf.logit(formula=reduced_formula, data=filtered_data).fit(disp=False)
            except Exception as e:
                #If the base does not converge then there is no way to compare with
                continue
            else:
                #Akaike Information Criterion (AIC)
                aic_base += [reduced_model.aic]

            for combination in all_combinations:
                dummy_name = f'dummy_{"_".join(combination)}'
                delta_name = f'delta_{"_".join(combination)}'

                #Creating dummy and delta variables
                filtered_data[dummy_name] = filtered_data[categorical_var].apply(lambda x: 1 if x in combination else 0)
                filtered_data[delta_name] = filtered_data[var] * filtered_data[dummy_name]

                #Creating dummy and delta variables
                filtered_data[dummy_name] = filtered_data[categorical_var].apply(lambda x: 1 if x in combination else 0)
                filtered_data[delta_name] = filtered_data[var] * filtered_data[dummy_name]

                #Calculating structural changes
                try:
                    #Complete model
                    complete_formula = f"{target} ~ {var} + {dummy_name} + {delta_name}"
                    complete_model = smf.logit(formula=complete_formula, data=filtered_data).fit(disp=False)
                except Exception as e:
                    #Not quantifiable
                    pvalues += [ 'No quant.' ]
                    aics += [np.inf]
                else:
                    #Calculation of the likelihood ratio test
                    llr_stat = 2 * (complete_model.llf - reduced_model.llf)  #Difference of log-likelihoods
                    df_diff = complete_model.df_model - reduced_model.df_model  #Difference in degrees of freedom
                    pvalue = complete_model.pvalues[f'{delta_name}']
                    pvalue_global = chi2.sf(llr_stat, df_diff)

                    #Structural change (SC), No Structural change (No SC)
                    if pvalue_global <= 0.001: #To reduce the risk of false positives
                        change = "Strong SC"
                    elif pvalue <= 0.001:
                        change = "SC"
                    else:
                        change = "No SC"
                    #Saving
                    pvalues += [ change ]
                    aics += [complete_model.aic] 
                    
            #Storing information. Variable results
            variables += [var]
            pvalues_var += [pvalues]
            aics_var += [aics]

            #Cleaning
            del filtered_data, reduced_formula, reduced_model, complete_formula, complete_model, change
            gc.collect()

        #Storing information
        end_time = datetime.now()
        total_time = end_time - start_time
        #Printing end message
        current_time = datetime.now().strftime("%H:%M:%S")
        message = f"Processed variables: {count}/{amount}. Time {total_time}"
        padding = " "*max(prev_length - len(message), 0)
        prev_length = len(message)
        sys.stdout.write(f"\r{message}{padding}")
        sys.stdout.flush()
        #Creating Dataframe with results
        columns = [f"{categorical_var}_{'_'.join(comb)}" for comb in all_combinations]
        #Results - pvalues
        PVALUES = pd.DataFrame(pvalues_var, index=variables, columns=columns )
        PVALUES.reset_index(inplace = True, names = "Variable")
        PVALUES.insert(1, "Base category", most_freq_cat)
        #Results - AIC
        AICS = pd.DataFrame(aics_var, index=variables, columns=columns )
        AICS.reset_index(inplace = True, names = "Variable")
        AICS.insert(1, "Base category", most_freq_cat)
        AICS.insert(2, "Base AIC", aic_base)
        return PVALUES, AICS


    def aicAnalizer(self, data:pd.DataFrame='', varlist:list=[]):
        """Transforms the results achieved with groupAnalyzer. 
        Calculate difference and apply rules.
        1. Data: Data frame with the information necessary to process.
        2. List with the variables to which the AIC analizer will be work whit.
        """
        vars = data.columns
        #Calculating best model
        min = data[varlist].min(axis=1).to_numpy()
        #Identify the column that has the minimum value
        result = (data[varlist]==min[:, np.newaxis]).astype(int)
        return  result

        
    def mutualIC(self, data:pd.DataFrame='', target:str='', varlist:list=[], vif=True, parallel=False):
        """A function for calculating Mutual Information Clasifier
        1. Data: Data frame with the information necessary to process.
        2. Target: Objective variable or variable to be explained, must contain ones or zeros.
        3. List with the variables to which the Multual Information will be calculated.
        4. Boolean to indicate if you want to calculate VIF
        5. Boolean to indicate whether it will work in parallel or not."""

        #Initializing with predefined variables or those loaded in the function
        data = self.data if not data else data
        target = self.target if not target else target
        varlist = self.varlist if not varlist else varlist
        parallel = self.parallel if not parallel else parallel
        #Ending early
        if not isinstance(data, pd.DataFrame) or data.empty or not target or not varlist:
            print("Incomplete information for calculations")
            return
        #Initializing variables
        info = {"Variable": [], "Mutual Information": [], "Imputed Percentage": []}

        prev_length = 0
        #Calculations for each variable
        start_time = datetime.now()
        amount = len(varlist)

        #Printing message
        current_time = datetime.now().strftime("%H:%M:%S")
        message = f"[{current_time}] Processing {amount} variables."
        padding = " "*max(prev_length - len(message), 0)
        sys.stdout.write(f"\r{message}{padding}")
        sys.stdout.flush()
        prev_length = len(message)
        
        #Convert to numpy arrays
        X1 = data[varlist]
        y = data[target].values

        #Imputing
        imputed_percentages = {}
        for column in varlist:
            missing_count = X1.loc[:, column].isna().sum()
            total_count = len(X1[column])
            imputed_percentages[column] = (missing_count / total_count) if total_count > 0 else 0
            # Imputar valores nulos con la media
            X1.loc[:, column] = X1.loc[:, column].fillna(X1.loc[:, column].mean())

        #Calculating mutual information
        mi_scores = mutual_info_classif(X1, y)

        #Calculating VIF
        if vif:
            vif_values = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]
            info["VIF"] = vif_values

        #Storing information
        for var, mi_score in zip(varlist, mi_scores):
            info["Variable"].append(var)
            info["Mutual Information"].append(mi_score)
            info["Imputed Percentage"].append(imputed_percentages[var])
        
        end_time = datetime.now()
        total_time = end_time - start_time
        #Printing end message
        current_time = datetime.now().strftime("%H:%M:%S")
        message = f"Processed {amount} variables. Time {total_time}"
        padding = " "*max(prev_length - len(message), 0)
        prev_length = len(message)
        sys.stdout.write(f"\r{message}{padding}")
        sys.stdout.flush()
        return pd.DataFrame(info)

    
    #Función para calcular valores SHAP para un subconjunto de datos
    #Esta funcion es la que se paralelizará
    def calculate_shap_values(self, batch, explainer):
        return explainer.shap_values(batch)

    #Dividir el conjunto de datos en bloques para paralelismo
    #Esta funcion usa la anterior como funcion hija
    def parallel_shap(self, X, feats_train, modelo, n_jobs=-1, batch_size=1000):
        n_batches = (len(X) + batch_size - 1) // batch_size  #Redondeo hacia arriba
        batches = np.array_split(X, n_batches)
        
        #Creando un explicador SHAP
        explainer = shap.TreeExplainer(modelo)
        #Ejecutar en paralelo
        shap_values = Parallel(n_jobs=n_jobs)(
            delayed(self.calculate_shap_values)(batch, explainer) for batch in batches
        )
        #Combinando resultados
        if isinstance(shap_values[0], list):  #Clasificación multiclase
            resultado = [np.vstack([sv[i] for sv in shap_values]) for i in range(len(shap_values[0]))]
        else:  #Regresión o clasificación binaria
            resultado = np.vstack(shap_values)
        #Verificando que el resultado a promediar se el indicado
        if isinstance(resultado, list):  #Si es clasificación multiclase
            shap_values_class = resultado[1]  #Usamos el primer conjunto de valores SHAP de la clase positiva
        else:
            shap_values_class = resultado
        #Obteniendo la importancia media por característica (promedio de valores absolutos)
        importance = np.mean(np.abs(shap_values_class), axis=0)
        #Creando un DataFrame para mostrar las importancias
        importance_df = pd.DataFrame({
            'Feature': feats_train,
            'Importance': importance
        })
        #Ordenando las importancias de mayor a menor
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        return importance_df.reset_index(drop=True)