# Machine-Learning-Framework
A general machine learning framework by Tianxiang Zhou in a streamlined manner.
This framework was built upon the KDD dataset with a CV score of .885.

## 1. EDA and Data Cleaning

Some useful takeways:
- Find duplicated columns
```python
def find_duplicate_columns(df = df):
    dup_cols = {}
    for i, c1 in enumerate(tqdm(df.columns)):
        for c2 in df.columns[i+1:]:
            if c2 not in dup_cols and np.all(df[c1] == df[c2]):
                dup_cols[c2] = c1
    return dup_cols
df.drop(find_duplicate_columns(df).keys, axis=1, inplace=True
```
- Dropping all constant columns(column that contains one unique value)
```python
def drop_all_constant(df):
    feature_counts = df.nunique(dropna=False)
    print(feature_counts.sort_values()[:5])
    try:
        constant_features = feature_counts[feature_counts==1].index.tolist()
        return constant_features        
    except:
        print('Opps! Something does not seem right.')
        return None
```
- Check if we have any missing values in the dataframe and generate plot
```python
def check_missing_values(df):
    if df.isna().sum().sum() == 0:
        print('There is no missing value in your dataset! Congrats!')
        return None
    else:
        missing = df.isna().sum().sort_values(ascending=False)
        print('The following columns have missing values: ')
        m_cols = []
        for i, j in zip(missing.index.tolist(), missing):
            if j == 0:
                break
            else:
                m_cols.append(i)
                print(i)
        msno.matrix(df.loc[:,m_cols])
        return np.array(m_cols)
```
- Plot top 10 important features from a Random Forest Model
```python
def top_ten_rf(df, rfc): 
    importances = rfc.feature_importances_
    indices = np.argsort(importances)[:10]
    print("Feature ranking:")
    for i in range(len(indices)):
        print("%d. feature %d (%f)" % (i + 1, indices[i], importances[indices[i]]))
    print("Feature ranking:")
    for i in tqdm(range(len(indices))):
        print(i + 1, df.columns[i], ":", importances[indices[i]])
    plt.figure(figsize=(30,9))
    plt.bar(range(10), importances[indices], color="r", align="center")
    plt.xticks(range(10),df.columns[indices]);
```
## 2. Create a robust kFOLD
- Create a Stratified 5-fold
```python
def create_strat_kfold(df):
    df['kfold'] = -1 #default it to -1
    df = df.sample(frac=1).reset_index(drop=True) #resample and reset index
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=128)
    for fold, (trainindex ,validationindex) in enumerate(kf.split(X=df, y=df.label.values)):
        df.loc[validationindex, 'kfold']=fold
    return df
```
- One step cross-validation generator for all types of machine learning problems
```python
"""
- -- binary classification
- -- multi class classification
- -- multi label classification
- -- single column regression
- -- multi column regression
- -- holdout
"""

class CrossValidation:
    def __init__(
            self,
            df, 
            target_cols,
            shuffle, 
            problem_type="binary_classification",
            multilabel_delimiter=",",
            num_folds=5,
            random_state=42
        ):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle,
        self.random_state = random_state
        self.multilabel_delimiter = multilabel_delimiter

        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
        
        self.dataframe["kfold"] = -1
    
    def split(self):
        if self.problem_type in ("binary_classification", "multiclass_classification"):
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception("Only one unique value found!")
            elif unique_values > 1:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds, 
                                                     shuffle=False)
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe[target].values)):
                    self.dataframe.loc[val_idx, 'kfold'] = fold

        elif self.problem_type in ("single_col_regression", "multi_col_regression"):
            if self.num_targets != 1 and self.problem_type == "single_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            if self.num_targets < 2 and self.problem_type == "multi_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            kf = model_selection.KFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe)):
                self.dataframe.loc[val_idx, 'kfold'] = fold
        
        elif self.problem_type.startswith("holdout_"):
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = int(len(self.dataframe) * holdout_percentage / 100)
            self.dataframe.loc[:len(self.dataframe) - num_holdout_samples, "kfold"] = 0
            self.dataframe.loc[len(self.dataframe) - num_holdout_samples:, "kfold"] = 1

        elif self.problem_type == "multilabel_classification":
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            targets = self.dataframe[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=targets)):
                self.dataframe.loc[val_idx, 'kfold'] = fold

        else:
            raise Exception("Problem type not understood!")

        return self.dataframe


if __name__ == "__main__":
    df = pd.read_csv("train_multilabel.csv")
    cv = CrossValidation(df, shuffle=True, target_cols=["attribute_ids"], 
                         problem_type="multilabel_classification", multilabel_delimiter=" ")
    df_split = cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts())
```

## 3. Training and Dispatching
- run model and save model by fold number as generated above
```python
def run_model(fold, model):
    df = pd.read_csv('TRAIN_CLEANED_FOLDS.csv')
    #extract training set and validation set
    train_df = df[df.kfold != 0]
    valid_df = df[df.kfold == 0]
    #extract labels for each set
    y_train = train_df.label.values
    y_valid = valid_df.label.values
    #drop unnecessary columns from training set and validation set
    train_df = train_df.drop(['label','kfold'], axis=1)
    valid_df = valid_df.drop(['label','kfold'], axis=1)
    #maintain the order of the variables, maybe not needed
    valid_df = valid_df[train_df.columns]
    #now we are ready to train
    #---------------------------------------------------------------
    #training, you can use any model you want
    rfc = model
    rfc.fit(train_df, y_train)
    preds = rfc.predict(valid_df)
    print('the score you had for this fold is: ', metrics.accuracy_score(preds, y_valid))
    
    model_name = str(model)[:10]
    joblib.dump(rfc,f'models/{model_name}_{fold}.pkl') #save the model to a pkl file
    joblib.dump(train_df.columns, f"models/{model_name}_{fold}_columns.pkl") #save the columns to a pkl file as well
```
## 4. Inference and Submission
- one-step predict and collect all test predictions from model created in each fold
```python
def predict(df): #input your testdf here
    predictions = []
    
    for FOLD in range(5):        
        clf = joblib.load(os.path.join('models/', f"RandomFore_{FOLD}.pkl"))
        cols = joblib.load(os.path.join('models/', f"RandomFore_{FOLD}_columns.pkl"))
        df = df[cols]
        preds = clf.predict(df)
        
        predictions.append(preds)
        
    return predictions
```

I demonstrated the whole **KAGGLE** competition pipeline using a random forest with a stratitfied k-fold. It's really easy to use and replicate. However, in real life data science, there are so many things that I haven't done, for instance, model maintenance and data aquisition. There are many other thing you can do to improve the score: feature engineering, stacking, bagging or just use stronger learner like GBDTs...
