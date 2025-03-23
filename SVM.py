import pandas as pd
import numpy as np
from scipy import stats
import re
import math
import nltk
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (roc_curve, auc, precision_recall_curve)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

project = input("input filename(caffe, incubator-mxnet, keras, pytorch, tensorflow):").strip()

def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F" 
                               u"\U0001F300-\U0001F5FF"  
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"  
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

nltk.download('stopwords')
from nltk.corpus import stopwords
NLTK_stop_words_list = stopwords.words('english')
final_stop_words_list = NLTK_stop_words_list

def remove_stopwords(text):

    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])

def clean_str(string):

    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

path = f'{project}.csv'
pd_all = pd.read_csv(path)
pd_all = pd_all.sample(frac=1, random_state=999)

pd_all['Title+Body'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
    axis=1
)

pd_tplusb = pd_all.rename(columns={
    "Unnamed: 0": "id",
    "class": "sentiment",
    "Title+Body": "text"
})
pd_tplusb.to_csv('Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])

datafile = 'Title+Body.csv'
REPEAT = 30
data = pd.read_csv(datafile).fillna('')
text_col = 'text'

original_data = data.copy()

data[text_col] = data[text_col].apply(remove_html)
data[text_col] = data[text_col].apply(remove_emoji)
data[text_col] = data[text_col].apply(remove_stopwords)
data[text_col] = data[text_col].apply(clean_str)

auc_values_SVM = []
auc_values_NB = []

for repeated_time in range(REPEAT):

    indices = np.arange(data.shape[0])
    train_index, test_index = train_test_split(
        indices, test_size=0.2, random_state=repeated_time
    )

    train_text = data[text_col].iloc[train_index]
    test_text = data[text_col].iloc[test_index]

    y_train = data['sentiment'].iloc[train_index]
    y_test = data['sentiment'].iloc[test_index]


    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1000
    )
    X_train = tfidf.fit_transform(train_text)
    X_test = tfidf.transform(test_text)


    clf_SVM = SVC(probability=True, random_state=repeated_time)
    clf = GaussianNB()

    params = {
        'var_smoothing': np.logspace(-12, 0, 13)
    }
    params_SVM = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto'],
        'class_weight': ['balanced']
    }

    pr_auc_scorer = make_scorer(average_precision_score, response_method = 'predict_proba')

    grid_SVM = GridSearchCV(
        clf_SVM,
        params_SVM,
        cv=5,
        scoring=pr_auc_scorer
    )
    grid = GridSearchCV(
        clf,
        params,
        cv=5,
        scoring='roc_auc'
    )
    X_train_dense = X_train.toarray()

    grid_SVM.fit(X_train_dense, y_train)
    best_clf_SVM = grid_SVM.best_estimator_
    best_clf_SVM.fit(X_train_dense, y_train)

    grid.fit(X_train_dense, y_train)
    best_clf = grid.best_estimator_
    best_clf.fit(X_train_dense, y_train)

    X_test_dense = X_test.toarray()
    y_scores = best_clf_SVM.predict_proba(X_test_dense)[:, 1]
    y_pred = best_clf.predict(X_test_dense)

    fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
    auc_val = auc(fpr, tpr)
    auc_values_NB.append(auc_val)

    fpr_SVM, tpr_SVM, _ = roc_curve(y_test, y_scores, pos_label=1)
    auc_val_SVM = auc(fpr_SVM, tpr_SVM)
    auc_values_SVM.append(auc_val_SVM)


final_auc_SVM = np.mean(auc_values_SVM)
final_auc_NB = np.mean(auc_values_NB)

before = np.array(auc_values_NB)
after = np.array(auc_values_SVM)

t_statistic, p_value = stats.ttest_rel(after, before)

alpha = 0.05
if p_value < alpha:
    print(f"p < {alpha}，Reject H0，the difference is statistically significant")
else:
    print(f"p ≥ {alpha}，Accept H0，the difference is statistically significant")

print(f"=== Results in  {project} ===")
print(f"Number of repeats:     {REPEAT}")

print(f"Average AUC of SVM:           {final_auc_SVM:.4f}")
print(f"Average AUC of NB:           {final_auc_NB:.4f}")