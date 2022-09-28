import pandas as pd
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

def loadDataset(Datasets: list, columnsToUse: list):
    #Load the first dataframe from the list
    df = pd.read_csv(Datasets[0], header=0, error_bad_lines=False)
    
    #If there are multiple dataframes then begin loading the next one and merge.
    if len(Datasets) > 2:
        for i in range(0,len(Datasets)):
            tempdf = pd.read_csv(Datasets[i], header=0, error_bad_lines=False)
            df = pd.concat([df, tempdf])
    
    #select only the specified columns
    df = df[columnsToUse]

    df=df.dropna()
    df = df.reset_index(drop=True)
    return df




def dataCleaning(df, textColumnName):
    #Lowercase text
    df['CleanText'] = df[textColumnName].str.lower() 
    #change abbreviations to full words
    df["CleanText"]=df[textColumnName].apply(lambda x:contractions(x))
    #Remove all non letters
    df["CleanText"]=df[textColumnName].apply(lambda x: ' '.join([re.sub("[^a-z]+",'', x) for x in nltk.word_tokenize(x)]))
    return df

def contractions(s):
    #change abbreviations to full words
    s = re.sub(r"won't", "will not",s)
    s = re.sub(r"would't", "would not",s)
    s = re.sub(r"could't", "could not",s)
    s = re.sub(r"\'d", " would",s)
    s = re.sub(r"can\'t", "can not",s)
    s = re.sub(r"n\'t", " not", s)
    s= re.sub(r"\'re", " are", s)
    s = re.sub(r"\'s", " is", s)
    s = re.sub(r"\'ll", " will", s)
    s = re.sub(r"\'t", " not", s)
    s = re.sub(r"\'ve", " have", s)
    s = re.sub(r"\'m", " am", s)
    return s



def removeStopWords(df):
    #Remove stopwords from a corpus
    from nltk.corpus import stopwords
    stop = stopwords.words("english")
    df["CleanText"]=df["CleanText"].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
    return df


def lemWords(df):
    #Lemmatize all words in a corpus
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    df["CleanText"]=df["CleanText"].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in nltk.word_tokenize(x)]))
    return df

def stemWords(df):
    #Lemmatize all words in a corpus
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    df["CleanText"]=df["CleanText"].apply(lambda x: ' '.join([stemmer.stem(word) for word in nltk.word_tokenize(x)]))
    return df


def overSampler(x, y):
    #Random Over Sampling for imbalanced training datasets
    from imblearn.over_sampling import RandomOverSampler,SMOTE, ADASYN
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(x, y)
    return X_resampled, y_resampled



def FitOversamplePlot(df):
    #Train Test Split
    X_train,X_test,Y_train, Y_test = train_test_split(df['CleanText'], df['Sentiment'], test_size=0.25, random_state=30)
    
    #Generate TF-IDF
    vectorizer= TfidfVectorizer()
    tf_x_train = vectorizer.fit_transform(X_train)
    tf_x_test = vectorizer.transform(X_test)
    #tf_x_train, Y_train = overSampler(tf_x_train,Y_train)
    
    #Create classifier
    clf = LinearSVC(random_state=0)
    #Train Classifier
    clf.fit(tf_x_train,Y_train)
    
    return clf, tf_x_test, Y_test
    

    
def EvaluateModel(clf, tf_x_test, Y_test):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    
    
    #Generate model predictions
    y_test_pred=clf.predict(tf_x_test)
        
    print(classification_report(Y_test, y_test_pred,output_dict=True))
    
    def show_confusion_matrix(confusion_matrix):
        hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
        hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30,ha='right')
        plt.ylabel('True sentiment')
        plt.xlabel('Predicted sentiment')  

    #Create confusion_matrix object
    cm = confusion_matrix(Y_test, y_test_pred)
    encode_map = {'negative': "negative",'neutral': "neutral",'positive': "positive"}
    
    df_cm = pd.DataFrame(cm, index=encode_map, columns=encode_map)
    #Display confusion matrix
    show_confusion_matrix(df_cm)

def SaveModel(clf, filename):
    import pickle
    #Save machine learning model to specified location
    pickle.dump(clf, open(filename, 'wb'))


if __name__ == "__main__":
    datasets =['test_sent_emo.csv','train_sent_emo.csv','dev_sent_emo.csv']
    columnsToUse = ["Utterance","Sentiment"]
    
    df = loadDataset(datasets, columnsToUse)
    df = dataCleaning(df, "Utterance")
    #df = removeStopWords(df)
    df = lemWords(df)
    clf, tf_x_test, Y_test = FitOversamplePlot(df)
    EvaluateModel(clf, tf_x_test, Y_test)
    #SaveModel(clf, "SentimentAnalysisModel")
    
    
    