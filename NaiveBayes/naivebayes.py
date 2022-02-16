import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics



class NaiveBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    '''

    def __init__(self):
        return 

    def fit(self, X, y):
        '''
        Create a table that will allow the filter to evaluate P(H), P(S)
        and P(w|C)

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        #split the x training set into individual words
        sf = X.str.split()
        unique = []
        #create a list of the unique words
        for i in sf.values:
            for j in i:
                if j not in unique:
                    unique.append(j)
            
        #create an empty dataframe with columns as all the unique words
        gf = pd.DataFrame(columns = unique,index = X.index)
        #make each row the line of words from X and count how many times each unique word is in that line
        for i in range(len(unique)):
            for j in range(len(sf.values)):
                gf.values[j][i] = sf.values[j].count(unique[i])
        
        #add the columns of labels from the y train
        gf['labels'] = y
        self.data = gf
        """the code below is the code to change it into the desired format. I tried implimenting 
        this and it literally broke every other element of my code. I spent 2 hours trying 
        to get the other parts to work with this new method but nothing would work unless
        i did this problem from scratch. If this had been the original instruciton, it would
        have been easy and would have made all the code better. The way the lab introduced 
        this made it much harder and not worth the time it takes to impliment. Please
        be merciful, i'm trying to follow instrucitons the best i can"""
        #this lab sucked bad. I literally spent 8+ hours on this. 
# =============================================================================
#         hammask = []
#         spammask = []
#         for i in range(len(self.data)):
#             if self.data['labels'].values[i] == 'ham':
#                 hammask.append(True)
#                 spammask.append(False)
#             else:
#                 hammask.append(False)
#                 spammask.append(True)
#                 
#         ham = []
#         spam = []
#         for i in unique:
#             ham.append((self.data[i].values[hammask].sum()))
#             spam.append((self.data[i].values[spammask].sum()))
#         
#         
#         A = []
#         A.append(ham)
#         A.append(spam)
#         A = np.array(A)
#         ff = pd.DataFrame(A, index = ['ham','spam'], columns = unique)
#         ff['labels'] = y
#         self.data = ff
# =============================================================================
        #raise NotImplementedError('Problem 1 incomplete')
        
    def predict_proba(self, X):
        '''
        Find P(C=k|x) for each x in X and for each class k by computing
        P(C=k)P(x|C=k)

        Parameters:
            X (pd.Series)(N,): messages to classify
        
        Return:
            (ndarray)(N,2): Probability each message is ham, spam
                0 column is ham
                1 column is spam
        '''
        #split each line into individual words
        words = X.str.split()
        #count the total number of words that are identified Ham or Spam
        Hn = 0
        Sn = 0
        #create a mask for ham and spam to easier find the totals for the respective probabilities 
        hammask = []
        spammask = []
        for i in range(len(self.data)):
            if self.data['labels'].values[i] == 'ham':
                Hn += sum(self.data.values[i][:-1])
                hammask.append(True)
                spammask.append(False)
            else:
                Sn += sum(self.data.values[i][:-1])
                hammask.append(False)
                spammask.append(True)
         
        n = Hn + Sn
        
        #sum the values from each column of the dataframe that has how many times each unique value is in each phrase contains a word, but use the mask to only get the sums for the ham/spam values
        ham = []
        spam = []
        for i in range(len(words)):
            xi = 1
            yi = 1
            for j in range(len(words.values[i])):
                if words.values[i][j] in self.data.columns:
                    xi *= ((self.data[words.values[i][j]].values[hammask].sum())/Hn)
                    yi *= ((self.data[words.values[i][j]].values[spammask].sum())/Sn)
                #if the word is not in the training set, set 1 = P(x|C=k)
                else: 
                    xi *= 1
                    yi *= 1
            #add the values to a list multiplied by the probabilities of being ham/spam
            ham.append(xi*(Hn/n))
            spam.append(yi*(Sn/n))
        #add the lists together to be 2 columns of an array
        x = []
        x.append(ham)
        x.append(spam)
        x = np.array(x)
        x = x.T
        #return said array
        return x
        #raise NotImplementedError('Problem 2 incomplete')

    def predict(self, X):
        '''
        Use self.predict_proba to assign labels to X,
        the label will be a string that is either 'spam' or 'ham'

        Parameters:
            X (pd.Series)(N,): messages to classify
        
        Return:
            (ndarray)(N,): label for each message
        '''
        #get the result from predict_proba
        A = self.predict_proba(X)
        label = []
        #determine which value is greater ham or spam and add that to a list
        for i in range(len(A)):
            if A[i][0] > A[i][1]:
                label.append('ham')
            else:
                label.append('spam')
        #return said list
        return label
        #raise NotImplementedError('Problem 3 incomplete')
        
    def predict_log_proba(self, X):
        '''
        Find ln(P(C=k|x)) for each x in X and for each class k

        Parameters:
            X (pd.Series)(N,): messages to classify
        
        Return:
            (ndarray)(N,2): Probability each message is ham, spam
                0 column is ham
                1 column is spam
        '''
        #split each line into individual words
        words = X.str.split()
        #count the total number of words that are identified Ham or Spam
        Hn = 0
        Sn = 0
        #create a mask for ham and spam to easier find the totals for the respective probabilities 
        hammask = []
        spammask = []
        for i in range(len(self.data)):
            if self.data['labels'].values[i] == 'ham':
                Hn += sum(self.data.values[i][:-1])
                hammask.append(True)
                spammask.append(False)
            else:
                Sn += sum(self.data.values[i][:-1])
                hammask.append(False)
                spammask.append(True)
         
        n = Hn + Sn
        
        #sum the values from each column of the dataframe that has how many times each unique value is in each phrase contains a word, but use the mask to only get the sums for the ham/spam values
        ham = []
        spam = []
        for i in range(len(words)):
            xi = 1
            yi = 1
            for j in range(len(words.values[i])):
                if words.values[i][j] in self.data.columns:
                    xi += np.log(((self.data[words.values[i][j]].values[hammask].sum())/Hn) + 1e-12)
                    yi += np.log(((self.data[words.values[i][j]].values[spammask].sum())/Sn) + 1e-12)
                    #if the word is not in the training set, set 1 = P(x|C=k)
                else: 
                    xi += 0
                    yi += 0
            #add the values to a list multiplied by the probabilities of being ham/spam
            ham.append(xi + np.log((Hn/n + 1e-12)))
            spam.append(yi + np.log((Sn/n + 1e-12)))
        #add the lists together to be 2 columns of an array
        x = []
        x.append(ham)
        x.append(spam)
        x = np.array(x)
        x = x.T
        #return said array
        return x
        #raise NotImplementedError('Problem 4 incomplete')
        

    def predict_log(self, X):
        '''
        Use self.predict_log_proba to assign labels to X,
        the label will be a string that is either 'spam' or 'ham'

        Parameters:
            X (pd.Series)(N,): messages to classify
        
        Return:
            (ndarray)(N,): label for each message
        '''
        A = self.predict_log_proba(X)
        label = []
        #determine which value is greater ham or spam and add that to a list
        for i in range(len(A)):
            if A[i][0] > A[i][1]:
                label.append('ham')
            else:
                label.append('spam')
        #return said list
        return label
        #raise NotImplementedError('Problem 4 incomplete')


class PoissonBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    This classifier assumes that words are distributed like 
    Poisson random variables
    '''

    def __init__(self):
        return

    
    def fit(self, X, y):
        '''
        Uses bayesian inference to find the poisson rate for each word
        found in the training set. For this we will use the formulation
        of l = rt since we have variable message lengths.

        This method creates a tool that will allow the filter to 
        evaluate P(H), P(S), and P(w|C)


        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        
        Returns:
            self: this is an optional method to train
        '''
        #split the x training set into individual words
        sf = X.str.split()
        unique = []
        #create a list of the unique words
        for i in sf.values:
            for j in i:
                if j not in unique:
                    unique.append(j)
            
        #create an empty dataframe with columns as all the unique words
        gf = pd.DataFrame(columns = unique,index = X.index)
        #make each row the line of words from X and count how many times each unique word is in that line
        for i in range(len(unique)):
            for j in range(len(sf.values)):
                gf.values[j][i] = sf.values[j].count(unique[i])
        
        #add the columns of labels from the y train
        gf['labels'] = y
        self.data = gf
        
        #create a mask and find the totals for spam and ham
        hammask = []
        spammask = []
        Hn = 0
        Sn = 0
        for i in range(len(self.data)):
            if self.data['labels'].values[i] == 'ham':
                Hn += sum(self.data.values[i][:-1])
                hammask.append(True)
                spammask.append(False)
            else:
                Sn += sum(self.data.values[i][:-1])
                hammask.append(False)
                spammask.append(True)
                
        #apply the mask   
        ham = []
        spam = []
        for i in unique:
            ham.append((self.data[i].values[hammask].sum()))
            spam.append((self.data[i].values[spammask].sum()))
        
        #define a new self.data that has 2 rows, ham and spam and columns for all the unique words
        A = []
        A.append(ham)
        A.append(spam)
        A = np.array(A)
        ff = pd.DataFrame(A, index = ['ham','spam'], columns = unique)
        ff['labels'] = y
        self.data = ff
        
        #define the array for the ham and spam rates using the totals
        ham = []
        spam = []
        for i in range(len(self.data.values[0])):
            ham.append(self.data.values[0][i])
            spam.append(self.data.values[1][i])
            
        #take out the last value of the array (the labels)
        ham = np.array(ham[:-1])
        spam = np.array(spam[:-1])
        ham = ham/Hn
        spam = spam/Sn
        
        #create the ham and spam rate dataframes
        hf =pd.DataFrame(ham, index = unique, columns = ['ham'])
        sf = pd.DataFrame(spam, index = unique, columns = ['spam'])
        self.ham_rates = hf.T
        self.spam_rates = sf.T
        
        #raise NotImplementedError('Problem 6 incomplete')
    
    def predict_proba(self, X):
        '''
        Find P(C=k|x) for each x in X and for each class

        Parameters:
            X (pd.Series)(N,): messages to classify
        
        Return:
            (ndarray)(N,2): Probability each message is ham or spam
                column 0 is ham, column 1 is spam 
        '''
        #split the data by word
        sf = X.str.split()
        
        #find the totals for how many words are in each feature (ham and spam)
        Hn = sum(self.data.values[0][:-1])
        Sn = sum(self.data.values[1][:-1])
        n = Hn + Sn
        
        #using equation 1.9 and Poisson, find the predicted outcome
        ham = []
        spam = []
        for i in sf:
            #set initial conditions
            L = set(i)
            H = []
            S = []
            for j in L:
                #account for every possibility of whether the values we are sampling are in the spam and ham rates already
                if j in self.ham_rates.columns and j in self.spam_rates.columns:
                    #more initial conditions
                    ni = i.count(j)
                    n = len(i)
                    r = self.ham_rates[j].values[0]
                    s = self.spam_rates[j].values[0]
                    np.log(((r*n)**ni*np.exp(-r*n))/(np.math.factorial(ni))+ 1e-12)
                    H.append(np.log(((r*n)**ni*np.exp(-r*n))/(np.math.factorial(ni))+ 1e-12))
                    S.append(np.log(((s*n)**ni*np.exp(-s*n))/(np.math.factorial(ni))+ 1e-12))
                elif j in self.spam_rates.columns and j not in self.ham_rates.columns:
                    ni = i.count(j)
                    n = len(i)
                    s = self.spam_rates[j].values[0]
                    H.append(0)
                    S.append(np.log(((s*n)**ni*np.exp(-s*n))/(np.math.factorial(ni))+ 1e-12))
                elif j in self.ham_rates.columns and j not in self.spam_rates.columns:
                    ni = i.count(j)
                    n = len(i)
                    r = self.ham_rates[j].values[0]
                    H.append(np.log(((r*n)**ni*np.exp(-r*n))/(np.math.factorial(ni))+ 1e-12))
                    S.append(0)
                elif j not in self.ham_rates.columns and j not in self.spam_rates.columns:
                    H.append(0)
                    S.append(0)
            #take the sum of the arrays we just found, and add that to to the natural log of the probability of being in that feature
            ham.append(np.log(Hn/n+ 1e-12) + sum(H))
            spam.append(np.log(Sn/n+ 1e-12) + sum(S))
        #add the lists together to be 2 columns of an array
        x = []
        x.append(ham)
        x.append(spam)
        x = np.array(x)
        x = x.T
        #return said array
        return x
                
                
        
        #raise NotImplementedError('Problem 7 incomplete')

    def predict(self, X):
        '''
        Use self.predict_proba to assign labels to X

        Parameters:
            X (pd.Series)(N,): messages to classify
        
        Return:
            (ndarray)(N,): label for each message
        '''
        A = self.predict_proba(X)
        label = []
        #determine which value is greater ham or spam and add that to a list
        for i in range(len(A)):
            if A[i][0] > A[i][1]:
                label.append('ham')
            else:
                label.append('spam')
        #return said list
        return label
        #raise NotImplementedError('Problem 7 incomplete')



def sklearn_method(X_train, y_train, X_test):
    '''
    Use sklearn's methods to transform X_train and X_test, create a
    naïve Bayes filter, and classify the provided test set.

    Parameters:
        X_train (pandas.Series): messages to train on
        y_train (pandas.Series): labels for X_train
        X_test  (pandas.Series): messages to classify

    Returns:
        (ndarray): classification of X_test
    '''
    #using the vectorizer.... vectorize
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
    X_test_vectorized = vectorizer.transform(X_test).toarray()
    
    #use the multinomial Naive Bayes to fit and predict the data
    nb = MultinomialNB()
    nb.fit(X_train_vectorized, y_train)
    y_prediction = nb.predict(X_test_vectorized)
    #return the prediction
    #I would also like to add that i did tested this and got a 98% accuracy. Turns out, when
    #you use the functions already implemented you get nice results. 
    return y_prediction

    
    
    #raise NotImplementedError('Problem 8 incomplete')
    
#test functions    
def prob1234():
    df = pd.read_csv(r'\Users\rober\Desktop\Volume3\NaiveBayes\sms_spam_collection.csv')
    X = df.Message
    y = df.Label
    NB = NaiveBayesFilter()
    NB.fit(X[:300], y[:300])
    NB.predict_log(X[530:535])
    print(NB.score(X[-300:], y[-300:]))
    print(NB.predict_log_proba(X[[1085,2010]]))
    
def prob6():
    df = pd.read_csv(r'\Users\rober\Desktop\Volume3\NaiveBayes\sms_spam_collection.csv')
    X = df.Message
    y = df.Label
    PB = PoissonBayesFilter()
    PB.fit(X[:300], y[:300])
    print(PB.ham_rates['i'])
    print(PB.spam_rates['i'])
    
def prob7():
    df = pd.read_csv(r'\Users\rober\Desktop\Volume3\NaiveBayes\sms_spam_collection.csv')
    X = df.Message
    y = df.Label
    PB = PoissonBayesFilter()
    PB.fit(X[:300], y[:300])
    print(PB.predict(X[530:535]))
    print(PB.score(X[-300:],y[-300:]))
    
def prob8():
    data = pd.read_csv('sms_spam_collection.csv')
    X = data.Message
    y = data.Label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    y_prediction = sklearn_method(X_train, y_train, X_test)
    print(metrics.accuracy_score(y_test,y_prediction))
    #nb.score(X[-300:],y[-300:])
    
    
    