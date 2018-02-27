from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from nltk import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn import metrics

import pprint as pp



###############################################################################
stemmer = EnglishStemmer()

def stemming_tokenizer(text):
    stemmed_text = []
    for word in word_tokenize(text, language='english'):
        if word not in stopwords.words('english'):
            stemmed_text.append(stemmer.stem(word))
    return stemmed_text
###############################################################################

def trainClassifier(pipeline, parameters):
    # Find the best combination of parameters.
    # We use n_jobs set to -1 so GridSearchCV will automatically run using 
    # all of the CPU cores.
    grid_search = GridSearchCV(pipeline,
                               parameters,
                               scoring=metrics.make_scorer(metrics.matthews_corrcoef),
                               cv=10,
                               n_jobs=-1,
                               verbose=10)
    
    # Start an exhaustive search to find the best combination of parameters
    # according to the selected scoring-function.
    print
    grid_search.fit(X_train, Y_train)
    print
    
    # Print results for each combination of parameters.
    number_of_candidates = len(grid_search.cv_results_['params'])
    print("Results:")
    for i in range(number_of_candidates):
    	print(i, 'params - %s; mean - %0.3f; std - %0.3f' %
    			(grid_search.cv_results_['params'][i],
    			grid_search.cv_results_['mean_test_score'][i],
    			grid_search.cv_results_['std_test_score'][i]))
    
    print
    print("Best Estimator:")
    pp.pprint(grid_search.best_estimator_)
    print
    print("Best Parameters:")
    pp.pprint(grid_search.best_params_)
    print
    print("Used Scorer Function:")
    pp.pprint(grid_search.scorer_)
    print
    print("Number of Folds:")
    pp.pprint(grid_search.n_splits_)
    print
    
    
    # Let's train the classifier that achieved the best performance, considering
    # the selected scoring function, on the entire original training set.
    Y_predicted = grid_search.predict(X_test)

    # Evaluate the performance of the classifier on the original test set.
    output_classification_report = metrics.classification_report(Y_test,
                                                                 Y_predicted,
                                                                 target_names=target_names)
    print
    print("----------------------------------------------------")
    print(output_classification_report)
    print("----------------------------------------------------")
    print
    
    # Compute the confusion matrix.
    confusion_matrix = metrics.confusion_matrix(Y_test, Y_predicted)
    print
    print("Confusion Matrix: True-Classes X Predicted-Classes")
    print(confusion_matrix)
    print
    
    # Compute the Accuracy Score.
    accuracy_score = metrics.accuracy_score(Y_test, Y_predicted)
    print
    print("Accuracy Score: " + str(accuracy_score))
    print
    
    # Compute Matthews CorrCoef.
    matthews_corrcoef = metrics.matthews_corrcoef(Y_test, Y_predicted)
    print
    print("Matthews CorrCoef: " + str(matthews_corrcoef))
    print
    
###############################################################################



# Dataset containing Positive and Negative sentences.
data_folder_training_set = "./Positve_negative_sentences/Training"
data_folder_test_set     = "./Positve_negative_sentences/Test"

training_dataset = load_files(data_folder_training_set)
test_dataset = load_files(data_folder_test_set)
print
print("----------------------")
print(training_dataset.target_names)
print("----------------------")
print


# Load training set.
X_train, X_test_DUMMY_to_ignore, Y_train, Y_test_DUMMY_to_ignore = train_test_split(training_dataset.data,
													training_dataset.target,
													test_size=0.0)
target_names = training_dataset.target_names

# Load test set.
X_train_DUMMY_to_ignore, X_test, Y_train_DUMMY_to_ignore, Y_test = train_test_split(test_dataset.data,
													test_dataset.target,
													train_size=0.0)

target_names = training_dataset.target_names
print
print("----------------------")
print("Creating Training Set and Test Set")
print
print("Training Set Size")
print(Y_train.shape)
print
print("Test Set Size")
print(Y_test.shape)
print
print("Classes:")
print(target_names)
print("----------------------")


# Vectorization object.
vectorizer = TfidfVectorizer(strip_accents= None, preprocessor = None,)
    

###############################################################################
# Classifier 1: KNeighbors.
###############################################################################
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier()


# Create pipeline to assemble steps.
pipelineKnc = Pipeline([('vect', vectorizer),('knc', knc),])


# Setting parameters.
# We try different values for n-grams and the number of neighbors.
parametersKnc = {
	'vect__tokenizer': [None, stemming_tokenizer],
	'vect__ngram_range': [(1, 1), (1, 2)],
	'knc__n_neighbors': [5,10,15,20],
	}
 
 
trainClassifier(pipelineKnc, parametersKnc)


###############################################################################
# Classifier 2: SGDClassifier - LASSO.
###############################################################################
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()


# Create pipeline to assemble steps.
pipelineClf = Pipeline([('vect', vectorizer), ('clf', clf)])


# Setting parameters.
# We try different values for n-grams, default alpha (0.00001), type of penalty
# set to l2 (LASSO) and different number of iterations.
parametersClf = {
	'vect__tokenizer': [None, stemming_tokenizer],
	'vect__ngram_range': [(1, 1), (1, 2)],
      'clf__penalty' : ['l2'],
      'clf__n_iter' : (10, 50, 80)
	}
 
 
trainClassifier(pipelineClf, parametersClf)


###############################################################################
# Classifier 3: Multinomial Naive Bayes.
###############################################################################
from sklearn.naive_bayes import MultinomialNB
nbc = MultinomialNB()


# Create pipeline to assemble steps.
pipelineNbc = Pipeline([('vect', vectorizer), ('nbc', nbc)])


# Setting parameters.
# We try different values for n-grams and different values for alpha.
parametersNbc = {
	'vect__tokenizer': [None, stemming_tokenizer],
	'vect__ngram_range': [(1, 1), (1, 2)],
      'nbc__alpha': [.001, .01, 1.0, 10.],
	}
 
 
trainClassifier(pipelineNbc, parametersNbc)

###############################################################################
