# Transfer-learning-approach-in-NLP
Here in this repository we will classify the  News data using UNIVERSAL SENTENCE embedings approach
Question classification task and data preprocessing
To respond correctly to a question given a large collection of texts, classifying questions into fine-grained classes is crucial in question answering as a retrieval task.  Our goal is to categorize questions into different semantic classes that impose constraints on potential answers so that they can be utilized in later stages of the question answering process. For example, when considering the question Q: What Canadian city has the largest population? The hope is to classify this question as having answer type location, implying that only candidate answers that are locations need consideration.

The dataset we use is the TREC Question Classification dataset, There are entirely 5452 training and 500 test samples, that is 5452 + 500 questions each categorized into one of the six labels.

ABBR - 'abbreviation': expression abbreviated, etc.
DESC - 'description and abstract concepts': manner of an action, description of sth. etc.
ENTY - 'entities': animals, colors, events, food, etc.
HUM - 'human beings': a group or organization of persons, an individual, etc.
LOC - 'locations': cities, countries, etc.
NUM - 'numeric values': postcodes, dates, speed,temperature, etc
We want our model to be a multiclass classification model that takes strings as input and output probability for each of the 6 class labels. With this in mind, you know how to prepare the training and testing data for it.
