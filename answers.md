# ChatbotWiz 


## Code

Code is availalbe on GitHub at: [https://github.com/Vachonni/ChatbotWiz](https://github.com/Vachonni/ChatbotWiz)

Please, take the time to read the `README.md` file prior to this file.


## The task

From chatbot conversations to topic classification. 

Include the steps:

* Parsing 
* Topic modelling
* Topic classificaton traing and evaluation
* (search)

### 1. Parse

Code is in `src/data/parse.py`.

Steps: 

1. Unquote 
2. Split by conversation
3. Split by event
4. Get the first message sent by customer. 

Answer is save in a dataframe (instead of a list, as asked). If you want it as a list, do: `df['first_msg_user'].to_list()`

### 2. Cluster

Code is in `src/modelling/topic_modelling.py`.

Used BERTtopic package. Reference here: [https://pypi.org/project/bertopic/](https://pypi.org/project/bertopic/). There are a lot of visualization tools to play with ;) 

Could have been done with LDA.

The 3 topics are formed with these 3 sets of words. They correspond to the most frequent words in the cluster that are less in the other clusters.

1. code password log new
2. printer print scan attached 
3. ticket follow
 
NOTE: Topic #3 has only 2 words since the others were repeting these same words in a uni-gram/bi-gram or are in different order in bi-gram.

### A. Predict

I have trained 2 models and evaluated 3 models, using a pretrained one.

All evaluation are performed in `src/modelling/evaluate.py` and results are in the **Results** section bellow.

Prior to the training and evaluation, I did a quick EDA and I split the data in a stratified manner, even if the class are quite balanced.   

![Distribution of topics](topic_hist.jpg)


#### Basic classifier

Code is in `src/modelling/clf_base.py`.

I have implemented a linear SVM with SGD training from [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html).

I used a Sklearn pipeline to include the text embedding steps (bag-of-word vectorization and tf-idf)

#### Classifier with hyper parameter tuning

Code is in `src/modelling/clf_HP_Search.py`.

Using the same model (and pipeline) as in the basic classifier case, I performed and hyper-parameter search. 

#### Transformer model with zero-shot-classification model

Code is in `src/modelling/evaluate.py`, as no training was necessary,

I used the topic names exactly as descibed in **2.Cluster**. 

I is possible to choose your embedding model, even one you finu-tune on your data. Here, I just used the default one.

#### Other

Obviouly, with a lot of ressources and data, trining your own transformer model would probably give the best solutions as context might be important when trying to determine a topic.


### B. Search

I haven't completed this task.  

A possible high level solution:

1. Embedding of quesions and messages
2. Ranking the different message's embeddings according to their distance to the question's embedding, the closer the better.

Tools to acheive this solution:

1. If good ressources, got for `sentence_transformers` from HuggingFace [here](https://huggingface.co/sentence-transformers), choosing model according to needs (resources, language,...). If low ressources, got for simple tf-idf as in A. Predict task.
2. Any vector distance evalution can do it. Choose wizely according to your KPIs. A good tool is the one develop by Meta, [Faiss](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) 


## Results

### Scores

| Model              | CV - Acc - Mean | CV - Acc - Std |  Test - Accuracy | Test - f1 weighted |
| :---               |    :----:       | :---:          |:----:            |          :---:     |
| **clf base**       | 0.87		      | 0.16           | 0.86             | 0.86               |
| **clf HP search**  | 0.93            | 0.13           | 0.86             | 0.86               |
| **HGF Zero Shot**  | -               | -              | 1.00             | 1.00               |

Both **clf base** and **clf HP search** provide the same results on the test set. Still, we can see from the Cross Validation (CV) evaluation that the model resulting from the hyperparameter search is more precise (less bias) 
and more stable (less variance).

Best results on the test set are achevied by the zero shot classification model from Huggingface. This might be explain by the very little amount of data available to train the classifiers


### Ressources: Time & compute

When time and computing ressources are important to consider, it might be worth it to go for a less performing model. It's a trade-off.



