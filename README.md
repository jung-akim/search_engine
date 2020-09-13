## Review-based Search Engine
### Background
Since COVID-19, e-commerce market value has risen by dramatically and the companies are more competitive in satisfying customer's needs.
One of the keys in determining the purchase is the reviews. According to BrightLocal's Local Consumer Review Survey 2014, 88% of consumers say they trust online reviews as much as personal recommendations and the average consumer reads 10 reviews before feeling able to trust a business<br>
[link1](https://searchengineland.com/88-consumers-trust-online-reviews-much-personal-recommendations-195803), [link2](https://www.brightlocal.com/research/local-consumer-review-survey/)

### The goal of this project
For both consumers and the company, the goal is to retrieve the optimal product with most relevant reviews about pros and cons that consumers would want to check before purchasing.

### What is the plan for this project?
The input would be a few sentences with keywords that the consumer is looking for in a product. The first output is the product that matches the key qualities with positive sentiment based on the reviews. After the optimal product is chosen, the reviews will be divided into pros and cons and sorted in terms of relevancy to the query. Each review can be summarized into few sentences, if too long.

### Dataset
The dataset is from "Amazon Customer Reviews Dataset" that are publicly available in S3 bucket in AWS US East Region. The dataset used for this project is the subset of "amazon_reviews_us_Electronics_v1_00.tsv" file which contains information of each review on different electronics. It has 3,093,869 reviews. There may be other datasets for training a model such as sentiment or summary.
Dataset download instruction and description is in this [link](https://s3.amazonaws.com/amazon-reviews-pds/readme.html)

### Tools
tensorflow_hub has Universal Sentence Encoding which is a context-based sentence embedding that uses transformers. USE is developed by google in 2018 which can be used for text similarity, sentiment analysis, etc. For summarization, PEGASUS - Reddit_TIFU was used.

### Notebooks
1. [train_sentiment.ipynb](train_sentiment.ipynb)<br>
Trains a neural network model for sentiment analysis using Universal Sentence Encoder embeddings. Each embedding is a 512-dimensional vector representation of a product's review. The product reviews used for this training contains fewer than 10 reviews. The products with 10 or more reviews are used for the search engine.
2. [train_summarizer.ipynb](train_summarizer.ipynb)<br>
Compares candidate models for summarization of reviews. There are 5 candidate models: BART trained with CNN dataset, DISTILBART - CNN, BART with XSUM(BBC) articles, DISTILBART - XSUM, and PEGASUS trained with Reddit_TIFU. These models are capable of abstractive summarization.
3. [USE_PEGASUS.ipynb](USE_PEGASUS.ipynb)<br>
Using the sentiment analysis model and custom-made (weighted) similarity matrix, "Positive Similarity score" is computed for each product, and the products with the highest scores are selected to be shown to the user. Positive Similarity score is a measurement of how strongly reviews of a product contain positive sentiment and relevancy to the user's query. The output products are shown with the reviews that are similar to the user's query. Some long reviews are summarized into fewer sentences using the summarizer model.

### Directories
1. Datasets<br>
[data](data) contains a raw tsv dataset, pickled datasets(products with 10 or more reviews and products with fewer than 10 reviews), and parquet file written by pyspark.
(The data folder is empty in this repository due to its size. If you want to run this repository, please contact me.)
2. Model<br>
[model](model) contains the keras model trained for sentiment analysis.
(The model folder is empty in this repository due to its size. If you want to run this repository, please contact me.)
3. Search Engine App<br>
[search_engine_app](search_engine_app) contains the HTML, CSS, javascript file for implementing Flask app.
4. Models<br>
[train_sentiment](train_sentiment) contains the train-test split datasets.
(The train_sentiment folder is empty in this repository due to its size. If you want to run this repository, please contact me.)
5. Universal Sentence Encoder<br>
[use](use) contains the downloaded USE embedder. It can be downloaded directly from tensorflow_hub link without having to download it.
(The use folder is empty in this repository due to its size. If you want to run this repository, please contact me.)

### Python scripts
[utils.py](utils.py)<br>
Inlcudes helper functions to compute the custom-made positive similarity scores.

### References
- Universal Sentence Encoder, Daniel Cer and Yinfei Yang and Sheng-yi Kong and Nan Hua and Nicole Limtiaco and Rhomni St. John and Noah Constant and Mario Guajardo-Cespedes and Steve Yuan and Chris Tar and Yun-Hsuan Sung and Brian Strope and Ray Kurzweil, 2018, arXiv:1803.11175
- PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization, Jingqing Zhang and Yao Zhao and Mohammad Saleh and Peter J. Liu, 2019, arXiv:1912.08777
- Sentiment Analysis with Tensorflow 2 and Keras using Python, Curiously, 2019, https://www.curiousily.com/posts/sentiment-analysis-with-tensorflow-2-and-keras-using-python/
- Hugging Face datasets, https://huggingface.co/models?filter=summarization
- TransformerSum documentation, https://transformersum.readthedocs.io/en/latest/general/about.html#extractive-vs-abstractive-summarization
- Comparison of different Word Embeddings on Text Similarity - A use case in NLP, Intellica.AI, 2019, https://medium.com/@Intellica.AI/comparison-of-different-word-embeddings-on-text-similarity-a-use-case-in-nlp-e83e08469c1c

### Python Packages
##### tensorflow
Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo,
Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis,
Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow,
Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia,
Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster,
Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens,
Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker,
Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas,
Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke,
Yuan Yu, and Xiaoqiang Zheng.
TensorFlow: Large-scale machine learning on heterogeneous systems,
2015. Software available from tensorflow.org.

##### spacy
Honnibal, M., & Montani, I. (2017). spaCy 2: *Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing.*

##### nltk
Bird, Steven, Edward Loper and Ewan Klein (2009).
Natural Language Processing with Python.  O'Reilly Media Inc.

##### pySpark
https://spark.apache.org/docs/latest/api/python/index.html

##### pyarrow
https://readthedocs.org/projects/pyarrow/downloads/pdf/latest/

##### keras
Keras, Chollet, Francois and others, 2015, url[https://keras.io]

##### transformers
HuggingFace's Transformers: State-of-the-art Natural Language Processing, Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush, ArXiv, 2019, abs/1910.03771

##### pytorch
Automatic differentiation in PyTorch, Paszke, Adam and Gross, Sam and Chintala, Soumith and Chanan, Gregory and Yang, Edward and DeVito, Zachary and Lin, Zeming and Desmaison, Alban and Antiga, Luca and Lerer, Adam, 2017

##### pandas
Wes McKinney. Data Structures for Statistical Computing in Python, Proceedings of the 9th Python in Science Conference, 51-56 (2010)

##### numpy
* Travis E. Oliphant. A guide to NumPy, USA: Trelgol Publishing, (2006).
* Stéfan van der Walt, S. Chris Colbert and Gaël Varoquaux. The NumPy Array: A Structure for Efficient Numerical Computation, Computing in Science & Engineering, 13, 22-30 (2011), DOI:10.1109/MCSE.2011.37

##### sklearn
Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, Édouard Duchesnay. Scikit-learn: Machine Learning in Python, Journal of Machine Learning Research, 12, 2825-2830 (2011)

##### matplotlib
John D. Hunter. Matplotlib: A 2D Graphics Environment, Computing in Science & Engineering, 9, 90-95 (2007), DOI:10.1109/MCSE.2007.55

##### gensim
Radim Rehurek and Petr Sojka, Software Framework for Topic Modelling with Large Corpora, *Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks*, 45--50, 2010, May 22, ELRA, Valletta, Malta, URL: http://is.muni.cz/publication/884893/en

##### ipython
Fernando Pérez and Brian E. Granger. IPython: A System for Interactive Scientific Computing, Computing in Science & Engineering, 9, 21-29 (2007), DOI:10.1109/MCSE.2007.53

##### Pycharm
JetBrains, 2017. Pycharm. [online] JetBrains. Available at: <https://www.jetbrains.com/pycharm/> [Accessed 11 April 2017].