# Clickbait Classification of YouTube Video Titles

## Objective

<p align="justify">
First we are going to discuss the related work in this field to realize what are the past works done and what are the different methodologies used there. This will help us in identifying different ways of transforming the data so that it could be used properly by the machine learning algorithms. After that we will do the Exploratory Data Analysis which will help us understand the dataset. It is followed by the data pre-processing which is an important step as it helps in converting the string data to values understandable by the algorithms. After that the last step is training the ML model and comparing the accuracy and other different results.
</p>

<hr>

## Description

<p align="justify">
In this project, we are getting the data from kaggle which contains many types of different datasets. The dataset that we have found will serve its purpose sufficiently in this project. Natural Language Processing along with Machine Learning has been used for training and testing purposes. The various algorithms will be briefly discussed in the below sections. Machine learning algorithms used in this project include Multinomial Naive Bayes, Support Vector Machine (SVM), Random Forest and Bidirectional Encoder Representations from Transformers (BERT).
</p>

<hr>

## Technical Libraries Needed

Google collab with inbuilt GPU.

Libraries of python:

- Scikit Learn
- Numpy
- Pandas
- Seaborn,
- Matplotlib
- Tensorflow

<p align="justify">
The innovation in our approach is the use of a multilingual dataset of video titles. Using dataset two that has headlines in Indian regional languages like Hindi, Malayalam, Tamil, Telugu and Kannada, we plan to identify and analyze the syntactic features of clickbait video titles. The use of transfer learning with BERT to perform classification is also unique. BERT being pre-trained specifically for NLP tasks can be used to solve this problem statement as the input consists of textual data in the form of video titles that need contextual understanding. Thus we expect BERT to perform better than deep learning models like CNN.
</p>

<hr>

## Design Approach

The following steps were performed:

- <p align="justify"> To form the complete dataset consisting of both clickbait and non-clickbait titles, first a column titled isClickbait to both the clickbait and non clickbait datasets. The isClickbait value was filled with 1 for the clickbait dataset and 0 for non-clickbait dataset. The two csv files were then merged and the rows randomized. </p>
- Excess columns Video_ID and Favorites are dropped
- <p align="justify"> The text in the video title column was lowercase, extra spaces and stopwords removed. Punctuations are retained as they are important features in clickbait classification as seen in the literature survey. </p>
- The video titles are lemmatized using the WordNetLemmatizer from NLTK library.
- Tf-Idf vectorization is used to convert the title texts into a matrix of numeric values.
- <p align="justify"> Features of the video title such as no punctuations, mean length of title, views to likes ratio, views to dislikes ratio and sentiment score are extracted for each title and added to the vector as a new feature. </p>
- The dataset is split into train and test sets in a 8:2 ratio.
- <p align="justify"> The dataset is fit into the ML models one by one and predictions made on the test set. The accuracy scores are noted. </p>

<hr>

#### Flowchart

<p align="center">
    <img src="./images/image.png">
</p>

<p align="center">
    Flow Chart of the Approach
</p>

In the case of BERT:

- <p align="justify"> Each lemmatized video title is tokenized and converted to a vector representation using the pretrained tokenizers of BERT. </p>
- <p align="justify"> A base pretrained model for BERT is imported as a local variable and additional layers according to our problem statement are being added and compiled(transfer learning). </p>
- The vectors are then sent as input to the BERT model.
- The output of the BERT model is trained further using the tensorflow Dense layers.
- <p align="justify"> The models are trained based on this data and used to predict the final classification for the isClickbait column. </p>

<p align="center">
    <img src="./images/image2.png">
</p>

<p align="center">
    BERT Text Classifier
</p>

<p align="justify">
The dataset has been taken from a well known online community for data scientists known as kaggle. There are two different types of datasets that we have used. </p>

<hr>

#### Datasets

<p align="justify">
The <b> first dataset </b> had the attributes ID, Video Title, Views, Likes, Dislikes and Favorites. There were two csv files for the first dataset. The first csv file consisted of the data which were clickbaits and the second csv file was for the data which were non clickbaits. As these were two different files so we had to import them and then merge the data from both the files. After merging the datas from both the files we randomized the rows so that the clickbait and non clickbait data get mixed properly. This dataset contains more than 32,000 rows of data. This dataset contains youtube titles from various channels available on youtube. In this dataset the features such as ID and Favorites are of no use in training the Machine Learning and Deep Learning models because they are not affecting the result if the video is clickbait or not. So we are going to use the rest of the features such as Video Title, Views, Likes and Dislikes to train the models and then use it to classify other Videos. We added another column named “isClickbait”. The clickbait data were put as 1 and non clickbait data were put as 0 in this column and then jumbled. </p>

<b> Columns: </b> ID, Video Title, Views, Likes, Dislikes, Favorites, isClickbait

<p align="justify"> The second dataset that we are going to use is updated on a daily basis. It has many more features compared to the first dataset. It has features such as Video ID, Video Title, Published At, Channel ID, Channel Title, Category ID, Trending Date, View Count, Likes, Dislikes, Comment Count and Description. The unique proposition of this dataset is that it has headlines in Hindi, Malayalam, Tamil, Telugu and Kannada, so with this dataset we can even classify titles with multilingual headlines. The features such as Video ID, Published At, Channel ID, Category ID, Trending Date are not useful for training because they are just used as identifiers. The main problem with this dataset is that we have to manually read the rows and classify it as clickbait or non-clickbait. So we have to add another column called “isClickbait” and put all the values there after the manual classification. The useful features that we are going to use are Video Title, Channel Title, View Count, Likes, Dislikes, Comment Count and Description, these will be used to train the different models. </p>

<p align="justify"> <b> Columns: </b> Video_ID, Video_Title, Published_At, Channel_ID, Channel_Title, Category_ID, Trending_Date, View_Count, Likes, Dislikes,Comment_Count, Description, isClickbait </p>

#### Limitation

<p align="justify">
One of the major limitations of the BERT text classification algorithm is lack of ability to handle long text sequences. By default, BERT supports up to 512 tokens. There are multiple ways to overcome it: Ignore text after 512 tokens. </p>

<hr>

## Results and Discussions

### Data Analysis

#### Preprocessing Results: Lower Casing, Stop Word Removal and Lemmatization

<p align="center">
    <img src="./images/image-9.png">
</p>

<hr>

#### Missing Numbers in Dataset

<p align="center">
    <img src="./images/image-1.png">
</p>

<hr>

#### Word Cloud of Video Titles

<p align="center">
    <img src="./images/image-2.png">
</p>

<hr>

#### Top 20 Unigrams

<p align="center">
    <img src="./images/image-3.png">
</p>

<hr>

#### Top 20 Bigrams

<p align="center">
    <img src="./images/image-4.png">
</p>

<hr>

#### Top 20 Trigrams

<p align="center">
    <img src="./images/image-5.png">
</p>

<hr>

#### Target Distribution

<p align="center">
    <img src="./images/image-6.png">
</p>

<hr>

#### Top 20 Clickbait Headline Words

<p align="center">
    <img src="./images/image-7.png">
</p>

<hr>

#### Top 20 Non - Clickbait Headline Words

<p align="center">
    <img src="./images/image-8.png">
</p>

<hr>

<p align="justify">
The final output for each row of data is a binary label of 0 referring to the title being non-clickbait and 1 referring to the title being clickbait. We expect an accuracy above 95% for the english video titles dataset and above 90% for the multilingual video titles dataset.
</p>

<hr>

### Model Accuracy

#### Naive Bayes

<p align="center">
    <img src="./images/image-10.png">
</p>

<p align="center">
    Naive Bayes Classifier Results
</p>

<hr>

#### Random Forest

<p align="center">
    <img src="./images/image-11.png">
</p>

<p align="center">
    Random Forest Classifier Results
</p>

<hr>

#### SVM

<p align="center">
    <img src="./images/image-12.png">
</p>

<p align="center">
    SVM Classifier Results
</p>

<hr>

#### XG Boost

<p align="center">
    <img src="./images/image-13.png">
</p>

<p align="center">
    Feature Extraction using XG Boost
</p>

<hr>

#### BERT

<p align="center">
    <img src="./images/image-14.png">
</p>

<p align="center">
    Training the BERT Model
</p>

<hr>

### Comparison

<p align="center">
    <img src="./images/image-15.png">
</p>

<p align="center">
    Result Comparison
</p>

<hr>

### Indian Language Dataset

<p align="center">
    <img src="./images/image-16.png">
</p>

<p align="center">
    Result from Indian Language Dataset
</p>

<br>
<p align="justify">
As seen above, the accuracy of prediction is 70% on the Indian languages youtube dataset. This is almost at par with the state of the art results obtained by training mBERT.
</p>
<br>

<p align="center">
    <img src="./images/image-17.png">
</p>

<p align="center">
    Manual Classification Results on Indian Language Trained Model
</p>

<br>
<p align="justify">
As shown above we have correctly classified three clickbait titles in English, Hindi and Tamil.
</p>

<hr>

## Summary

<p align="justify">
The results that we have got from the BERT text classification algorithm is better than other Machine Learning algorithms. The testing accuracy for BERT was 97% in the case of the English Language dataset and 70% in the case of the Indian Language dataset. For the Machine Learning algorithms such as Naive Bayes, Random Forest and SVM, the testing accuracy was 92%, 93% and 96% respectively.
</p>
