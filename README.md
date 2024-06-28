INDEX
Practical No.
Title of the Practical
Date
Page No.
Sign
1
Install, configure and run Hadoop and HDFS and explore HDFS
01/04/24



2
Implement an application that stores big data in Hbase / MongoDB and manipulate it using R / Python.
06/04/24


3
Implement Regression Model to import a data from web storage. Name the dataset and now do Linear Regression to find out relation between variables. Also check the model is fit or not.
15/04/24


4
Apply Multiple Regression on a dataset having a continuous independent variable.
19/04/24


5
Build a Classification Model (Logistic Regression) a. Install relevant package for classification. 
b. Choose classifier for classification problem. 
c. Evaluate the performance of classifier.
22/04/24


6
Build a Clustering Model. (K-Means, Agglomerative) 
a. Select clustering algorithm for unsupervised learning. 
b. Plot the cluster data using R/python visualizations.
26/04/24


7
Implement SVM classification technique
29/04/24


8
Implement Decision tree classification technique
04/05/24


9
Naïve Bayes Implementation
11/05/24



Practical 1
Aim:
Install, Configure and Run Hadoop and HDFS and explore HDFS
Theory:
Pre requisite to install Hadoop is java
Hadoop supports Java version 8
Steps:
▪ Install java
▪ Install Hadoop
▪ Configuration files
Step 1: Download Java from oracle website
Search for java SE development kit 8 download
https://www.oracle.com/java/technologies/downloads/#java8-windows


Oracle registration
Provide your email address and password if already registered, else register.
Jdk is downloaded
Step 2: Install Java


Copy this file

And paste it here
Now Java will be available at C:\ Java


Step 3: Setting environment variables for java
Windows->settings->system->environment variables for system->edit the system environment
Variables
Click on environment variables
Set java home and path for java
Set user variables
Set system variables
System variable->path->edit->new


Click Ok-> OK->close
Java successfully installed
Step 4: Java version checking
Go to command prompt
Check Java version


Step 5: Download Hadoop for the local system
https://hadoop.apache.org/releases.html

Hadoop downloaded
Move files to c drive

Step 6: Install Hadoop now to our local system
Unzip Hadoop setup file


Rename Hadoop 3.1.3 as Hadoop

Step 7: Configuration of Hadoop
Hadoop->etc-> Hadoop
Important files:
Core-site.xml
Hdfs-site.xml
Mapred-site.xml
Yarn-site.xml
Hadoop-env windows command prompt file
Open all these files in notepad++
Core-site.xml
<configuration>
<property>
<name>fs.defaultFS</name>
<value>hdfs://localhost:9000</value>
</property
</configuration>

Mapred-site.xml
<configuration>
<property>
<name>mapred.framework.name</name>
<value>yarn</value>
</property>
</configuration>

Yarn-site.xml
<configuration>
<!-- Site specific YARN configuration properties -->
<property>
<name>yarn.nodemanager.aux-services</name>
<value>mapreduce_shuffle</value>
</property>
<property>
<name>yarn.nodemanager.auxservices.mapreduce.shuffle.class</name>
<value>org.apache.hadoop.mapred.ShuffleHandler</value>
</property>
</configuration>

Create data directory and subdirectories in Hadoop

Hdfs-site.xml file
<configuration>
<property>
<name>dfs.replication</name>
<value>1</value>
</property>
<property>
<name>dfs.namenode.name.dir</name>
<value>file:///C:/hadoop/data/namenode</value>
</property>
<property>
<name>dfs.datanode.data.dir</name>
<value>file:///C:/hadoop/data/datanode</value>
</property>
</configuration>
Hadoop-env.xml
set JAVA_HOME=C:\Java\jdk1.8.0_241

Set home and path for Hadoop



Other configuration files
Extract it

Copy bin folder from HadoopConfiguration-Fixbin folder and replace Hadoop\bin folder with this
Step 8: Verification of Hadoop installation
Go to command prompt.
Type:
hdfs namenode -format
set of files pop up on the terminal.
That means successful installation of Hadoop.

Namenode is successfully started.
Open new terminal and start all Hadoop daemons
Go to Hadoop location.
C:\hadoop\sbin
Type the command
start-all.cmd


All the nodes will start successfully.
Also try the following command to start yarn daemons.
start-yarn.cmd


Practical 2
Aim: Implement an application that stores big data in Hbase / MongoDB and manipulate it using R / Python.
Theory:
MongoDB is an open-source and the leading NoSQL database. It is a document-oriented
database that offers high performance, easy scalability, and high availability. It uses documents
and collections to organize data rather than relations. This makes it an ideal database
management system for the storage of unstructured data.
MongoDB uses replica sets to ensure there is a high availability of data. Each replica set is
made up of two or more replicas of data. This gives its users the ability to access their data at
any time. The replica sets also create fault tolerance. MongoDB scales well to accommodate
more data. It uses the sharing technique to scale horizontally and meet the changing storage
needs of its users. MongoDB was developed to help developers unleash the power of data and
software.
MongoDB is an unstructured database. It stores data in the form of documents. MongoDB is
able to handle huge volumes of data very efficiently and is the most widely used NoSQL
database as it offers rich query language and flexible and fast access to data.
The Architecture of a MongoDB Database
The information in MongoDB is stored in documents. Here, a document is analogous
to rows in structured databases.
• Each document is a collection of key-value pairs
• Each key-value pair is called a field
• Every document has an _id field, which uniquely identifies the documents
• A document may also contain nested documents
• Documents may have a varying number of fields (they can be blank as well)
These documents are stored in a collection. A collection is literally a collection of documents in MongoDB. This is analogous to tables in traditional databases.

Unlike traditional databases, the data is generally stored in a single collection in MongoDB, so
there is no concept of joins (except $lookup operator, which performs left-outer-join like
operation). MongoDB has the nested document instead.
PyMongo is a Python library that enables us to connect with MongoDB. It allows us to perform
basic operations on the MongoDB database.
We have chosen Python to interact with MongoDB because it is one of the most commonly
used and considerably powerful languages for data science. PyMongo allows us to retrieve the
data with dictionary-like syntax.
We can also use the dot notation to access MongoDB data. Its easy syntax makes our job a lot
easier. Additionally, PyMongo’s rich documentation is always standing there with a helping
hand. We will use this library for accessing MongoDB.
Steps of the installation:
Step 1: Download MongoDB
Go to official website: MongoDB Community server
https://www.mongodb.com/try/download/community

Click on download
Step 2: Install MongoDB
It will download msi file. Click on it and Start the installation.


Click on Complete

Keep all these settings as it is. Click on next



Step 3: Verify MongoDB Installation
Now go to
C:\Program Files\MongoDB\Server\5.0\bin
Start command prompt from this location

To start mongo db server
Enter mongod command

It says c:\data\db directory not found
So create c:\data\db

Run the above command mongod once again

Mongo daemon is started now
To open the mongo shell
Go to
C:\Program Files\MongoDB\Server\5.0\bin
Start command prompt from this location
Fire the command: mongo

Mongo shell is started
To see all the default databases:
>show dbs

To create new database named my_database
To create collection in the database

And insert json values into it
To see entered collection
To see all the documents in the collection:

To set path of MongoDB server and shell
Copy in clipboard: C:\Program Files\MongoDB\Server\5.0\bin
Go to environment variables
Add mongodb path to system path variable

Run mongod and mongo command from command prompt once again from anywhere, it will
start mongo server and mongo shell

To get the effect of running mongod as service, Restart your windows operating system.
Restarted the machine
To check mongod service is running automatically:
Open command prompt
Give mongo command without running mongod command in another terminal.
It will start mongod server.

Step 4: Install MongoDB Python on Windows
We will be performing a few key basic operations on a MongoDB database in Python using
the PyMongo library.
Install package to use MongoDB
To install this package with conda run:
conda install -c anaconda pymongo


Step 5: Verify MongoDB Python Connection
To retrieve the data from a MongoDB database, we will first connect to it. Write and execute
the below code in your spider anaconda
import pymongo
mongo_uri = "mongodb://localhost:27017/"
client = pymongo.MongoClient(mongo_uri)
Let’s see the available databases:
print(client.list_database_names())
We will use the my_database database for our purpose. Let’s set the cursor to the same
database:
db = client.my_database
connect to analysis database
The list_collection_names command shows the names of all the available collections:
print(db.list_collection_names())
Let’s see the number of books we have. We will connect to the customers collection and then
print the number of documents available in that collection:
table=db.books
print(table.count_documents({}) ) #gives the number of documents in the table
Code:
import pymongo
mongo_uri = "mongodb://localhost:27017/"
client = pymongo.MongoClient(mongo_uri)
print(client.list_database_names())
db = client.my_database
print(db.list_collection_names())
table=db.books
zprint(table.count_documents({}))
Output:

Practical 3
Aim: 
Implement Regression Model to import a data from web storage. Name the dataset and now do Linear Regression to find out relation between variables. Also check the model is fit or not.  
Theory: 
Linear regression is one of the easiest and most popular Machine Learning algorithms. It is a statistical method that is used for predictive analysis. Linear regression makes predictions for continuous/real or numeric variables such as sales, salary, age, product price, etc.
Linear regression algorithm shows a linear relationship between a dependent (y) and one or more independent (y) variables, hence called as linear regression. Since linear regression shows the linear relationship, which means it finds how the value of the dependent variable is changing according to the value of the independent variable.
The linear regression model provides a sloped straight line representing the relationship between the variables.
Types of Linear Regression
Linear regression can be further divided into two types of the algorithm:
    • Simple Linear Regression:
If a single independent variable is used to predict the value of a numerical dependent variable, then such a Linear Regression algorithm is called Simple Linear Regression.
    • Multiple Linear regression:
If more than one independent variable is used to predict the value of a numerical dependent variable, then such a Linear Regression algorithm is called Multiple Linear Regression.
Linear Regression Line
A linear line showing the relationship between the dependent and independent variables is called a regression line. A regression line can show two types of relationship:
    • Positive Linear Relationship:
If the dependent variable increases on the Y-axis and independent variable increases on X-axis, then such a relationship is termed as a Positive linear relationship.
    • Negative Linear Relationship:
If the dependent variable decreases on the Y-axis and independent variable increases on the X-axis, then such a relationship is called a negative linear relationship.

Code:
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
# Load the dataset
diabetes = datasets.load_diabetes()
# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
# Description of the dataset
print(diabetes['DESCR'])
# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]
# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]
# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)
# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)
# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))
print('Nishi Jain-53004230036')
# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
Output:


Practical 4
Aim: 
Apply Multiple Regression on a dataset having a continuous independent variable.
Theory:
Multiple Linear Regression is an extension of Simple Linear regression as it takes more than one predictor variable to predict the response variable.
Some key points about MLR:
    • For MLR, the dependent or target variable(Y) must be the continuous/real, but the predictor or independent variable may be of continuous or categorical form.
    • Each feature variable must model the linear relationship with the dependent variable.
    • MLR tries to fit a regression line through a multidimensional space of data-points.
Assumptions for Multiple Linear Regression:
    • A linear relationship should exist between the Target and predictor variables.
    • The regression residuals must be normally distributed.
    • MLR assumes little or no multicollinearity (correlation between the independent variable) in data.
Applications of Multiple Linear Regression:
There are mainly two applications of Multiple Linear Regression:
    • Effectiveness of Independent variable on prediction:
    • Predicting the impact of changes
Multiple linear regression (MLR) is used to determine a mathematical relationship among
several random variables. In other terms, MLR examines how multiple independent variables
are related to one dependent variable. Once each of the independent factors has been
determined to predict the dependent variable, the information on the multiple variables can be
used to create an accurate prediction on the level of effect they have on the outcome variable.
The model creates a relationship in the form of a straight line (linear) that best approximates
all the individual data points.
Multiple linear regression formula:
Y = 
    • Y = the predicted value of the dependent variable
    • 0 = the y-intercept (value of y when all other parameters are set to 0)
    • 1X1= the regression coefficient (1) of the first independent variable (X1) (a.k.a. the effect that increasing the value of the independent variable has on the predicted y value)
    • … = do the same for however many independent variables you are testing
    • nXn = the regression coefficient of the last independent variable
    • E = model error (a.k.a. how much variation there is in our estimate of y)
A multiple regression considers the effect of more than one explanatory variable on some
outcome of interest. It evaluates the relative effect of these explanatory, or independent,
variables on the dependent variable when holding all the other variables in the model
constant.
Code:
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
diabetes = datasets.load_diabetes()
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
# Description of the dataset
print(diabetes['DESCR'])
print(diabetes.feature_names)
diabetes_X = diabetes_X[:, np.newaxis, 0]
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]
diabetes_y_train = diabetes_y[:-30]
diabetes_y_test = diabetes_y[-30:]
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
diabetes_y_pred = regr.predict(diabetes_X_test)
print('Age')
print("Coefficients: \n", regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test,diabetes_y_pred))
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))
plt.scatter(diabetes_X_test, diabetes_y_test, color='red')
plt.plot(diabetes_X_test, diabetes_y_pred, color='red', linewidth=2, label='Age')
plt.xticks(())
plt.yticks(())
plt.title('Multiple Regression')
#plt.xlabel('Age')
plt.ylabel('Disease Progression')
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
print(diabetes.feature_names)
diabetes_X = diabetes_X[:, np.newaxis, 3]
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]
diabetes_y_train = diabetes_y[:-30]
diabetes_y_test = diabetes_y[-30:]
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)
print('BP')
print('Coefficients: \n', regr.coef_)
print('Mean squared error: %.2f'% mean_squared_error(diabetes_y_test, diabetes_y_pred))
print('Coefficient of determination: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
plt.scatter(diabetes_X_test, diabetes_y_test, color='blue')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=2, label='BP')
plt.xticks(())
plt.yticks(())
plt.title('Multiple Regression')
plt.ylabel('Disease Progression')
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
print(diabetes.feature_names)
diabetes_X = diabetes_X[:, np.newaxis, 2]
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]
diabetes_y_train = diabetes_y[:-30]
diabetes_y_test = diabetes_y[-30:]
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
diabetes_y_pred = regr.predict(diabetes_X_test)
print('BMI')
print('Coefficients: \n', regr.coef_)
print('Mean squared error: %.2f' % mean_squared_error(diabetes_y_test,diabetes_y_pred))
print('Coefficient of determination: %.2f' % r2_score(diabetes_y_test,diabetes_y_pred))
plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='black', linewidth=2, label='BMI')
plt.xticks(())
plt.yticks(())
print('Nishi Jain-53004230036')
plt.title('Multiple Regression')
plt.ylabel('Disease Progression')
plt.legend()
plt.show()
Output:









Practical 5
Aim:
Build a Classification Model (Logistic Regression) 
a. Install relevant package for classification.  
b. Choose classifier for classification problem.  
c. Evaluate the performance of classifier.
Theory:
Classification is defined as the process of recognition, understanding, and grouping of objects and ideas into preset categories i.e. “sub-populations.” With the help of these pre-categorized training datasets, classification in machine learning programs leverage a wide range of algorithms to classify future datasets into respective and relevant categories. Based on training data, the Classification algorithm is a Supervised Learning technique used to categorize new observations. In classification, a program uses the dataset or observations provided to learn how to categorize new observations into various classes or groups. For instance, 0 or 1, red or blue, yes or no, spam or not spam, etc. Targets, labels, or categories can all be used to describe classes. The Classification algorithm uses labelled input data because it is a supervised learning technique and comprises input and output information. A discrete output function (y) is transferred to an input variable in the classification process (x). There are some types of classification algorithms. One of which is logistic regression.
Logistic Regression:
Logistic Regression is one of the simplest and commonly used Machine Learning algorithms for two-class classification. It is easy to implement and can be used as the baseline for any binary classification problem. Its fundamental concepts are also constructive in deep learning. Logistic regression describes and estimates the relationship between one dependent binary variable and independent variables. It is a special case of linear regression where the target variable is categorical in nature. It uses a log of odds as the dependent variable. Logistic Regression predicts the probability of occurrence of a binary event utilizing a logit function.

Code:
import pandas as pd
col_names = ['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']
# Load Dataset
pima = pd.read_csv('diabetes.csv', header=None, names=col_names)
pima.head()

#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

# import the class
from sklearn.linear_model import LogisticRegression
# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16)

# fit the model with data
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
class_names=[0,1] # name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


from sklearn.metrics import classification_report
target_names = ['without diabetes', 'with diabetes']
print(classification_report(y_test, y_pred, target_names=target_names))
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
 

Output:










Practical 6
Aim:
Build a Clustering Model. (K-Means, Agglomerative) 
    a. Select clustering algorithm for unsupervised learning. 
    b. Plot the cluster data using R/python visualizations.
Theory:
Clustering is the task of dividing the un-labelled data or data points into different clusters such that similar data points fall in the same cluster than those which differ from the others. In simple words, the aim of the clustering process is to segregate groups with similar traits and assign them into clusters.
Types of Clustering in Machine Learning
Clustering broadly divides into two subgroups:
    • Hard Clustering: Each input data point either fully belongs to a cluster or not. For instance, in the example above, every customer is assigned to one group out of the ten.
    • Soft Clustering: Rather than assigning each input data point to a distinct cluster, it assigns a probability or likelihood of the data point being in those clusters. For example, in the given scenario, each customer receives a probability of being in any of the ten retail store clusters.
K Means Clustering
K means is an iterative clustering algorithm that aims to find local maxima in each iteration. This algorithm works in these 5 steps:
    1. Specify the desired number of clusters K: Let us choose k=2 for these 5 data points in 2-D space.

    2. Randomly assign each data point to a cluster: Let’s assign three points in cluster 1, shown using red colour, and two points in cluster 2, shown using grey colour.

    3. Compute cluster centroids: The centroid of data points in the red cluster is shown using the red cross, and those in the grey cluster using a grey cross.

    4. Re-assign each point to the closest cluster centroid: Note that only the data point at the bottom is assigned to the red cluster, even though it is closer to the centroid of the grey cluster. Thus, we assign that data point to the grey cluster.

    5. Re-compute cluster centroids: Now, re-computing the centroids for both clusters.

Repeat steps 4 and 5 until no improvements are possible: Similarly, we’ll repeat the 4th and 5th steps until we’ll reach global optima, i.e., when there is no further switching of data points between two clusters for two successive repeats. It will mark the termination of the algorithm if not explicitly mentioned
Code:  1) K-Means clustering model
#Agglomerative clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2,
n_informative=2, n_redundant=0, n_clusters_per_class=1,
random_state=4)
# define the model
model = AgglomerativeClustering(n_clusters=2)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
print('Nishi Jain-53004230036')
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
Output:

Code:  2) Agglomerative Clustering
# Agglomerative clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model
model = AgglomerativeClustering(n_clusters=2)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
print('Nishi Jain-53004230036')
pyplot.show()
Output:












Practical 7
Aim:
Implement SVM classification technique
Theory:
SVM (Support Vector Machine) is a supervised machine learning algorithm. That is why training data is available to train the model. SVM uses a classification algorithm to classify a two-group problem. SVM focus on decision boundary and support vectors.
How SVM Works?
Here, we have two points in two-dimensional space, we have two columns x1 and x2. And we have some observations such as red and green, which are already classified. This is linearly separable data.

But, now how do we derive a line that separates these points? This means a separation or decision boundary is very important for us when we add new points. So to classify new points, we need to create a boundary between two categories, and when in the future we will add new points and we want to classify them, then we know where they belong. Either in a Green Area or Red Area.
So how can we separate these points?
One way is to draw a vertical line between two areas, so anything on the right is Red and anything on the left is Green. Something like that-

However, there is one more way, draw a horizontal line or diagonal line. You can create multiple diagonal lines, which achieve similar results to separate our points into two classes. But our main task is to find the optimal line or best decision boundary. And for this SVM is used. SVM finds the best decision boundary, which helps us to separate points into different spaces. SVM finds the best or optimal line through the maximum margin, which means it has max distance and equidistance from both classes or spaces. The sum of these two classes has to be maximized to make this line the maximum margin.

These, two vectors are support vectors. In SVM, only support vectors are contributing. That is why these points or vectors are known as support vectors. Due to support vectors, this algorithm is called a Support Vector Algorithm (SVM). In the picture, the line in the middle is a maximum margin hyperplane or classifier. In a two-dimensional plane, it looks like a line, but in a multi-dimensional, it is a hyperplane. That is how SVM works.
Code:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
dataset.head()

# Split Dataset into X and Y
X = dataset.iloc[:, [2,3]].values
Y = dataset.iloc[:, 4].values
# Split the X and Y dataset into Training set and Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
# Perform feature scaling- feature scaling helps us to normalize the data within a particular range
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fit SVM to the training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, Y_train)
# Predict the test set results
Y_pred = classifier.predict(X_test)
# Make the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cnf = confusion_matrix(Y_test, Y_pred)
print('Confusion Matrix:')
print(cnf)
print('Accuracy Score:')
accuracy_score(Y_test, Y_pred) 

#Visualise the Test set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1,
stop=X_set[:,0].max()+1,step=0.01),
np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1,step=0.01))
plt.contour(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set==j,0], X_set[Y_set==j,1],
c=ListedColormap(('red','green'))(i),label=j)
plt.title('SVM (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
print('Nishi Jain-53004230036')
plt.show() 
Output:

Practical 8
Aim:
Implement Decision tree classification technique.
Theory:
A decision tree is a non-parametric supervised learning algorithm for classification and regression tasks. It has a hierarchical tree structure consisting of a root node, branches, internal nodes, and leaf nodes. Decision trees are used for classification and regression tasks, providing easy-to-understand models
A decision tree is a hierarchical model used in decision support that depicts decisions and their potential outcomes, incorporating chance events, resource expenses, and utility. This algorithmic model utilizes conditional control statements and is non-parametric, supervised learning, useful for both classification and regression tasks. The tree structure is comprised of a root node, branches, internal nodes, and leaf nodes, forming a hierarchical, tree-like structure.

Code:
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split
# Import train_test_split function
from sklearn import metrics
#Import scikit-learn metrics module for accuracy calculation
#Loading Data
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv("diabetes.csv", header=None, names=col_names)
pima.head()

#Feature Selection
#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable

#Splitting Data
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# 70% training and 30% test

#Building Decision Tree Model
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
#Evaluating the Model
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Visualizing Decision Trees
!pip install graphviz
!pip install pydotplus
from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO
from six import StringIO
from IPython.display import Image
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
filled=True, rounded=True,
special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())
Output:



Practical 9
Aim: Naïve Bayes Implementation
Theory: 
Naive Bayes is a machine learning algorithm that is used by data scientists for classification. The naive Bayes algorithm works based on the Bayes theorem. Before explaining Naive Bayes, first, we should discuss Bayes Theorem. Bayes theorem is used to find the probability of a hypothesis with given evidence. This beginner-level article intends to introduce you to the Naive Bayes algorithm and explain its underlying concept and implementation.

In this equation, using Bayes theorem, we can find the probability of A, given that B occurred. A is the hypothesis, and B is the evidence.
P(B|A) is the probability of B given that A is True.
P(A) and P(B) are the independent probabilities of A and B.
Naive Bayes Classifier Algorithm
The Naive Bayes classifier algorithm is a machine learning technique used for classification tasks. It is based on Bayes’ theorem and assumes that features are conditionally independent of each other given the class label. The algorithm calculates the probability of a data point belonging to each class and assigns it to the class with the highest probability.
Naive Bayes is known for its simplicity, efficiency, and effectiveness in handling high-dimensional data. It is commonly used in various applications, including text classification, spam detection, and sentiment analysis.
Naive Bayes Theorem:
We are taking a case study in which we have the dataset of employees in a company, our aim is to create a model to find whether a person is going to the office by driving or walking using the salary and age of the person.

In the above image, we can see 30 data points in which red points belong to those who are walking and green belong to those who are driving. Now let’s add a new data point to it. Our aim is to find the category that the new point belongs to

Note that we are taking age on the X-axis and Salary on the Y-axis. We are using the Naive Bayes algorithm to find the category of the new data point. For this, we have to find the posterior probability of walking and driving for this data point. After comparing, the point belongs to the category having a higher probability.
In the above image, we can see 30 data points in which red points belong to those who are walking and green belong to those who are driving. Now let’s add a new data point to it. Our aim is to find the category that the new point belongs to
The posterior probability of walking for the new data point is:

and that for the driving is:

Steps Involved in the Naive Bayes Classifier Algorithm:
Step 1: We have to find all the probabilities required for the Bayes theorem for the calculation of posterior probability.
Step 2: Similarly, we can find the posterior probability of Driving, and it is 0.25
Step 3: Compare both posterior probabilities. When comparing the posterior probability, we can find that P(walks|X) has greater values, and the new point belongs to the walking category.
Implementation of Naive Bayes
We are using the Social network ad dataset. The dataset contains the details of users on a social networking site to find whether a user buys a product by clicking the ad on the site based on their salary, age, and gender.

Code:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
print('Nishi Jain-53004230036')
plt.show()
# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
Output:
  
