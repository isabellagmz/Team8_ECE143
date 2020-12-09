# Covid-19 Sentiment Analysis using Tweets
# ECE 143- Final Project - Group 8

**Steps to run our code:-**
1. Download our Github repository
2. Download the following dataset from Kaggle:- 
- https://www.kaggle.com/smid80/coronavirus-covid19-tweets-early-april
3. Make sure the dataset directory is in the directory containing the rest of the code. And the directory name for the dataset is archive

**Note** - You can also keep the dataset at some other place, but you will need to add the path of the dataset to line 17 in our main.py as well the Jupyter Notebook. The line of code that you will need to change is the following:-

```for dirname, _, filenames in os.walk('<Put Full Path here>'):```

4. Make sure you have the following libraries installed, you can do a ```pip install <library-name>``` or ```conda install <library-name>``` depending on the Python distribution you are using and the package managers that come with it

- numpy
- pandas
- matplotlib
- seaborn
- textblob
- collections
- re
- os
- wordcloud
- warnings
- jupyter lab(required to run the jupyter notebook)
  
5. Run the file main.py
6. You can also run the Jupyter Notebook cell by cell for a better visual experience.
