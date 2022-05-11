```python
# !pip install tashaphyne
# !pip install emoji
```


```python
import pandas as pd 
import numpy as np
import nltk
from tashaphyne.stemming import ArabicLightStemmer
import warnings as wr
import regex as re
from sklearn.model_selection import train_test_split
import emoji
import matplotlib.pyplot as plt
wr.filterwarnings("ignore")
```


```python
def Cleaning(copy_data):
    copy_data['class']=copy_data['class'].replace({'pos':0,'neg':1})
    copy_data.drop(copy_data.columns[copy_data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True) #remove unnamed columns . 
    copy_data['sentenses_len'] = 0
    copy_data['puretext'] = 0 
    copy_data['#english_words'] = 0 
    copy_data['#hashtags'] = 0 
    copy_data['#mentioning'] = 0 
    copy_data['#hyperlinks'] = 0 
    copy_data['#numbers'] = 0
    copy_data['#emojis']=0
    
    return copy_data
```


```python
def tokenization(x,idx): # take data copy
        tokens = nltk.word_tokenize(x['text'][idx])
        return tokens
def segmentation(x,idx): # take data copy 
    s_tokens = nltk.data.load('tokenizers/punkt/english.pickle')
    sentens = s_tokens.tokenize(x['text'][idx])
    x['sentenses_len'][i] = len(sentens) #save length of sentenses in the csv "data set "  file  
    # print(len(sentens))    
def drop_stop_words(x): #take data copy
    arb_stop_words = set(nltk.corpus.stopwords.words("arabic"))
    tokensOfpureTextWithoutstop=[token for token in x if token not in arb_stop_words]
    return tokensOfpureTextWithoutstop
def stemming_Light(x):
    ArListem = ArabicLightStemmer()
    stemming_Light =[ArListem.light_stem(token) for token in x]
    return stemming_Light
def stemming(x):
    st = nltk.ISRIStemmer()
    stemming_root =[st.stem(token) for token in x]
    return stemming_root
```


```python
def removing_mentioning(text):
    return re.sub(r"@[a-zA-Z0-9]+",'',text)
def removing_hashtags(text):
    return re.sub(r"#[a-zA-Z0-9أ-ى]+",'',text)
def remove_newlines_tabs(text):
    return ' '.join(text.replace('\n', ' ').replace('\t',' ').split())
def remove_numbers(text):
    return re.sub('\d+', '', text)
def remove_links(text):
    return re.sub(r'https?:\/\/.*[\r\n]*', '', text)
def remove_emojis(text):
    return emoji.replace_emoji(text, '').replace('☻', ' ')
def remove_english(text):
    return re.sub('[A-Za-z]+', '', text)
```


```python
def spliting (data_,target,test_size=0.33):
    X_train, X_test, y_train, y_test = train_test_split(data_, target, test_size=test_size, random_state=44, shuffle =True)
    return X_train, X_test, y_train, y_test

```

# Data set 


```python
data=pd.read_csv('data_set.csv')
data = Cleaning(data)
```


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 45274 entries, 0 to 45273
    Data columns (total 10 columns):
     #   Column          Non-Null Count  Dtype 
    ---  ------          --------------  ----- 
     0   class           45274 non-null  int64 
     1   text            45274 non-null  object
     2   sentenses_len   45274 non-null  int64 
     3   puretext        45274 non-null  int64 
     4   #english_words  45274 non-null  int64 
     5   #hashtags       45274 non-null  int64 
     6   #mentioning     45274 non-null  int64 
     7   #hyperlinks     45274 non-null  int64 
     8   #numbers        45274 non-null  int64 
     9   #emojis         45274 non-null  int64 
    dtypes: int64(9), object(1)
    memory usage: 3.5+ MB



```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>text</th>
      <th>sentenses_len</th>
      <th>puretext</th>
      <th>#english_words</th>
      <th>#hashtags</th>
      <th>#mentioning</th>
      <th>#hyperlinks</th>
      <th>#numbers</th>
      <th>#emojis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>اعترف ان بتس كانو شوي شوي يجيبو راسي لكن اليوم...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>توقعت اذا جات داريا بشوفهم كاملين بس لي للحين ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>#الاهلي_الهلال اكتب توقعك لنتيجة لقاء الهلال و...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>نعمة المضادات الحيوية . تضع قطرة💧مضاد بنسلين ع...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>الدودو جايه تكمل علي 💔</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>sentenses_len</th>
      <th>puretext</th>
      <th>#english_words</th>
      <th>#hashtags</th>
      <th>#mentioning</th>
      <th>#hyperlinks</th>
      <th>#numbers</th>
      <th>#emojis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>45274.000000</td>
      <td>45274.0</td>
      <td>45274.0</td>
      <td>45274.0</td>
      <td>45274.0</td>
      <td>45274.0</td>
      <td>45274.0</td>
      <td>45274.0</td>
      <td>45274.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.497283</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.499998</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.pie(data['class'].value_counts(), labels=['ham','spam'], counterclock=False, shadow=True, autopct='%1.1f%%', radius=1, startangle=0)
plt.show()
```


    
![png](output_11_0.png)
    


### Preprocessing


```python
for i in range(0,data.shape[0]):
    tokens = tokenization(data , i )
    data['#english_words'][i] = len(re.findall(r'[A-Za-z]+', data['text'][i]))
    data['#hyperlinks'][i] = len(re.findall(r'https?:\/\/.*[\r\n]*', data['text'][i]))
    data['#numbers'][i] = len(re.findall('\d+', data['text'][i]))
    data['#hashtags'][i] = len(re.findall(r"#[a-zA-Z0-9أ-ى]+", data['text'][i]))
    data['#mentioning'][i] = len(re.findall(r"@[a-zA-Z0-9]+", data['text'][i]))
    data['#emojis'][i]=emoji.emoji_count(data['text'][i])
    segmentation(data,i)
    pure_tokens = drop_stop_words(tokens)
#     pure_tokens = stemming_Light(pure_tokens) #-> light_streamer , we don't use it . 
    pure_tokens = stemming(pure_tokens)
    data['puretext'][i] = ' '.join(pure_tokens)
    # print(tokens)
```


```python
data.drop('text',axis=1,inplace=True)
```


```python
data['puretext'] = data['puretext'].apply(removing_mentioning)
data['puretext'] = data['puretext'].apply(removing_hashtags)
data['puretext'] = data['puretext'].apply(remove_numbers)
data['puretext'] = data['puretext'].apply(remove_links)
data['puretext'] = data['puretext'].apply(remove_emojis)
data['puretext'] = data['puretext'].apply(remove_english)
```


```python
copy_data=data.iloc[:,[1,3,4,5,6,7,8]]
```


```python
copy_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sentenses_len</th>
      <th>#english_words</th>
      <th>#hashtags</th>
      <th>#mentioning</th>
      <th>#hyperlinks</th>
      <th>#numbers</th>
      <th>#emojis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



#### Counting 


```python
from sklearn.feature_extraction.text import CountVectorizer
counting_vec =CountVectorizer(binary=False)
features = counting_vec.fit_transform(data['puretext']).astype('int8')
counting =pd.DataFrame(features.toarray(), columns= counting_vec.vocabulary_.keys())
```


```python
X_train, X_test, y_train, y_test=spliting (counting,data['class'],test_size=0.4)
comparing_train=[]
comparing_test=[]
```


```python
from sklearn.svm import LinearSVC

linear_svc=LinearSVC(C=.03)

linear_svc.fit(X_train,y_train)

print('SVCModel Train Score is : ' , linear_svc.score(X_train, y_train))
print('SVCModel Train Score is : ' , linear_svc.score(X_test, y_test))

comparing_train.append(linear_svc.score(X_train, y_train))
comparing_test.append(linear_svc.score(X_test, y_test))
```

    SVCModel Train Score is :  0.8313208658518627
    SVCModel Train Score is :  0.7500276090557703



```python
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
plot_confusion_matrix(linear_svc, X_test, y_test) 
plt.show()
```


    
![png](output_22_0.png)
    


### Binary Encoding


```python
from sklearn.feature_extraction.text import CountVectorizer
counting_vec =CountVectorizer(binary=True)
features = counting_vec.fit_transform(data['puretext']).astype('int8')
counting =pd.DataFrame(features.toarray(), columns= counting_vec.vocabulary_.keys())
```


```python
X_train, X_test, y_train, y_test=spliting (counting,data['class'],test_size=0.4)

```


```python
linear_svc=LinearSVC(C=.03)

linear_svc.fit(X_train,y_train)

print('SVCModel Train Score is : ' , linear_svc.score(X_train, y_train))
print('SVCModel Train Score is : ' , linear_svc.score(X_test, y_test))

```

    SVCModel Train Score is :  0.83102635841555
    SVCModel Train Score is :  0.7506902263942573



```python
comparing_train.append(linear_svc.score(X_train, y_train))
comparing_test.append(linear_svc.score(X_test, y_test))
```


```python
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
plot_confusion_matrix(linear_svc, X_test, y_test) 
plt.show()
```


    
![png](output_28_0.png)
    


### TF-IDF


```python
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
features = vec.fit_transform(data['puretext']).astype('float32')
tf_idf =pd.DataFrame(features.toarray(), columns= vec.vocabulary_.keys())

```


```python
X_train, X_test, y_train, y_test=spliting (tf_idf,data['class'],test_size=0.4)

```


```python
linear_svc=LinearSVC(C=.03)

linear_svc.fit(X_train,y_train)

print('SVCModel Train Score is : ' , linear_svc.score(X_train, y_train))
print('SVCModel Train Score is : ' , linear_svc.score(X_test, y_test))

comparing_train.append(linear_svc.score(X_train, y_train))
comparing_test.append(linear_svc.score(X_test, y_test))
```

    SVCModel Train Score is :  0.7809600942423797
    SVCModel Train Score is :  0.7308669243511872



```python
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
plot_confusion_matrix(linear_svc, X_test, y_test) 
plt.show()
```


    
![png](output_33_0.png)
    



```python
import matplotlib.pyplot as plt
```


```python
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
x=['Counting','encoding','TF-IDF']
plt.bar(x,comparing_train,color='#FFA726')
plt.title('Accuracy on Training')

plt.subplot(1,2,2)
plt.bar(x,comparing_test,color='yellowgreen')
plt.title('Accuracy on Testing')


plt.show()
```


    
![png](output_35_0.png)
    



```python

```


```python

```
