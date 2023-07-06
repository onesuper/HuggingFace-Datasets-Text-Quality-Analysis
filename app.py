import streamlit as st
import requests
import os

enable_xorbits = False


if enable_xorbits:
    import xorbits.pandas as pd
    import xorbits.numpy as np
    import xorbits
    xorbits.init(n_worker=1, n_cpu=2)
else:
    import pandas as pd
    import numpy as np

st.set_page_config(page_title="Analyzing Text Corpus on Hugging Face", page_icon=":bar_chart:", layout="wide")
st.sidebar.title('A Tool for Analyzing Text Corpus on Hugging Face')
st.sidebar.markdown(
    '''
This tool retrieves parquet files from Hugging Face, identifies and quantifies
junk data, duplication, contamination, and biased content in dataset using Pandas Dataframe,
and accelerates time-consuming processes using Xorbits.
    '''
)

st.sidebar.header("Please Paste The HF Dataset Name Here:")

#@st.cache_data
def load_dataset(j, name, fraction):

    if not os.path.exists('train.gzip'):
        with st.spinner('Downloading file from remote server'):
            import pandas
            train_urls = [f['url'] for f in j['parquet_files'] if f['config'] == name and f['split'] == 'train']
            train_dataset = pandas.concat([pandas.read_parquet(url, engine='pyarrow') for url in train_urls], ignore_index=True)
            train_dataset.to_parquet('train.gzip')

    if not os.path.exists('test.gzip'):
        with st.spinner('Downloading file from remote server'):
            import pandas
            test_urls = [f['url'] for f in j['parquet_files'] if f['config'] == name and f['split'] == 'validation']
            test_dataset = pandas.concat([pandas.read_parquet(url, engine='pyarrow') for url in test_urls], ignore_index=True)
            test_dataset.to_parquet('test.gzip')

    train_dataset = pd.read_parquet('train.gzip', engine='pyarrow')

    test_dataset = pd.read_parquet('test.gzip', engine='pyarrow')

    dataset = {
        "train": train_dataset[:int(len(train_dataset)*fraction)],
        "test": test_dataset[:int(len(test_dataset)*fraction)],
    }
    
    return dataset


def get_hugging_face_dataset(name):
    r = requests.get("https://datasets-server.huggingface.co/parquet?dataset=" + dataset_name)
    return r.json()


dataset_name = st.sidebar.text_input('Dataset Name', 'blog_authorship_corpus')

with st.spinner('Loading meta'):
    hf_datasets = get_hugging_face_dataset(dataset_name)
    subsets = set([x['config'] for x in hf_datasets['parquet_files']])
    subset_option = st.sidebar.selectbox("Choose a subset", subsets)
    sample_rate_option = st.sidebar.slider('Select sample rate', value=0.05, min_value=0.1, max_value=1.0, step=0.1)

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Introduction", "Junk Data🤖", "Contamination🧹", "Short Documents🌐", "Biased Content🛡️", "Duplication🔍"])
with tab0:

    st.markdown(
    '''
### Why this matters?
LLMs are trained on immense datasets to have a broader understanding of language and improve
their performance. 
However, the quality of the datasets can affect the performance and biases of the models. 

Large datasets often have quality issues, so practitioners need to clean and preprocess
the data to remove biases, noise, and toxicity.

This tool illustrates how to analyze and quantify the quality
of any text corpus on [Hugging Face](https://huggingface.co/blog/hub-duckdb) using pandas.

### Data Preparation
#### 1.Retrieving parquet files from Hugging Face Dataset Server
First you can get the list of the Parquet files URLs with a simple HTTP call.
```python
r = requests.get("https://datasets-server.huggingface.co/parquet?dataset=blog_authorship_corpus")
j = r.json()
urls = [f['url'] for f in j['parquet_files'] if f['split'] == 'train']
urls
['https://huggingface.co/datasets/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/blog_authorship_corpus/blog_authorship_corpus-train-00000-of-00002.parquet',
 'https://huggingface.co/datasets/blog_authorship_corpus/resolve/refs%2Fconvert%2Fparquet/blog_authorship_corpus/blog_authorship_corpus-train-00001-of-00002.parquet']
```
    
#### 2.Read URLs into Pandas Dataframe

Use the pandas library to read multiple Parquet files from a list of URLs and concatenate 
them into a single DataFrame:
```python
import pandas as pd
parts = pd.read_parquet(url) for url in urls]
df = pd.concat(parts, ignore_index=True)
```

#### 3.Addressing out-of-memory & performance issues
Since the pandas library makes use of in-memory data structures to store and operate on data,
 which means that if the dataset your read from hugging face is too large to fit in memory, 
 it will cause an error on pandas. So we use [Xorbits](https://xorbits.io) for dealing with
 larger datasets and use my laptop's cpu more efficiently. 


The use of Xorbits is as simple as: 

 ```python
import xorbits.pandas as pd
import xorbits.numpy as np
 ```

 ---
'''
    )
    with st.expander("View raw data"):
        with st.spinner("Loading..."):
            datasets = load_dataset(hf_datasets, subset_option, sample_rate_option)

            train, test = st.tabs([
                "Train (%d rows)" % len(datasets['train']),
                "Test (%d rows)" % len(datasets['test'])
            ])

            train.dataframe(datasets['train'][:20])
            test.dataframe(datasets['test'][:20])

with tab1:
    st.header("Junk Data")


    st.markdown('''
Large-scale datasets often contain an uneven distribution of text representation, which includes
a significant amount of nonsensical and boilerplate text - such as HTML tags.

The presence of such "noise" or irrelevant content in the dataset is detrimental to the 
training of predictive models, specifically those that operate by predicting the next token based on all previous ones.
Therefore, it's crucial to clean the dataset and remove these undesired elements prior to the training phase.    

This piece of Python code calculated a measure of "impurity" in text documents, and then computing
 the proportion of documents that exceed a certain impurity threshold. It defines a compiled regular expression that matches
   any of the following suspicious characters: `&, #, <, >, {, }, [, ]`.
        ''')


    metrics, code = st.tabs(['Metrics', 'Code'])

    with metrics:

        with st.spinner('Calculating impurity ratio...'):
            df = datasets['train']

            import re
            RE_SUSPICIOUS = re.compile(r'[&#<>{}\[\]\\]')

            def impurity(text, min_len=10):
                """returns the share of suspicious characters in a text"""
                if text == None or len(text) < min_len:
                    return 0
                else:
                    return len(RE_SUSPICIOUS.findall(text))/len(text)

            df['impurity'] = df['text'].apply(impurity, min_len=10)
            total_num_docs = len(df)
            impurity_num_docs = len(df[df['impurity']  > 0.01])
            impurity_ratio = impurity_num_docs / total_num_docs

            col1, col2, col3 = st.columns(3)
            col1.metric(label="Junk Doc Count", value="%d" % impurity_num_docs)
            col2.metric(label="Total Doc Count", value="%d" % total_num_docs)
            col3.metric(label="Junk Doc Ratio", value="%.2f%%" % (impurity_ratio * 100))

        st.dataframe(df[['text', 'impurity']].sort_values(by='impurity', ascending=False)[:20])
    with code:
        st.code(
            '''
            import re

            RE_SUSPICIOUS = re.compile(r'[&#<>{}\[\]\\]')

            def impurity(text, min_len=10):
                """returns the share of suspicious characters in a text"""
                if text == None or len(text) < min_len:
                    return 0
                else:
                    return len(RE_SUSPICIOUS.findall(text))/len(text)


            df['impurity'] = df['text'].apply(impurity, min_len=10)
            total_num_docs = len(df)
            impurity_num_docs = len(df[df['impurity']  > 0.001])
            impurity_ratio = impurity_num_docs / total_num_docs
            '''
        )


with tab2:    
    st.header('Contamination')

    st.markdown('''
Typically, ensuring the segregation of training and testing data is rather straightforward in machine learning.
However, things become complicated in the context of large language models
where both the training and benchmarking datasets are collected from the internet.

For instance, the performance evaluation of a large language model using benchmark data
(like question-answer pairs) can be significantly affected if the benchmark data also features
in the model's training set. The procedure of eliminating instances from the training datasets that intersect with
 the existing benchmarking datasets is called "decontamination".


This Python code below is being used to quantify the contamination problem lying in the datasets,
 i.e., the proportion of documents in the test set that also appear in the training set using N-grams.

The approach here is from GPT-3 paper. OpenAI defined a test document as contaminated 
if any N-gram overlap existed with any training document. 
(They used a range of N values between 8 and 13 depending on dataset.)
When constructing the WebText dataset, OpenAI researchers decontaminated the data by
eliminating all Wikipedia content from the training set. This was necessary as Wikipedia
data was heavily used in their benchmark datasets.
        ''')

    metrics, code = st.tabs(['Metrics', 'Code'])
    with metrics:

        with st.spinner('Calculating contamination ratio...'):
            train_dataset = datasets['train']
            test_dataset = datasets['test']

            from nltk import ngrams
            from datasketch import MinHash, MinHashLSH

            def process_data(df):
                minhashes = {}
                for idx, r in df.iterrows():
                    minhash = MinHash(num_perm=128)
                    for d in ngrams(r['text'], 13):
                        s = "".join(d).encode('utf-8')
                        minhash.update(s)
                    minhashes[idx] = minhash
                return minhashes

            train_minhashes = process_data(train_dataset)
            test_minhashes = process_data(test_dataset)

            lsh = MinHashLSH(threshold=0.8, num_perm=128)

            for idx, minhash in train_minhashes.items():
                lsh.insert(idx, minhash)

            duplicates_count = 0
            for idx, minhash in test_minhashes.items():
                result = lsh.query(minhash)
                if len(result) > 0:
                    duplicates_count += 1

            train_dataset_count = len(train_dataset)            
            test_dataset_count = len(test_dataset)
            contaminate_ratio = duplicates_count / test_dataset_count

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(label="Train Set Size", value="%d" % train_dataset_count)
            col2.metric(label="Test Set Size", value="%d" % test_dataset_count)
            col3.metric(label="Overlapped Docs", value="%d" % duplicates_count)
            col4.metric(label="Contaminated Ratio", value="%.2f%%" % (contaminate_ratio * 100))
    with code:
        st.code(
            '''
            from nltk import ngrams
            from datasketch import MinHash, MinHashLSH

            def process_data(df):
                minhashes = {}
                for idx, r in df.iterrows():
                    minhash = MinHash(num_perm=128)
                    for d in ngrams(r['text'], 13):
                        s = "".join(d).encode('utf-8')
                        minhash.update(s)
                    minhashes[idx] = minhash
                return minhashes

            train_minhashes = process_data(train_dataset)
            test_minhashes = process_data(test_dataset)

            lsh = MinHashLSH(threshold=0.8, num_perm=128)

            for idx, minhash in train_minhashes.items():
                lsh.insert(idx, minhash)

            duplicates_count = 0
            for idx, minhash in test_minhashes.items():
                result = lsh.query(minhash)
                if len(result) > 0:
                    duplicates_count += 1

            train_dataset_count = len(train_dataset)            
            test_dataset_count = len(test_dataset)
            contaminate_ratio = duplicates_count / test_dataset_count         
            '''
        )

with tab3:
    st.header("Too-Short Documents")

    st.markdown('''
The aim of language modeling is to master the generation of text based on preceding tokens. 
In this scenario, eliminating extremely brief documents (text consisting of fewer than approximately
 100 tokens) from the corpus could aid in the reduction of noise, by producing contiguous text to 
 model dependencies within the text.


 Use the Hugging Face Transformers library to tokenize text and then calculate the proportion
 of documents that are "too short" in a dataset. This example converts text into tokens that the BERT
 model can understand. Choose a tokenizer for your model.
    ''')
    metrics, code = st.tabs(['Metrics', 'Code'])

    with metrics:
        with st.spinner('Calculating too-short ratio...'):
            from transformers import BertTokenizer

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            df = datasets['train']
            # Create a new column with the number of tokens for each text
            df['text_length'] = df['text'].apply(lambda text: len(tokenizer.tokenize(text)))
            total_num_docs = len(df)
            too_short_docs = len(df[df['text_length'] < 100]) 
            too_short_doc_ratio = too_short_docs / total_num_docs

            col1, col2, col3 = st.columns(3)
            col1.metric(label="Too-Short Doc Count", value="%d" % too_short_docs)
            col2.metric(label="Total Doc Count", value="%d" % total_num_docs)
            col3.metric(label="Too Short Doc Ratio", value="%.2f%%" % (too_short_doc_ratio * 100))

    #         col1, _ = st.columns([2, 1])

    #         import seaborn as sns
    #         import matplotlib.pyplot as plt
    #         fig, ax = plt.subplots(figsize=(10, 5))
    #         ax.set_title('Distribution of text length (in tokens)')
    #         sns.histplot(data=df, x='text_length', ax=ax)
    #         plt.axvline(100, color='r', linestyle='--')
    #         col1.pyplot(fig)
    with code:
        st.code(
        '''
        from transformers import BertTokenizer

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        df = datasets['train']
        # Create a new column with the number of tokens for each text
        df['text_length'] = df['text'].apply(lambda text: len(tokenizer.tokenize(text)))
        total_num_docs = len(df)
        too_short_docs = len(df[df['text_length'] < 100]) 
        too_short_doc_ratio = too_short_docs / total_num_docs            
        '''
        )

with tab4:    
    st.header('Toxic Content')
    st.markdown('''
It is crucial in the training of language models to be vigilant and potentially apply tools 
to exclude toxic content from the pre-training datasets. This practice helps to
prevent the models from demonstrating bias or generating detrimental content in subsequent applications.

One approach to address this issue is by scanning the text for **offensive words**. 
For instance, the creators of the C4 dataset have implemented such a 
filtering mechanism. The follow code references this 
[word ](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/blob/master/en) that they open source.

The following code utilizes the word list to quantify the "biased content ratio" in the dataset.

    ''')

    metrics, code = st.tabs(['Metrics', 'Code'])
    with metrics:
        with st.spinner('Calculating toxic ratio...'):
            df = datasets['train']

            with open('./List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words', 'r') as f:
                lines = f.readlines()

            banned_words = [line.rstrip('\n') for line in lines]
            df['banned_words_in_text'] = df['text'].apply(lambda text: [word for word in banned_words if word in text.lower().split()])    
            df['matches'] = df['banned_words_in_text'].apply(lambda words: len(words) > 0)
            total_num_docs = len(df)
            biased_num_docs = df['matches'].sum()
            biased_content_ratio = biased_num_docs / total_num_docs
            col1, col2, col3 = st.columns(3)

            col1.metric(label="Total Doc Count", value="%d" % total_num_docs)
            col2.metric(label="Biased Doc Count", value="%d" % biased_num_docs)
            col3.metric(label="Biased Ratio", value="%.2f%%" % (biased_content_ratio * 100))
            st.dataframe(df[df['matches']][['text', 'banned_words_in_text']][:20])
    with code:
        st.code(
            '''
            with open('./List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words', 'r') as f:
                lines = f.readlines()

            banned_words = [line.rstrip('\n') for line in lines]
            df['banned_words_in_text'] = df['text'].apply(lambda text: [word for word in banned_words if word in text.lower().split()])    
            total_num_docs = len(df)
            df['matches'] = df['banned_words_in_text'].apply(lambda words: len(words) > 0)
            biased_num_docs = df['matches'].sum()
            biased_content_ratio = biased_num_docs / total_num_docs            
            '''
        )



with tab5:
    st.header("Duplication")

    st.markdown(
        '''
When datasets are created by scraping raw text from the Internet, this will often result
in the same sequences being repeated multiple times. [This paper](https://arxiv.org/abs/2107.06499) mentions a single 50 word sequence that is 
repeated in the C4 dataset 60,000 times.

Deduplication helps prevent models from outputting verbatim training data when
there are many duplicates, and makes models less vulnerable to privacy attacks.
Deduplication can also improve model training efficiency and prevent benchmark contamination.

### Tools & Tutorials

 The [GPT-3](https://arxiv.org/abs/2005.14165) paper mentions they fuzzily deduplicated documents 
 within each dataset using Spark’s MinHashLSH implementation with 10 hashes.

[deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets) 
is an ExactSubstr deduplication implementation (written in Rust) along with the scripts to 
perform ExactSubstr deduplication and inspect the results (written in Python).

 [datasketch](https://github.com/ekzhu/datasketch) gives you probabilistic data structures that 
 can process and search very large amount of data super fast, with little loss of accuracy.

[This article](https://huggingface.co/blog/dedup) provides a MinHash walkthrough to demonstrate
how to implement a parallelel deduplication.
 
 The following code uses the [datasketch](https://github.com/ekzhu/datasketch) library and LSH (Locality Sensitive Hashing) 
 to deduplicate the dataset. For each text in the DataFrame, it creates a query MinHash object
 and performs a query on the LSH index to find similar documents.

 It worths to mention that the de-duplication process usually requires a lot of computational resources
 (CPU and RAM) due to the size of web crawl datasets and it's therefore recommended to run such
 computations in distributed settings.
        '''
    )


    metrics, code = st.tabs(['Metrics', 'Code'])
    with metrics:
        with st.spinner('Calculating duplication ratio...'):
            df = datasets['train']

            from datasketch import MinHashLSH, MinHash

            lsh = MinHashLSH(threshold=0.85, num_perm=128)

            for i, text in enumerate(df['text']):
                minhash = MinHash(num_perm=128)
                for word in text.split():
                    minhash.update(word.encode('utf-8'))
                lsh.insert(str(i), minhash)

            unique_documents = set()

            for i, text in enumerate(df['text']):
                query_minhash = MinHash(num_perm=128)
                for word in text.split():
                    query_minhash.update(word.encode('utf-8'))
                results = lsh.query(query_minhash)
                unique_documents.add(results[0])

            total_unique_documents = len(unique_documents)
            total_documents = len(df)
            duplication_ratio = (total_documents - total_unique_documents) / total_documents

            col1, col2, col3 = st.columns(3)
            col2.metric(label="Total Documents", value="%d" % total_documents)
            col1.metric(label="Unique Docs Pairs", value="%d" % total_unique_documents)
            col3.metric(label="Duplication Ratio", value="%.2f%%" % (duplication_ratio * 100))
    with code:
        st.code(
            '''
            from datasketch import MinHashLSH, MinHash

            lsh = MinHashLSH(threshold=0.85, num_perm=128)

            for i, text in enumerate(df['text']):
                minhash = MinHash(num_perm=128)
                for word in text.split():
                    minhash.update(word.encode('utf-8'))
                lsh.insert(str(i), minhash)

            unique_documents = set()

            for i, text in enumerate(df['text']):
                query_minhash = MinHash(num_perm=128)
                for word in text.split():
                    query_minhash.update(word.encode('utf-8'))
                results = lsh.query(query_minhash)
                unique_documents.add(results[0])

            total_unique_documents = len(unique_documents)
            total_documents = len(df)
            duplication_ratio = (total_documents - total_unique_documents) / total_documents
            '''
        )