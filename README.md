
# Hugging Face Datasets Text Quality Analysis

![](./screenshot.png)

The purpose of this repository is to let people evaluate the quality of datasets on Hugging Face. It retrieves parquet files from Hugging Face, identifies the junk data, duplication, contamination, biased content, and other quality issues within a given dataset.

## Related Resources

* [Google Colab Notebook](https://colab.research.google.com/drive/1c8rWB2gtUrBHQcmmvA_NAXxc7Cexn1vM?usp=sharing)


## Instructions

1.Prerequisites
Note that the code only works `Python >= 3.9` and `streamlit >= 1.23.1`

```
$ conda create -n datasets-quality python=3.9
$ conda activate datasets-quality
```

2. Install dependencies
```
$ cd HuggingFace-Datasets-Text-Quality-Analysis
$ pip install -r requirements.txt
```

3.Run Streamlit application
```
python -m streamlit run app.py
```

## Todos

- [ ] Introduce more dimensions to evaluate the dataset quality
- [ ] Optimize the deduplication example using parallel computing technique
- [ ] More robust junk data detecting 
- [ ] Use a classifier to evaluate the quality of data
- [ ] Test the code with larger dataset in a cluster environment
- [ ] More data frame manipulation backend: e.g. Dask
