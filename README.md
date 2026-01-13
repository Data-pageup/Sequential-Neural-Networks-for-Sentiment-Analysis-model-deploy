# Sequential Neural Networks for Sentiment Analysis

### [ Live Demo] (https://sequentialneuralnetworks-sentimentanalysis.streamlit.app/)

## Overview
This project implements **Sequential Deep Learning Models** such as **RNN**, **LSTM**, and **GRU** to perform **Sentiment Analysis** on textual data.  
The goal is to classify a given text (such as a movie review or news article) as either **positive** or **negative** by understanding the contextual meaning of words using neural sequence models.
 
---

## Objectives
- To preprocess and clean textual data effectively for NLP tasks.  
- To represent words as numerical vectors for deep learning models.  
- To build and compare the performance of **RNN**, **LSTM**, and **GRU** architectures.  
- To evaluate model accuracy, training loss, and generalization ability.  
- To deploy the best-performing model using an interactive **Streamlit** web interface.

---

## Dataset Description
The dataset consists of two categories of text documents — **real** and **fake** news articles (or positive/negative reviews).  
Each record contains the following fields:

| Column | Description |
|---------|-------------|
| `title` | Headline or title of the article/review |
| `text` | Main content of the text |
| `subject` | Category or topic |
| `date` | Published date |
| `label` | Target variable (`1` = real/positive, `0` = fake/negative) |

Initially, both *fake* and *real* datasets were combined into a single DataFrame, followed by data cleaning and text preprocessing.

---

## Methodology

### 1. Data Preprocessing
Steps performed:
- Removal of punctuation, special characters, and stopwords  
- Conversion to lowercase  
- Tokenization and lemmatization  
- Sequence padding and truncation to a fixed length  
- Train–test split for model evaluation  

This ensures that all text inputs are of uniform length and numerical form before feeding into sequential models.

---

### 2. Text Vectorization
Since neural networks cannot process raw text directly, words were converted into numeric sequences using:
- **Tokenizer** (Keras): Converts text to integer sequences  
- **Padding:** Ensures equal input length using `pad_sequences`  
- **Embedding Layer:** Maps each word index to a dense vector representation capturing semantic meaning  

---

### 3. Model Architectures

#### (a) Recurrent Neural Network (RNN)
- Sequentially processes words, capturing limited contextual information.
- Struggles with **long-term dependencies** due to vanishing gradients.

#### (b) Long Short-Term Memory (LSTM)
- Overcomes RNN’s vanishing gradient problem with memory cells and gates.
- Efficient in remembering long-term dependencies and context flow.

#### (c) Gated Recurrent Unit (GRU)
- A simplified version of LSTM with fewer parameters.
- Faster to train while achieving comparable performance.

---

### 4. Model Training
All models were compiled using:
```python
loss = 'binary_crossentropy'
---
optimizer = 'adam'
metrics = ['accuracy']
