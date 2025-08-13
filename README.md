# queuerious_detector: Queue Classification for Customer Support Tickets

This repository contains the code and pipeline for automating the classification of customer support tickets into the correct support queue. The project is part of our SAIDS Capstone work and uses natural language processing (NLP) techniques to improve ticket routing accuracy. 

Team Members: 
Allen, D.      toobrite@umich.edu
Petrov, A.     alynap@umich.edu
Hoang, Y.      yhoang@umich.edu

## Problem, Impact, and Motivation
Organizations selling digital products or services often process thousands of customer-support tickets daily. Each incoming message needs to be classified and routed to the best-fitting queue, whether it be Billing, Technical Support, Returns, General Inquiry, or another specialized team. 

Our project aims to reduce ticket misrouting by training machine learning models to predict the correct queue based on its text content. We are motivated by the opportunity to improve mean time-to-resolution, reduce inefficient use of company resources, decrease the amount of unnecessary ticket reassignments, and provide scalabilitiy in customer support operations. 

# Data Source
Our pipeline uses a dataset of support tickets containing, but not limited to these vital features:

    - Ticket Subject: Text description of the ticket subject.

    - Ticket Body: Text description of the ticket issue.

    - Queue Label: The historically assigned queue for ticket resolution.

The dataset is stored in the repository under data/raw/, and can also be found at the following link: https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets/data. This data falls under the CC BY 4.0 license. 

# Project Setup
The study methodology was set up as follows: 
1. Data Loading + Preprocessing
    - Concatenate all CSV files in data/raw/
    - Map similar queues to broader catergories (e.g Technical Support + IT Support → Technical & IT Support)
    - Redact personally identifiable information (PII).
    - Clean and normalize text fields.
    - Deduplicate and split into train, validation, and test set while following    procudures to prevent data leakage.
2. Feature Engineering
    - TF-IDF Vectorization
    - SBERT Sentence Embeddings
3. Model Training
    - Logistic Regression (with TF-IDF)
    - Random Forest (with SBERT)
    - Support Vector Classifier (with SBERT)
    - eXtreme Gradient Boosting (with SBERT)
4. Evaluation
    - Generate classification reports for each model
    - Reports are stored under reports/

# Project Report + Poster
The project report can be found at: [Queuerious Detector Report](https://toobrightideas.medium.com/querious-detector-using-ai-to-assist-with-support-queue-ticket-assignment-98c4dbc08e21)

The project poster can be found at: [Queuerious Detector Poster](https://drive.google.com/file/d/1ocw1dFYbfPDnF-3UuoxvRmwP2QtKVx-n/view?usp=sharing)

# How to Run the Code
Clone this repo using:
```bash
git clone https://github.com/dallen014/queuerious_detector.git
cd queuerious_detector
```

Install dependencies (only needs to be done once):
```bash
pip install -r requirements.txt
```

Run the full pipeline for all models:
```bash
make pipeline-all
```

OPTIONAL: Run only the TF-IDF + Logistic Regression pipeline:
```bash
make pipeline-lg
```

OPTIONAL: Run only SBERT + all SBERT-based models:
```bash
make pipeline-sbert-all
```
# See the SVC Model in Action
We included a notebook: `3.01-see-SVC-in-action.ipynb` in the notebooks section. This notebook demonstrates how the trained Support Vector Classifier can:

- Predict the most likely queue for new, unseen support tickets
- Output probabilities for the top three most likely queues
- Generate a LIME interpretability plot showing which words pushed the model toward or away from its prediction

This is especially useful for understanding model behavior in a production-like setting, aiding human reviewers in making informed decisions when using AI-assisted ticket routing.


## Acknowledgements

This project was structured using the [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) template.

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>
