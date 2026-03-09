import pandas as pd
import numpy as np
import os
from lexical_diversity import lex_div as ld
import textstat
import spacy
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from textblob import TextBlob
nlp = spacy.load("en_core_web_sm")

# non-lemmatized lexical diversity measures

def mean_segmental_TTR(text):

    tokens = ld.tokenize(text)
    msttr = ld.msttr(tokens)

    return msttr

def moving_average_TTR(text):

    tokens = ld.tokenize(text)
    mattr = ld.mattr(tokens)

    return mattr

def maas_TTR(text):

    tokens = ld.tokenize(text)
    maas_ttr = ld.maas_ttr(tokens)

    return maas_ttr

def measure_of_tld(text):

    tokens = ld.tokenize(text)
    mtld = ld.mtld(tokens)

    return mtld

#lemmatized lexical diversity measures

def lemmatized_mean_segmental_TTR(text):
    
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    lemmatized_msttr = ld.msttr(lemmas)

    return lemmatized_msttr

def lemmatized_moving_average_TTR(text):

    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    lemmatized_mattr = ld.mattr(lemmas)

    return lemmatized_mattr

def lemmatized_maas_TTR(text):

    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    lemmatized_maas_ttr = ld.maas_ttr(lemmas)

    return lemmatized_maas_ttr

def lemmatized_measure_of_tld(text):

    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    lemmatized_mstld = ld.mtld(lemmas)

    return lemmatized_mstld

def num_unique_words(text):
    
    doc = nlp(text)
    unique_tokens = {token for token in doc}

    return len(unique_tokens)

def lexical_density(text):

    doc = nlp(text)

    content_word_tags = ["ADJ", "ADV", "NOUN", "PROPN", "VERB"]
    num_content_words = len([token for token in doc if token.pos_ in content_word_tags])
    num_words =  len([token for token in doc if not token.is_punct and not token.is_space])

    return num_content_words / num_words

def lemmatized_MATTR_content_words(text):

    doc = nlp(text)
    content_word_tags = ["ADJ", "ADV", "NOUN", "PROPN", "VERB"]

    tokens = [token.lemma_ for token in doc if token.pos_ in content_word_tags]
    mattr_ct = ld.mattr(tokens)

    return mattr_ct

# sentence statistics

def avg_sentence_len(text):

    doc = nlp(text)
    num_sents = len([sent for sent in doc.sents])
    num_words = len([token for token in doc if not token.is_punct and not token.is_space])

    avg_sent_len = num_words / num_sents

    return avg_sent_len

def num_long_sentences(text):

    num_words = 35
    
    doc = nlp(text)
    sent_lens = [len(sent) for sent in doc.sents]
    num_long_sents = len([sent_len for sent_len in sent_lens if sent_len >= num_words])

    return num_long_sents

def num_short_sentences(text):
    
    num_words = 10
    
    doc = nlp(text)
    sent_lens = [len(sent) for sent in doc.sents]
    num_short_sents = len([sent_len for sent_len in sent_lens if sent_len <= num_words])

    return num_short_sents

def standard_deviation_sent_len(text):
    
    doc = nlp(text)
    sent_lens = [len(sent) for sent in doc.sents]

    return np.std(sent_lens)

def get_metrics(data):

    texts = data["text"].to_list()

    results = {"text" : [],
               "MATTR" : [],
               "L_MATTR" : [],
               "LEX_DEN" : [],
               "UNIQUE_WORDS" : [],
               "FLESCH" : [],
               "GUNNING_FOG" : [],
               "AVG_SENT_LEN" : [],
               "LONG_SENTS" : [],
               "SHORT_SENTS" : [],
               "SENT_LEN_STD" : [],
               "len_in_characters" : []}

    for text in texts:
        results["text"].append(text)
        results["MATTR"].append(moving_average_TTR(text))
        results["L_MATTR"].append(lemmatized_moving_average_TTR(text))
        results["LEX_DEN"].append(lexical_density(text))
        results["UNIQUE_WORDS"].append(num_unique_words(text))
        results["FLESCH"].append(textstat.flesch_reading_ease(text))
        results["GUNNING_FOG"].append(textstat.gunning_fog(text))
        results["AVG_SENT_LEN"].append(avg_sentence_len(text))
        results["LONG_SENTS"].append(num_long_sentences(text))
        results["SHORT_SENTS"].append(num_short_sentences(text))
        results["SENT_LEN_STD"].append(standard_deviation_sent_len(text))
        results["len_in_characters"].append(len(text))
    
    results = pd.DataFrame.from_dict(results)
    results["label"] = data["label"].values
    results["prompt_style"] = data["prompt_style"].values
    results["model"] = data["model"].values
    results["item_id"] = data["item_id"].values
    results["source"] = data["source"].values

    return results

def get_new_metrics(data):

    texts = data["text"].to_list()

    results = {
               "polarity" : [],
               "subjectivity" : [],
             }

    for text in texts:
        blob = TextBlob(text)
        results["polarity"].append(blob.sentiment.polarity)
        results["subjectivity"].append(blob.sentiment.subjectivity)
  
    
    data["polarity"] = results["polarity"]
    data["subjectivity"] = results["subjectivity"]
    

    return data

def process_texts(df, n_proc=128):
    
    data = df
    chunk_size = len(data) // n_proc  
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    results = []
    
    with ProcessPoolExecutor(max_workers=n_proc) as executor:
        futures = [executor.submit(get_new_metrics, chunk) for chunk in chunks]
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())
            
    df = pd.concat(results, axis=0)

    return df

if __name__ == "__main__":

    dataset_path = os.path.join("full_dataset", "linguistic_analysis_multistage.csv")
    df = pd.read_csv(dataset_path, index_col=0)
    df.dropna(axis=0, ignore_index=True, inplace=True)

    final_df = process_texts(df)

    save_path = os.path.join("full_dataset", "linguistic_analysis_multistage.csv")
    final_df.to_csv(save_path, encoding="utf-8")

    dataset_path = os.path.join("full_dataset", "linguistic_analysis.csv")
    df = pd.read_csv(dataset_path, index_col=0)
    df.dropna(axis=0, ignore_index=True, inplace=True)

    final_df = process_texts(df)

    save_path = os.path.join("full_dataset", "linguistic_analysis.csv")
    final_df.to_csv(save_path, encoding="utf-8")
    