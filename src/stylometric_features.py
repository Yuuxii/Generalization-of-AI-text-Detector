import stylo_metrix as sm 
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


metrics_to_exclude = ["L_REF",
                        "L_HASHTAG",
                        "L_MENTION",
                        "L_RT",
                        "L_LINKS",
                        "L_PLURAL_NOUNS",
                        "L_SINGULAR_NOUNS",
                        "L_NOUN_PHRASES",
                        "L_PUNCT",
                        "L_PUNCT_DOT",
                        "L_PUNCT_COM",
                        "L_PUNCT_SEMC",
                        "L_PUNCT_COL",
                        "L_PUNCT_DASH",
                        "L_POSSESIVES",
                        "PS_CONTRADICTION",
                        "PS_AGREEMENT",
                        "PS_EXAMPLES",
                        "PS_CONSEQUENCE",
                        "PS_CAUSE",
                        "PS_LOCATION",
                        "PS_TIME",
                        "PS_CONDITION",
                        "PS_MANNER",
                        "ST_TYPE_TOKEN_RATIO_LEMMAS",
                        "ST_HERDAN_TTR",
                        "ST_MASS_TTR",
                        "ST_SENT_WRDSPERSENT",
                        "ST_SENT_DIFFERENCE",
                        "ST_SENT_D_VP",
                        "ST_SENT_D_NP",
                        "ST_SENT_D_PP",
                        "ST_SENT_D_ADJP",
                        "ST_SENT_D_ADVP"]

stylo = sm.StyloMetrix("en", metrics=["Lexical", "Statistics", "Pronouns", "General"],
                    exceptions=metrics_to_exclude,
                    debug=False)


def get_stylo_metrics(data):

    texts = data["text"].to_list()

    results = stylo.transform(texts)

    results["label"] = data["label"].values
    results["prompt_style"] = data["prompt_style"].values
    results["model"] = data["model"].values
    results["item_id"] = data["item_id"].values
    results["source"] = data["source"].values

    return results

if __name__ == "__main__":

    dataset_path = os.path.join("full_dataset", "dataset_multistage_EN.csv")
    df = pd.read_csv(dataset_path, index_col=0)
    df.dropna(axis=0, ignore_index=True, inplace=True)

    final_df = get_stylo_metrics(df)

    save_path = os.path.join("full_dataset", "stylometric_analysis_multistage.csv")
    final_df.to_csv(save_path, encoding="utf-8")

    dataset_path = os.path.join("full_dataset", "dataset_full_EN.csv")
    df = pd.read_csv(dataset_path, index_col=0)
    df.dropna(axis=0, ignore_index=True, inplace=True)

    final_df = get_stylo_metrics(df)

    save_path = os.path.join("full_dataset", "stylometric_analysis.csv")
    final_df.to_csv(save_path, encoding="utf-8")