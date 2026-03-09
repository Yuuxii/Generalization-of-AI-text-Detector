import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import random
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

with open("1stage_prompts.json", "r", encoding="utf-8") as file:
    prompt_file = json.load(file)

class HumanWrittenDataset(Dataset):
    def __init__(self, dataset_type, prompt_type, prompt_subtype, tokenizer, split=0, n_examples=3):

        self.df = self.load_data(dataset_type, split=split)
        self.dataset_type = dataset_type
        self.prompt_type = prompt_type
        self.prompt_subtype = prompt_subtype
        self.n_examples = n_examples
        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.df)
    
    def __getitem__(self, idx):

        if self.prompt_type == 'in_context_learning':
            prompt = self.get_icl_prompts(idx, self.n_examples)
        else:
            prompt = self.get_prompts(idx, self.prompt_type, self.prompt_subtype)
        
        prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

        return prompt

    def load_data(self, dataset_type, split):
        if isinstance(split, int):
            if dataset_type == "abstracts":
                dataset_files = ["data-preprocessing/Data/abstracts/abstracts_processed_0.json"]
            elif dataset_type == "reviews":
                dataset_files = ["data-preprocessing/Data/reviews/reviews_processed_0.json"]
            elif dataset_type == "news":
                dataset_files = ["data-preprocessing/Data/news/news_processed_0.json", "data-preprocessing/Data/news/news_processed_1.json", "data-preprocessing/Data/news/news_processed_2.json"]
            elif dataset_type == "qa":
                dataset_files = ["data-preprocessing/Data/qa/qa_processed_0.json", "data-preprocessing/Data/qa/qa_processed_1.json"]
            else:
                print("Dataset type not found!")

            data = pd.read_json(dataset_files[split]) 

        return data

    def get_prompts(self, idx, prompt_type, prompt_subtype):

        row = self.df.iloc[idx]
        prompt = prompt_file[prompt_type][prompt_subtype][self.dataset_type]
        length = round(int(row.length_in_characters), -1) # round the length to the nearest 10

        df_for_sampling = self.df.copy().drop([idx])
        random_example = df_for_sampling.sample(n=1, ignore_index=True)
        random_example = random_example.iloc[0]

        if self.dataset_type == "abstracts":
            if len(row.categories) >= 2:
                # change the list into a string
                category = "{} and {}".format(", ".join(row.categories[:-1]), row.categories[-1])
            else:
                category = row.categories[0]

            prompt = prompt.format(title=row.title,
                                   length=length,
                                   category=category,
                                   example=random_example.abstract)

        if self.dataset_type == "news":

            prompt = prompt.format(highlights=row.highlights,
                                   length=length,
                                   example=random_example.article)
    
        if self.dataset_type == "reviews":
        
            prompt = prompt.format(length=length,
                                   item_name=row.title_meta,
                                   title=row.title_review,
                                   rating=row.rating,
                                   example=random_example.text)

            if prompt_subtype == "style_description":
                if row.rating == 5.0:
                    prompt += " Emphasize your satisfaction with the product."
                elif row.rating == 1.0:
                    prompt += " Emphasize your dissapointment with the product." 

        if self.dataset_type == "qa":
            prompt = prompt.format(length=length,
                                   question=row.question,
                                   example=random_example.long_answer)

        messages = [{"role" : "user", "content" : prompt}]
    
        return messages
    
    def get_icl_prompts(self, idx, n_examples):

        messages = []
        
        row = self.df.iloc[idx]
        few_shot_examples = self.df.sample(n=n_examples, ignore_index=True)

        content_dict = {"abstracts" : ["title", "abstract"],
         "news" : ["highlights", "article"],
         "reviews" : ["title_meta", "text"],
         "qa" : ["question", "long_answer"]}
        
        user_content = content_dict[self.dataset_type][0]
        assistant_content = content_dict[self.dataset_type][1]
       
        for i in range(n_examples):
            messages.extend([{"role" : "user", "content" : few_shot_examples.iloc[i][user_content]},
                            {"role" : "assistant", "content" : few_shot_examples.iloc[i][assistant_content]}])      
        
        messages.append({"role" : "user", "content" : row[user_content]})

        return messages
    