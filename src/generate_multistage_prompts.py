import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

with open("multi_stage_prompts.json", "r", encoding="utf-8") as file:
    prompt_file = json.load(file)

class HWDatasetForMultiStagePrompting(Dataset):
    def __init__(self, dataset_type, prompt_type, prompt_subtype, tokenizer, split=0, n_examples=3, generated_text=None, generated_feedback=None, iterative_indexs=None):

        self.df = self.load_data(dataset_type, split=split)
        self.dataset_type = dataset_type
        self.prompt_type = prompt_type
        self.prompt_subtype = prompt_subtype
        self.n_examples = n_examples
        self.tokenizer = tokenizer
        self.generated_text = generated_text
        self.generated_feedback = generated_feedback
        self.content_dict = {"abstracts" : "abstract",
         "news" : "article",
         "reviews" : "text",
         "qa" : "long_answer"}
        
    def __len__(self):

        return len(self.df)
    
    def __getitem__(self, idx):
        
        if self.prompt_type == "self-refine":

            if self.prompt_subtype == "init_prompt":
                prompt = self.get_init_prompt(idx)

            elif self.prompt_subtype == "feedback_prompt":
                prompt = self.get_feedback_prompt(idx)

            elif self.prompt_subtype == "iterate_prompt":
                prompt = self.get_iterate_prompt(idx)

            elif self.prompt_subtype == "evaluation_prompt":
                prompt = self.get_evaluation_prompt(idx)
        
        prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

        return idx, prompt, self.df.iloc[idx][self.content_dict[self.dataset_type]]
    
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

        else:
            if dataset_type == "abstracts":
                dataset_file = "data-preprocessing/Data/abstracts/abstracts_processed_additional_samples.json"
            elif dataset_type == "reviews":
                dataset_file = "data-preprocessing/Data/reviews/reviews_processed_additional_samples.json"
            elif dataset_type == "news":
                dataset_file = "data-preprocessing/Data/news/news_processed_additional_samples.json"
            elif dataset_type == "qa":
                dataset_file = "data-preprocessing/Data/qa/qa_processed_additional_samples.json"
            else:
                print("Dataset type not found!")
                
            data = pd.read_json(dataset_file) 
        return data
                

    def get_init_prompt(self, idx):

        row = self.df.iloc[idx]
        prompt = prompt_file["self-refine"]["init_prompt"][self.dataset_type]
        length = round(int(row.length_in_characters), -1) # round the length to the nearest 10

        if self.dataset_type == "abstracts":

            if len(row.categories) >= 2:
                # change the list into a string
                category = "{} and {}".format(", ".join(row.categories[:-1]), row.categories[-1])
            else:
                category = row.categories[0]

            prompt = prompt.format(title=row.title,
                                   length=length,
                                   category=category)
        if self.dataset_type == "news":

            prompt = prompt.format(highlights=row.highlights,
                                   length=length)
    
        if self.dataset_type == "reviews":
        
            prompt = prompt.format(length=length,
                                   item_name=row.title_meta,
                                   title=row.title_review,
                                   rating=row.rating)

        if self.dataset_type == "qa":
            prompt = prompt.format(length=length,
                                   question=row.question)

        messages = [{"role" : "user", "content" : prompt}]
    
        return messages

    def get_feedback_prompt(self, idx):

        prompt = prompt_file["self-refine"]["feedback_prompt"][self.dataset_type]

        prompt =  prompt.format(text=self.generated_text[idx])
        
        messages = [{"role" : "user", "content" : prompt}]
    
        return messages
    
    def get_iterate_prompt(self, idx):

        prompt = prompt_file["self-refine"]["iterate_prompt"]
        prompt = prompt.format(text=self.generated_text[idx],
                               feedback=self.generated_feedback[idx])
        
        messages = [{"role" : "user", "content" : prompt}]
    
        return messages
    
    def get_evaluation_prompt(self, idx):
        
        prompt = prompt_file["self-refine"]["evaluation_prompt"]
        
        prompt = prompt.format(text_a=self.generated_text[idx],
                               text_b=self.df.iloc[idx][self.content_dict[self.dataset_type]])
        
        messages = [{"role" : "user", "content" : prompt}]
    
        return messages
    

def get_single_feedback_prompt(generated_text, dataset_type):

    prompt = prompt_file["self-refine"]["feedback_prompt"][dataset_type]

    prompt =  prompt.format(text=generated_text)
    
    messages = [{"role" : "user", "content" : prompt}]

    return messages


def get_single_iterate_prompt(generated_text, generated_feedback):

    prompt = prompt_file["self-refine"]["iterate_prompt"]
    prompt = prompt.format(text=generated_text,
                            feedback=generated_feedback)
    
    messages = [{"role" : "user", "content" : prompt}]

    return messages

def get_single_evaluation_prompt(generated_text, human_text):
    
    prompt = prompt_file["self-refine"]["evaluation_prompt"]
    
    prompt = prompt.format(text_a=generated_text,
                            text_b=human_text)
    
    messages = [{"role" : "user", "content" : prompt}]

    return messages

def get_single_prompt(tokenizer,
                    dataset_type, 
                    prompt_subtype, 
                    generation=None,
                    feedback=None,
                    human_text=None):
    
    if prompt_subtype == "feedback_prompt":
        prompt = get_single_feedback_prompt(generation, dataset_type)

    elif prompt_subtype == "iterate_prompt":
        prompt = get_single_iterate_prompt(generation, feedback)

    elif prompt_subtype == "evaluation_prompt":
        prompt = get_single_evaluation_prompt(generation, human_text)

    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    
    return prompt