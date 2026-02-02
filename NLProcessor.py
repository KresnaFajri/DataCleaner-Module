#This module only works under the LLM_ENV environment
#Activate LLM_Env environment first in this device to use this python NLProcessor module
#or install these modules on your system

#Happy Fine-Tuning!

import re
import string
import numpy as np
import itertools
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
import torch

class NLProcessor:
    def __init__(self):
        pass
    def ExtractLexicalFeatures(self,token):
        return {
            "is_capitalized":token[0].isupper(),
            "is_all_caps":token.isupper(),
            "is_all_lower":token.islower(),
            "has_digit":any(ch.isdigit() for ch in token),
            "has_symbol":any(not ch.isalnum() for ch in token),
            "prefix2":token[:2].lower(),
            "prefix3":token[:3].lower(),
            "suffix2":token[-2:].lower(),
            "suffix3":token[-3:].lower()}

    class NERProcessor:
        def __init__(self,model_name,label2id,lexical_extractor =None ):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.label2id = label2id
            self.lexical_extractor = lexical_extractor #optional values

        def tokenize_and_align(self,tokens, labels = None):
            # 1) Tokenize while also keep word ids mapping
            encoding = self.tokenizer(
                tokens,
                is_split_into_words = True,
                truncation = True,
                padding = "max_length",
                max_length = 128,
                return_attention_mask =True,
                padding_side = "right"
            )

            word_ids = encoding.word_ids()

            aligned_labels = []
            aligned_lexical = []

            prev_word_id = None

            # 2)Align Labels
            for word_id in word_ids:
                if word_id is None:
                    # CLS/SEP/PAD Token
                    aligned_labels.append(self.label2id["O"])
                    aligned_lexical.append([0] * self.lexical_extractor.dim)

                elif word_id != prev_word_id:
                    aligned_labels.append(self.label2id[labels[word_id]])
                    aligned_lexical.append(self.lexical_extractor(tokens[word_id]))
                else:
                    #sub word
                    aligned_labels.append(self.label2id["O"])
                    aligned_lexical.append([0] * self.lexical_extractor.dim)
                
                prev_word_id = word_id

            encoding["attention_mask"][0] = 1
            aligned_labels[0] = aligned_labels[0] or 0

            encoding["labels"] = aligned_labels
            encoding["lexical_feats"] = aligned_lexical

            return encoding

            # 4)Return batch-ready data dictionary
            #    output = {
            #        "input_ids":torch.tensor(encoding["input_ids"]),
            #        "attention_mask":torch.tensor(encoding["attention_mask"])
            #    }
            #    if labels is not None:
            #        output["labels"] = torch.tensor(aligned_labels)
                
            #    if self.lexical_extractor:
            #        output["lexical_feats"] = torch.tensor(aligned_lexical)
                
            #    return output
        
    class LexicalExtractor:
        def __init__(self,enc_prefix2, enc_prefix3,enc_suffix2,enc_suffix3):
            self.enc_prefix2 = enc_prefix2
            self.enc_prefix3 = enc_prefix3
            self.enc_suffix2 = enc_suffix2
            self.enc_suffix3 = enc_suffix3
            self.dim = 9

        def __call__(self,token):
            feats = {
                "is_capitalized":int(token[0].isupper()),
                "is_all_caps":int(token.isupper()),
                "is_all_lower":int(token.islower()),
                "has_digit":int(any(ch.isdigit() for ch in token)),
                "has_symbol":int(any(not ch.isalnum() for ch in token)),
                "prefix2":token[:2].lower(),
                "prefix3":token[:3].lower(),
                "suffix2":token[-2:].lower(),
                "suffix3":token[-3:].lower()
            }
            #Convert prefix/suffix to int id
            feats["prefix2"] = self.enc_prefix2.get(feats["prefix2"],0)
            feats["prefix3"] = self.enc_prefix3.get(feats["prefix3"],0)
            feats["suffix2"] = self.enc_suffix2.get(feats["suffix2"],0)
            feats["suffix3"] = self.enc_suffix3.get(feats["suffix3"],0)
            
            return list(feats.values())