import os
import json
import jsonlines
import h5py
import networkx as nx
import math
import numpy as np
import string
import pdb

import ast
from fuzzywuzzy import fuzz
import spacy

class ImageFeaturesDB(object):
    def __init__(self, img_ft_file, image_feat_size):
        self.image_feat_size = image_feat_size
        self.img_ft_file = img_ft_file
        self._feature_store = {}

    def get_image_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if key in self._feature_store:
            ft = self._feature_store[key]
        else:
            # pdb.set_trace()
            with h5py.File(self.img_ft_file, 'r') as f:
                ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                self._feature_store[key] = ft
        return ft

class ImaginationImageFeaturesDB(object):
    def __init__(self, img_ft_file, image_feat_size):
        self.image_feat_size = image_feat_size
        self.img_ft_file = img_ft_file
        self._feature_store = {}

    def get_image_feature(self, path_id_instr_idx):
        # key = '%s_%s' % (path_id, instr_idx) #instr_idx is 0, 1 or 2.
        key = path_id_instr_idx #This should be a string.
        if key in self._feature_store:
            ft = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                self._feature_store[key] = ft
        return ft


def load_instr_datasets(anno_dir, dataset, splits):
    data = []
    for split in splits:
        if "/" not in split:    # the official splits
            if dataset == 'r2r':
                with open(os.path.join(anno_dir, 'R2R_%s_enc.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'fgr2r':
                print('Reading FGR2R')
                with open(os.path.join(anno_dir, 'FGR2R_%s.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'r2r_last':
                with open(os.path.join(anno_dir, 'LastSent', 'R2R_%s_enc.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'r2r_back':
                with open(os.path.join(anno_dir, 'ReturnBack', 'R2R_%s_enc.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'r4r':
                with open(os.path.join(anno_dir, 'R4R_%s_enc.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'rxr':
                new_data = []
                with jsonlines.open(os.path.join(anno_dir, 'rxr_%s_guide_enc_xlmr.jsonl'%split)) as f:
                    for item in f:
                        new_data.append(item)
        else:   # augmented data
            print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
            with open(split) as f:
                new_data = json.load(f)

        # Join
        data += new_data
    return data

def construct_instrs(anno_dir, dataset, splits, tokenizer=None, max_instr_len=512, aug_flag = False):
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits)):
        if dataset == 'rxr':
            # rxr annotations are already split
            new_item = dict(item)
            if 'path_id' in item:
                new_item['instr_id'] = '%d_%d'%(item['path_id'], item['instruction_id'])
            else: # test
                new_item['path_id'] = new_item['instr_id'] = str(item['instruction_id'])
            # new_item['instr_encoding'] = item['instr_encoding'][:max_instr_len]
            new_item['instr_encoding'] = item['instr_encoding'] #leaving out max_instr_len cropping
            data.append(new_item)
        else:
            # Split multiple instructions into separate entries
            for j, instr in enumerate(item['instructions']):
                if j>2 and not aug_flag:
                    print('construct_instrs(): Skipping split: ', splits, 'pathid_instrid: ',  item['path_id'], '-', j)
                    continue #Just to get baseline with aug max performance. Uncomment this after baseline experiment.
                new_item = dict(item)
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j) #TODO: Can j be > 2? 
                new_item['instruction'] = instr
                # new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
                new_item['instr_encoding'] = item['instr_encodings'][j] #leaving out max_instr_len cropping
                del new_item['instructions']
                del new_item['instr_encodings']

                # ''' BERT tokenizer '''
                # instr_tokens = ['[CLS]'] + tokenizer.tokenize(instr)[:max_instr_len-2] + ['[SEP]']
                # new_item['instr_encoding'] = tokenizer.convert_tokens_to_ids(instr_tokens)
                          
                data.append(new_item)
    return data


# Function to filter out punctuation tokens and store indices of non-punctuation tokens only.
def filter_punctuation_with_indices(tokens):
    filtered_tokens = []
    indices = []
    for i, token in enumerate(tokens):
        if token not in string.punctuation:
            filtered_tokens.append(token)
            indices.append(i)
    return filtered_tokens, indices

# Function to find the segment with the maximum similarity score
def find_best_segment(instr_tokens, sub_instr_tokens, threshold=85):
    # Filter out punctuation from the original tokens
    filtered_tokens, indices = filter_punctuation_with_indices(instr_tokens)
    
    max_similarity = -1
    best_segment = (0, 0, 0)
    
    # Sliding window approach on filtered tokens with the length of window being the length of sub_instr_tokens
    for i in range(len(filtered_tokens) - len(sub_instr_tokens) + 1):
        instr_window = filtered_tokens[i:i + len(sub_instr_tokens)]
        similarity = fuzz.ratio(" ".join(instr_window), " ".join(sub_instr_tokens))
        
        if similarity > max_similarity:
            max_similarity = similarity
            # Map back to original indices
            start_idx = indices[i]
            end_idx = indices[i + len(sub_instr_tokens) - 1] + 1
            best_segment = (start_idx, end_idx, similarity)
    
    return best_segment

#This function is used to find the segmentation points of sub-instrs from original R2R instructions and write to a file. Needs to be run only once to create the dataset.
def construct_sub_instr_segmentations_score_maximize(r2r_anno_dir, dataset, fgr2r_anno_dir, fg_dataset, splits, tokenizer=None, aug_flag = False, \
output_dir = '/nfs/stak/users/perincha/sw-soundwave/repos/VLN-HAMT/datasets/R2R/annotations'):
    assert len(splits) == 1
    metadata_save = []
    r2r_dataset = load_instr_datasets(r2r_anno_dir, dataset, splits)
    for i, fg_item in enumerate(load_instr_datasets(fgr2r_anno_dir, fg_dataset, splits)):
        # Split multiple instructions into separate entries
        r2r_curr = r2r_dataset[i]
        assert r2r_curr['path_id'] == fg_item['path_id'] #Ensure same instructions are being read from R2R and FGR2R.
        
        for j, instr in enumerate(fg_item['instructions']): #iterate through 3 instructions.
            if j>2 and not aug_flag: #Ignore samples with >3 instructions just like what was done for imaginations. (only misses about 10 samples across splits)
                print('construct_sub_instr_segmentations_score_maximize(): Skipping split: ', splits, 'pathid_instrid: ',  fg_item['path_id'], '-', j)
                continue #Just to get baseline with aug max performance. Uncomment this after baseline experiment.
            fg_item_dict = dict(fg_item)
            instr_id = '%s_%d' % (fg_item_dict['path_id'], j)
            curr_r2r_instr_encoding = r2r_curr['instr_encodings'][j]
            curr_r2r_instr_tokens = tokenizer.convert_ids_to_tokens(curr_r2r_instr_encoding)
            # print('Instruction tokens: ', curr_r2r_instr_tokens, '\n')
            curr_fg_sub_instrs = fg_item_dict['new_instructions']
            curr_fg_sub_instrs_list = ast.literal_eval(curr_fg_sub_instrs)[j] #Pull out sub-instr only for the current instruction - not all 3.
            num_subs_instrs = len(curr_fg_sub_instrs_list) #num of sub-instructions present in the current instruction.

            segmentation_idxs = []
            sub_instr_token_list = []
            start_idx = 1 #starting at 1 because the first token will be CLS in R2R instr which should be ignored.
            for curr_sub_instr in curr_fg_sub_instrs_list: #iterate through sub-instructions within an instruction.
                curr_sub_instr = ' '.join(curr_sub_instr) #convert list of tokens to a string.
                # print('SI - ', curr_sub_instr)
                curr_sub_instr_tokens = tokenizer.tokenize(curr_sub_instr)
                match_tuple = find_best_segment(curr_r2r_instr_tokens, curr_sub_instr_tokens) #(start_idx, begin_idx, similarity_score)
                start_idx = match_tuple[0]
                end_idx = match_tuple[1]-1 #inclusive index
                if match_tuple:
                    segmentation_idxs.append((start_idx, end_idx))
                else:
                    print(f"{'-'*10}-Match not found in instruction!-{'-'*10}")
                    print(f'Instr_id: {instr_id}')
                    print('Instruction tokens: ', curr_r2r_instr_tokens, '\n')
                    print('Sub instruction tokens: ', curr_sub_instr_tokens)
                    segmentation_idxs.append((None, None))
                sub_instr_token_list.append(curr_sub_instr_tokens)

            instr_save_dict = {'path_id': fg_item_dict['path_id'], 'instruction_id': instr_id, 'trajectory_length':len(curr_fg_sub_instrs_list), \
            'instruction': instr, 'sub-instructions': curr_fg_sub_instrs_list, \
            'instruction_tokens': curr_r2r_instr_tokens, 'sub-instructions_tokens': sub_instr_token_list, 'instr_segmentation_indices': segmentation_idxs}
            metadata_save.append(instr_save_dict)
            
    # metadata_save_file = os.path.join(output_dir, f'fgr2r_subinstrs_segmentation_data_{splits[0]}.json')
    # json.dump(metadata_save, open(metadata_save_file, 'w'), indent='\t')
    return


# Load the English NLP model
nlp = spacy.load('en_core_web_sm')

def extract_nouns_with_indices(sentence):
    # Process the sentence with spaCy
    doc = nlp(sentence)
    
    # Extract noun phrases
    nouns_info = []
    for chunk in doc.noun_chunks:
        # Extract nouns within the noun phrase
        for token in chunk:
            if token.pos_ == "NOUN":
                nouns_info.append((token.text, token.i))
    
    return nouns_info

def merge_subword_tokens(tokens):
    merged_tokens = []
    token_mapping = []
    i = 0
    while i < len(tokens):
        if tokens[i].startswith('##'):
            if not merged_tokens:
                # Handle case where the first token starts with ##
                # Remove the ## and retain the text that comes right after and add this as a valid idx since we aren't combining it with another token but retaining as it is.
                merged_tokens.append(tokens[i][2:])
                token_mapping.append(i)
                print('Found sub-instr starting with ##')
            else:
                # Append to the previous token
                merged_tokens[-1] += tokens[i][2:]
        else:
            # Add new token
            merged_tokens.append(tokens[i])
            token_mapping.append(i)
        i += 1
    return merged_tokens, token_mapping

def filter_non_room_noun_tuples(input_list_of_tuples, filter_list):
    # Function to check if any filter word is in the string
    def contains_filter(word):
        return any(f in word for f in filter_list)

    # Filter the list of tuples
    result = [item for item in input_list_of_tuples if item[0]=='room' or not contains_filter(item[0])]
    return result

def remove_duplicates_from_sublists(input_list):
    result = []
    for sublist in input_list:
        # Use a set to remove duplicates within the sublist and then convert back to list
        seen = set()
        unique_sublist = []
        for item in sublist:
            item_tuple = tuple(item)  # Convert the inner list to a tuple to make it hashable
            if item_tuple not in seen:
                unique_sublist.append(item)
                seen.add(item_tuple)
        result.append(unique_sublist)
    return result

def extract_noun_phrases_after_merging_split_tokens(curr_sub_instr_toks, excluded_noun_set):
    merged_tokens, merged_mapping = merge_subword_tokens(curr_sub_instr_toks) #to account of #s in the tokens that are separated as separated words by Spacy.
    text = ' '.join(merged_tokens)
    doc = nlp(text)   
    # noun_phrases = []
    # noun_phrases_excluded_list_flag = []
    noun_phrases_2 = []

    offset_cls_indices = 0
    if str(doc[:3]) == '[CLS]': #corner case detected when the sub-instr segmentation fails and the entire instruction is a sub-instruction. CLS gets into the sub-instr and the indices from Spacy are off.
        offset_cls_indices = -3+1 #shift back by 3 to account for '[', 'CLS', ']' and move ahead by 1 to go the token right after CLS.
        # print('CLS detected')
    for noun_phrase_chunks in doc.noun_chunks:
        # Find the start and end indices in the original token list
        np_start_idx = noun_phrase_chunks.start
        np_end_idx = noun_phrase_chunks.end
        # Adjust end index to not exceed the length of the document -> This is to correct an issue detected with Spacy's filtering itself.
        #Spacy does some weird things. Eg: if text is 'stop in front of the chair thats next to the window', doc actually becomes [(0, stop), (1, in), (2, front), (3, of), (4, the), (5, chair), (6, that), (7, s), (8, next), (9, to), (10, the), (11, window)] as it splits thats to that, s
        #Similarly, cannot gets split into can and not.
        if np_end_idx+offset_cls_indices > len(merged_tokens):
            np_end_idx = len(merged_tokens) - offset_cls_indices #Not a perfect solution.
            if np_start_idx >= np_end_idx: #This is not a clean solution but it shouldn't affect downstream too much.
                np_start_idx = np_end_idx-1 #-1 because end_idx is 1 beyond the last accessible index. usage of list[start_idx:end_idx] would make sense.
            # assert np_start_idx <= np_end_idx, 'Index was clipped incorrectly.'
        start_idx = merged_mapping[np_start_idx + offset_cls_indices] 
        end_idx = merged_mapping[np_end_idx - 1 + offset_cls_indices]
        # noun_phrases.append((noun_phrase_chunks.text, start_idx, end_idx))
        # noun_phrases_secondary_filtered = []
        # noun_phrases_excluded_list_flag_secondary_filtered = []
        if len(noun_phrase_chunks.text)!=0:
            noun_phrase_words = set(noun_phrase_chunks.text.split())

            #Filtering stage 1.
            noun_phrase_to_be_excluded = not noun_phrase_words.isdisjoint(excluded_noun_set)
            #Filtering stage 2 for sub-segments of the above filtered noun-phrase.
            if noun_phrase_to_be_excluded: #Some excluded noun-phrases can contain important information eg: "front door" -> "door" should still be extracted. 
                common_words = noun_phrase_words.intersection(excluded_noun_set)
                # print('Common words with that of excluded filtering list: ', common_words)
                noun_phrases_2.append((noun_phrase_chunks.text, start_idx, end_idx, True)) #bool flag denotes whether the noun-phrase should be excluded.
                #Special filtering stage for phrases with room.
                #Treat room as special to preserve "room" along with "dining" in "dining room" for example.
                noun_phrase_copy = list(map(str.lower, noun_phrase_chunks.text.split())).copy()
                if 'room' in noun_phrase_copy and len(noun_phrase_copy)>1:
                    # noun_phrase_copy.remove('room')
                    nouns_in_room_phrase_info = extract_nouns_with_indices(' '.join(noun_phrase_copy)) #should include room.
                    # print('Before removing non-room nouns: ', nouns_in_room_phrase_info)

                    #If Spacy does not detect 'room' as a noun, manually add room and its index. This was observed in "past couch and dining room table".
                    if 'room' not in [noun_info[0] for noun_info in nouns_in_room_phrase_info]:
                        room_index = [curr_noun_idx for curr_noun_idx, curr_noun in enumerate(noun_phrase_copy) if curr_noun=='room'][0]
                        nouns_in_room_phrase_info.append(('room', room_index))
                        print('Room was not detected as a noun despite being present.')
                        del room_index

                    nouns_in_room_phrase_info_filtered = filter_non_room_noun_tuples(nouns_in_room_phrase_info, list(excluded_noun_set)) #room if noun is in filter list but keep 'room' because if 'dining room', we want to keep room as well not just dining.

                    print('After removing non-room nouns: ', nouns_in_room_phrase_info_filtered)
                    all_nouns = [noun_info[0] for noun_info in nouns_in_room_phrase_info_filtered]
                    # find index of room.
                    if 'room' in all_nouns and len(nouns_in_room_phrase_info_filtered) > 1: #Mostly redundant, room will most likely be treated as a noun. 
                        room_idx_list = [noun_info[1] for noun_info in nouns_in_room_phrase_info_filtered if noun_info[0]=='room']
                        room_idx = room_idx_list[0]
                    if len(nouns_in_room_phrase_info_filtered) > 1: #Are there more nouns apart from 'room' like 'dining'.
                        for noun_room_info in nouns_in_room_phrase_info_filtered:
                            if np.abs(noun_room_info[1] - room_idx) < 2: #Select only nouns right next to room. This might be an overkill. Obs: This radius fails in cases like "past couch and dining room table" where room is separated by more than an index from couch. Ignoring this issue here because not masking something even if it has imaginations is better than masking some vital information that cannot be conveyed by imaginations.
                                noun_phrases_2.append((noun_room_info[0], start_idx + noun_room_info[1], start_idx + noun_room_info[1], False))
                                # print(f"{'*'*40} Preserving 'room' adjacent nouns: {noun_room_info[0]}{'*'*40}")
                            else:
                                pass
                                # print(f"{'*'*40} Deleting 'room' related nouns because of adjacency filter: {noun_room_info[0]}{'*'*40}")

                nouns_in_noun_phrase_info = extract_nouns_with_indices(noun_phrase_chunks.text)
                for noun_info in nouns_in_noun_phrase_info:
                    noun = noun_info[0]
                    noun_idx = noun_info[1]
                    if noun in list(excluded_noun_set):
                        continue
                    # print(f'Selecting "{noun}" from noun phrase "{noun_phrase_chunks.text}"')
                    # noun_phrases_secondary_filtered.append((noun, start_idx + noun_idx, start_idx + noun_idx))
                    # noun_phrases_excluded_list_flag_secondary_filtered.append(False)
                    noun_phrases_2.append((noun, start_idx + noun_idx, start_idx + noun_idx, False)) #bool flag denotes whether the noun-phrase should be excluded.
            else:
                noun_phrases_2.append((noun_phrase_chunks.text, start_idx, end_idx, False)) #bool flag denotes whether the noun-phrase should be excluded.

            # noun_phrases_excluded_list_flag.append(noun_phrase_to_be_excluded)
            # assert len(noun_phrases_excluded_list_flag) == len(noun_phrases)  
            # assert len(noun_phrases_excluded_list_flag_secondary_filtered) == len(noun_phrases_secondary_filtered)
            # noun_phrases_excluded_list_flag = noun_phrases_excluded_list_flag + noun_phrases_excluded_list_flag_secondary_filtered
            # noun_phrases = noun_phrases + noun_phrases_secondary_filtered
    # return noun_phrases, noun_phrases_excluded_list_flag
    return noun_phrases_2

def annotate_noun_phrases_from_subinstrs(sub_instr_seg_data, split, imagine_v2_generated_flag_dict, output_dir = None):
    # assert len(splits) == 1
    excluded_noun_list = ['straight', 'stop', 'wait', 'left', 'right', 'turn', 'front', 'degree', 'side', 'veer', 'way', 'direction', 'intersection', 'exit', 'center', 'step', 'top', 'corner', 'one', 'room']
    excluded_noun_list += ['middle', 'bottom', 'you', 'end', 'it', 'degrees'] #These noun-phrases can be ignored so that they aren't masked.
    excluded_noun_list += ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth']
    excluded_noun_list += ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    excluded_noun_list += ['clockwise', 'counter clockwise']
    excluded_noun_set = set(excluded_noun_list)

    metadata_save = []

    if split == 'val_train_seen':
        generated_flag_split = 'train'
    else:
        generated_flag_split = split
    imagination_v2_generated_flag = imagine_v2_generated_flag_dict[generated_flag_split]
    sanity_check_counter = 0
    instr_jump = 0
    sub_instr_seg_data = sub_instr_seg_data[instr_jump:]
    for i, sub_instr_seg_item in enumerate(sub_instr_seg_data): #within instruction
        sub_instr_seg_item_dict = dict(sub_instr_seg_item)
        curr_sub_instrs_toks = sub_instr_seg_item_dict["sub-instructions_tokens"]
        curr_sub_instrs_indices = sub_instr_seg_item_dict["instr_segmentation_indices"]
        curr_instr_id = sub_instr_seg_item_dict['instruction_id']
        imagine_generated_flags = imagination_v2_generated_flag[curr_instr_id]
        cur_instr_token = sub_instr_seg_item_dict['instruction_tokens']
        sanity_check_counter+=1
        print(f'\nViewing example {sanity_check_counter}')
        print('Current Instruction token: ', ' '.join(cur_instr_token), '\n')
        
        noun_phrases_in_subinstr_indices = []
        for sub_instr_idx, curr_sub_instr_toks in enumerate(curr_sub_instrs_toks): #within sub-instrs            
            curr_sub_instr_indices = curr_sub_instrs_indices[sub_instr_idx]
            retrieved_sub_instr_toks_from_instr = cur_instr_token[curr_sub_instr_indices[0]:curr_sub_instr_indices[1]+1]
            # sub_instr_str = ' '.join(curr_sub_instr_toks) #operating directly on cleaner sub-instr string. This has no punctuations (most likely).
            # Try with retrieved instr tokens directly.
            sub_instr_str = ' '.join(retrieved_sub_instr_toks_from_instr) #operating on retrieved sub-instr tokens from instr. This has punctuations.
            curr_sub_instr_toks = retrieved_sub_instr_toks_from_instr
            print('Sub-instruction: ', sub_instr_str)

            #Method 2
            noun_phrases = extract_noun_phrases_after_merging_split_tokens(curr_sub_instr_toks, excluded_noun_set)
            noun_phrases_excluded_list_flag = [item[3] for item in noun_phrases]
            print('\nNoun-phrases method 2: ', noun_phrases)

            # Filter invalid noun phrases and extract indices of valid noun phrases.
            noun_phrase_indices = []
            for noun_phrase_idx, noun_phrase in enumerate(noun_phrases):
                if noun_phrases_excluded_list_flag[noun_phrase_idx]:
                    # print(f"{'-'*10} Excluding noun phrase \"{noun_phrase[0]}\" after filtering.{'-'*10}")
                    continue
                noun_phrase_start_idx = noun_phrase[1]
                noun_phrase_end_idx = noun_phrase[2]
                noun_phrase_indices.append([noun_phrase_start_idx + curr_sub_instr_indices[0], noun_phrase_end_idx + curr_sub_instr_indices[0]]) #map index reference from within sub-instruction to instruction.
                extracted_noun_phrase_indices_from_instr = cur_instr_token[(noun_phrase_start_idx+curr_sub_instr_indices[0]):(noun_phrase_end_idx+1+curr_sub_instr_indices[0])]
                
                # print('Extracted inst tokens of noun-phrases: ', ' '.join(extracted_noun_phrase_indices_from_instr))
            noun_phrases_in_subinstr_indices.append(noun_phrase_indices)                     
            print('-'*120)
            print('\n')
        
        # Validation    
        noun_phrases_in_subinstr_indices = remove_duplicates_from_sublists(noun_phrases_in_subinstr_indices)
        print('\nPerforming sanity check by retrieving noun phrases...')
        noun_phrases_save = []
        num_sub_instrs_saved = len(noun_phrases_in_subinstr_indices)
        cur_instr_token_copy = cur_instr_token.copy()
        print(f'There were {num_sub_instrs_saved} sub-instructions:')
        for sub_instr_saved_idx, sub_instr_saved in enumerate(noun_phrases_in_subinstr_indices):
            all_np_in_sub_instr = []
            num_noun_phrase_in_subinstr_saved = len(sub_instr_saved)
            print(f'\nSub_instr {sub_instr_saved_idx} has {num_noun_phrase_in_subinstr_saved} noun-phrases that will be saved.')
            if num_noun_phrase_in_subinstr_saved==0:
                all_np_in_sub_instr.append([])            
            for noun_phrase_saved in sub_instr_saved:
                validated_noun_phrase_from_instr = cur_instr_token[noun_phrase_saved[0]:noun_phrase_saved[1]+1]
                print('Validated noun_phrase: ', ' '.join(validated_noun_phrase_from_instr))
                # all_np_in_sub_instr.append(validated_noun_phrase_from_instr)
                all_np_in_sub_instr.append(' '.join(validated_noun_phrase_from_instr))
                cur_instr_token_copy[noun_phrase_saved[0]:noun_phrase_saved[1]+1] = ['<MASK>']*(noun_phrase_saved[1]+1 - noun_phrase_saved[0])
            noun_phrases_save.append(all_np_in_sub_instr)
        print(f'\nEffectively: \ninstr toks {cur_instr_token} will end up like \ninstr toks {cur_instr_token_copy}.')
        print('-'*200)

        sub_instr_seg_item_dict['noun_phrase_indices'] = noun_phrases_in_subinstr_indices
        sub_instr_seg_item_dict['noun_phrases'] = noun_phrases_save

        metadata_save.append(sub_instr_seg_item_dict)
            
    # metadata_save_file = os.path.join(output_dir, f'fgr2r_nounphrase_segmentation_data_{split}.json')
    # json.dump(metadata_save, open(metadata_save_file, 'w'), indent='\t')
    return


def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs

 
def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading),math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)

def new_simulator(connectivity_dir, scan_data_dir=None):
    import MatterSim

    # Simulator image parameters
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60

    sim = MatterSim.Simulator()
    if scan_data_dir:
        sim.setDatasetPath(scan_data_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.initialize()

    return sim

def get_point_angle_feature(sim, angle_feat_size, baseViewId=0, minus_elevation=False):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    if minus_elevation:
        base_elevation = (baseViewId // 12 - 1) * math.radians(30)
    else:
        base_elevation = 0
        
    for ix in range(36):
        if ix == 0:
            sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        assert state.viewIndex == ix

        heading = state.heading - base_heading
        elevation = state.elevation - base_elevation

        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)
    return feature

def get_all_point_angle_feature(sim, angle_feat_size, minus_elevation=False):
    return [get_point_angle_feature(
        sim, angle_feat_size, baseViewId, minus_elevation=minus_elevation
        ) for baseViewId in range(36)]

