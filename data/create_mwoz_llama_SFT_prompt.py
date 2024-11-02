# adapted from # adapated from https://github.com/Yushi-Hu/IC-DST/blob/main/run_codex_experiment.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
import copy
from collections import defaultdict
from tqdm import tqdm
from config import CONFIG

from utils.helper import SpeedLimitTimer, PreviousStateRecorder
from utils.typo_fix import typo_fix
from utils.sql import sql_pred_parse, sv_dict_to_string

from prompts.prompting_mwoz_correctionlm import conversion, pure_table_prompt, get_llama_SFT_prompt
from retriever.code.embed_based_retriever import EmbeddingRetriever
from eval.evaluate_metrics import evaluate

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_fn', type=str, help="training data file (few-shot or full shot)", required=True)
parser.add_argument('--retriever_dir', type=str, required=True, help="sentence transformer saved path")  
parser.add_argument('--output_fn', type=str, default="running_log.json", help="file to save running log and configs")
parser.add_argument('--mwz_ver', type=str, default="2.1", choices=['2.1', '2.4'], help="version of MultiWOZ")  
parser.add_argument('--test_fn', type=str, default='', help="file to evaluate on, empty means use the test set")
args = parser.parse_args()

NUM_EXAMPLE=12
 
# read the selection pool
with open(args.train_fn) as f:
    train_set = json.load(f)

# read the ontology and the test set
if args.mwz_ver == '2.1':
    ontology_path = CONFIG["ontology_21"]
    if args.test_fn == "":
        test_set_path = "./data/mw21_100p_test.json"
else:
    ontology_path = CONFIG["ontology_24"]
    if args.test_fn == "":
        test_set_path = "./data/mw24_100p_test.json"

# evaluate on some other file
if args.test_fn:
    test_set_path = args.test_fn

print("Producing SFT prompt: ", test_set_path)

with open(ontology_path) as f:
    ontology = json.load(f)
with open(test_set_path) as f:
    test_set = json.load(f)

# load the retriever
retriever = EmbeddingRetriever(datasets=[train_set], 
                               model_path=args.retriever_dir,
                               search_index_filename=os.path.join(args.retriever_dir, "train_index.npy"), 
                               sampling_method="pre_assigned")

def run(test_set, turn=-1, use_gold=False):
    # turn and use_gold are for analysis purpose
    # turn = -1 means evalute all dialogues
    # turn = 0 means evaluate single-turn dialogues
    # turn = 1 means evalute two-turn dialogues... etc.
    # when use_gold = True, the context are gold context (for analysis purpose)

    timer = SpeedLimitTimer(second_per_step=3.1)  # openai limitation 20 queries/min

    result_dict = defaultdict(list)  # use to record the accuracy

    selected_set = test_set
    # if needed, only evaluate on particular turns (analysis purpose)
    if turn >= 0:
        if not use_gold:
            raise ValueError("can only evaluate particular turn when using gold context")
        selected_set = [d for d in test_set if len(d['dialog']['usr']) == turn + 1]
    

    # start experiment
    all_result = []
    n_total = 0
    n_correct = 0
    n_correct_tlb = 0
    total_acc = 0
    total_dst_f1 = 0
    total_tlb_f1 = 0

    out_f = args.output_fn
    out = open(out_f, "w")

    for data_item in tqdm(selected_set):
        n_total += 1

        prompt_text, completion = get_llama_SFT_prompt(
            data_item, examples=retriever.item_to_nearest_examples(data_item, k=NUM_EXAMPLE))
       
        prompt_text_no_tables = prompt_text.replace(conversion(pure_table_prompt), "")

        line = {"SFT_prompt":prompt_text, "SFT_prompt_no_table":prompt_text_no_tables, "SFT_completion": completion}
        print("prompt_text: ", prompt_text)
        print("completion: ", completion)
        out.write(json.dumps(line))
        out.write("\n")

        # record the prompt
        data_item['prompt'] = prompt_text
  
        all_result.append(data_item)

    return all_result


if __name__ == "__main__":

    all_results = run(test_set)

    with open(args.output_fn, 'w') as f:
        json.dump(all_results, f, indent=4)
