import os
import json
import argparse
import copy
from collections import defaultdict
from tqdm import tqdm
from utils.helper import SpeedLimitTimer, PreviousStateRecorder
from config import CONFIG

from llama_quantization_completion_class import LM

from utils.sql_sgd import sql_pred_parse, sv_dict_to_string
from prompting_sgd_correctionlm import get_correctionlm_prompt

from retriever.code.embed_based_retriever import EmbeddingRetriever
from evaluate_metrics import evaluate

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_fn', type=str, help="training data file (few-shot or full shot)", required=True)  # e.g. "./data/mw21_10p_train_v3.json"
parser.add_argument('--retriever_dir', type=str, required=True, help="sentence transformer saved path")  # "./retriever/expts/mw21_10p_v3_0304_400_20"
parser.add_argument('--output_dir', type=str, default="./expts/debug", help="directory to save running log and configs")
parser.add_argument('--test_fn', type=str, default='', help="file to evaluate on, empty means use the test set")
parser.add_argument('--model', type=str, default='', help="model")
args = parser.parse_args()

# create the output folder
os.makedirs(args.output_dir, exist_ok=True)

with open(os.path.join(args.output_dir, "exp_config.json"), 'w') as f:
    json.dump(vars(args), f, indent=4)

NUM_EXAMPLE=4

# read the selection pool
with open(args.train_fn) as f:
    train_set = json.load(f)

# evaluate on some other file
if args.test_fn:
    test_set_path = args.test_fn

print("Running ICL on: ", test_set_path)

#with open(ontology_path) as f:
#    ontology = json.load(f)
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
    
    prediction_recorder = PreviousStateRecorder()  # state recorder

    # start experiment
    all_result = []
    n_total = 0
    n_correct = 0
    n_correct_tlb = 0
    total_acc = 0
    total_dst_f1 = 0
    total_tlb_f1 = 0
    
    LM_instance = LM(args.model)

    for data_item in tqdm(selected_set):
        n_total += 1

        completion = ""
        if use_gold:
            prompt_text, completion = get_correctionlm_prompt(
                data_item, examples=retriever.item_to_nearest_examples(data_item, k=NUM_EXAMPLE))
        else:
            predicted_context = prediction_recorder.state_retrieval(data_item)
            modified_item = copy.deepcopy(data_item)
            modified_item['last_slot_values'] = predicted_context
            examples = retriever.item_to_nearest_examples(
                modified_item, k=NUM_EXAMPLE)
            prompt_text, completion = get_correctionlm_prompt(
                data_item, examples=examples, given_context=predicted_context)

   
        data_item['prompt'] = prompt_text

      
        try:
            completion = LM_instance.completion(prompt_text)     
            print("full completion: ", completion)

            stop = ['--', '\n', ';', '#']
            stop_index = len(completion)
            for i, c in enumerate(completion):
                if c in stop:
                    stop_index = i
                    break
            completion = completion[:stop_index]

            completion = completion.split("FROM")[1].strip()
            completion = completion.split(";")[0].strip()
            temp_parse = sql_pred_parse(completion)
            print("completion: ", completion)

        except:
            completion = ""

        # aggregate the prediction and the history states
        predicted_slot_values = {}
        try:
            predicted_slot_values = sql_pred_parse(completion)  # a dictionary
            print(predicted_slot_values)
        except:
            print("the output is not a valid SQL query")
            data_item['not_valid'] = 1
            
        context_slot_values = data_item['last_slot_values']  # a dictionary

        # merge context and prediction
        if use_gold:
            all_slot_values = context_slot_values.copy()
        else:
            all_slot_values = prediction_recorder.state_retrieval(
                data_item).copy()


        final_predicted_slot_values = {}
        for domain_slot, value in predicted_slot_values.items():
            if domain_slot in all_slot_values and value == all_slot_values[domain_slot]:
                continue
            else:
                final_predicted_slot_values[domain_slot] = value



        for s, v in predicted_slot_values.items():
            if s in all_slot_values and v == "[DELETE]":
                del all_slot_values[s]
            elif v != "[DELETE]":
                all_slot_values[s] = v

        # some slots may contain multiple values
        all_slot_values = {k: v.split('|')[0] for k, v in all_slot_values.items()}
        
        # record current turn prediction
        prediction_recorder.add_state(data_item, all_slot_values)


        data_item['pred_turn_change_second_pass'] = final_predicted_slot_values
        data_item['pred_second_pass'] = all_slot_values
        data_item['completion'] = completion
        all_result.append(data_item)

        # print the result
        print(f"this is the {n_total - 1}th example. {data_item['ID']}_turn_{data_item['turn_id']}", flush=True)
        print(f"pred turn change: {sv_dict_to_string(final_predicted_slot_values, sep='-')}", flush=True)
        print(f"gold turn change: {sv_dict_to_string(data_item['turn_slot_values'], sep='-')}", flush=True)
        print(f"pred states: {sv_dict_to_string(all_slot_values, sep='-')}", flush=True)
        print(f"gold states: {sv_dict_to_string(data_item['slot_values'], sep='-')}", flush=True)

        this_jga, this_acc, dst_f1 = evaluate(all_slot_values,data_item['slot_values'])
        this_tlb, _, tlb_f1 = evaluate(final_predicted_slot_values,data_item['turn_slot_values'])
        total_acc += this_acc
        total_dst_f1 += dst_f1
        total_tlb_f1 += tlb_f1
        if this_jga:
            n_correct += 1
            result_dict[data_item['turn_id']].append(1)
        else:
            result_dict[data_item['turn_id']].append(0)
        
        if this_tlb:
            n_correct_tlb += 1
            print("\n=====================correct!=======================", flush=True)
        else:
            print("\n=====================wrong!=======================", flush=True)

        print("\n", flush=True)
    print(f"jga correct {n_correct}/{n_total}  =  {n_correct / n_total}")
    print(f"tlb correct {n_correct_tlb}/{n_total}  =  {n_correct_tlb / n_total}")
    print(f"Slot Acc {total_acc/n_total}")
    print(f"DST Joint F1 {total_dst_f1/n_total}")
    print(f"TLB Joint F1 {total_tlb_f1/n_total}")
    print()


    # calculate the accuracy of each turn
    for k, v in result_dict.items():
        print(f"accuracy of turn {k} is {sum(v)}/{len(v)} = {sum(v) / len(v)}")

    return all_result


if __name__ == "__main__":

    all_results = run(test_set)

    with open(os.path.join(args.output_dir, "running_log.json"), 'w') as f:
        json.dump(all_results, f, indent=4)
