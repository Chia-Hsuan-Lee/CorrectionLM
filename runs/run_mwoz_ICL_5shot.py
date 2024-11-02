# adapted from https://github.com/Yushi-Hu/IC-DST/blob/main/run_zeroshot_codex_experiment.py
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
from utils.gpt4o_completion import chatgpt_completion
from utils.llama_quantization_completion_class import LM

from prompts.prompting_mwoz_vanilla_icl import get_icl_prompt, conversion, table_prompt
from eval.evaluate_metrics import evaluate

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default="./expts/zero-shot",
                    help="directory to save running log and configs")
parser.add_argument('--mwz_ver', type=str, default="2.4",
                    choices=['2.1', '2.4'], help="version of MultiWOZ")
parser.add_argument('--test_fn', type=str, default='',
                    help="file to evaluate on, empty means use the test set")
parser.add_argument('--lm', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct',
                    help="langauge model to use for zeroshot prompting. Input should be gpt4 or a valid HF model name")
args = parser.parse_args()

# create the output folder
os.makedirs(args.output_dir, exist_ok=True)

with open(os.path.join(args.output_dir, "exp_config.json"), 'w') as f:
    json.dump(vars(args), f, indent=4)

NUM_EXAMPLE = 5

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

with open(ontology_path) as f:
    ontology = json.load(f)
with open(test_set_path) as f:
    test_set = json.load(f)
    

demonstration_examples = [
    {
        "ID": "PMUL1605.json",
        "turn_id": 4,
        "domains": [
            "train",
            "hotel"
        ],
        "dialog": {
            "sys": [
                "",
                "sure ! where will you be departing from ?",
                "i am sorry , i am confused . you are leaving from cambridge to go to cambridge ?",
                "train tr0189 arrives at 18:01 . should i reserve tickets for that 1 ?",
                "booking was successful , the total fee is 71.59 gbp payable at the station . reference number is : v7h3p3g7 . anything else ?"
            ],
            "usr": [
                "i am looking for a train on tuesday to arrive in cambridge by 19:00 , can you help me please ?",
                "i will be leaving from cambridge .",
                "no . i am leaving from cambridge to go to broxbourne .",
                "sure please make a booking for 4 people .",
                "yes , i am looking for a 2 star hotel in the west part of town ."
            ]
        },
        "slot_values": {
            "hotel-area": "west",
            "hotel-stars": "2",
            "train-book people": "4",
            "train-destination": "broxbourne",
            "train-day": "tuesday",
            "train-arriveby": "19:00",
            "train-departure": "cambridge"
        },
        "turn_slot_values": {
            "hotel-area": "west",
            "hotel-stars": "2"
        },
        "last_slot_values": {
            "train-book people": "4",
            "train-destination": "broxbourne",
            "train-day": "tuesday",
            "train-arriveby": "19:00",
            "train-departure": "cambridge"
        }
    },
    {
        "ID": "MUL1754.json",
        "turn_id": 1,
        "domains": [
            "train",
            "attraction"
        ],
        "dialog": {
            "sys": [
                "",
                "i can help you with that what type of attraction would you like to go too ?"
            ],
            "usr": [
                "i would like to find a place to go that s in the centre .",
                "any type is fine , i am not picky ."
            ]
        },
        "slot_values": {
            "attraction-type": "dontcare",
            "attraction-area": "centre"
        },
        "turn_slot_values": {
            "attraction-type": "dontcare"
        },
        "last_slot_values": {
            "attraction-type": "architecture",
            "attraction-area": "centre"
        }
    },
    {
        "ID": "MUL0658.json",
        "turn_id": 0,
        "domains": [
            "train",
            "hotel"
        ],
        "dialog": {
            "sys": [
                ""
            ],
            "usr": [
                "i would be in cambridge and i need to find a train that will be leaving from peterborough this sunday , can you help me find 1 ?"
            ]
        },
        "slot_values": {
            "train-destination": "cambridge",
            "train-day": "sunday",
            "train-departure": "peterborough"
        },
        "turn_slot_values": {
            "train-destination": "cambridge",
            "train-day": "sunday",
            "train-departure": "peterborough"
        },
        "last_slot_values": {}
    },
    {
        "ID": "SNG02322.json",
        "turn_id": 0,
        "domains": [
            "taxi"
        ],
        "dialog": {
            "sys": [
                ""
            ],
            "usr": [
                "i want a taxi to pick me up after 21:00 to take me to little saint marys church ."
            ]
        },
        "slot_values": {
            "taxi-leaveat": "21:00",
            "taxi-destination": "little saint marys church"
        },
        "turn_slot_values": {
            "taxi-leaveat": "21:00",
            "taxi-destination": "little saint marys church"
        },
        "last_slot_values": {}
    },
    {
        "ID": "MUL2395.json",
        "turn_id": 2,
        "domains": [
            "restaurant",
            "attraction"
        ],
        "dialog": {
            "sys": [
                "",
                "there are 79 attractions in the city . do you have a specific type of attraction or specific area of the attraction that you are interested in ?",
                "great ! how about all saints church ? it has fantastic architecture and free entrance . would you like more information ?"
            ],
            "usr": [
                "i am planning a trip to town and want to sight see a bit . can you let me know some attractions i may be interested in ?",
                "i would like something in the centre .",
                "could i get the address for it ? i would also like an expensive place to eat around it ."
            ]
        },
        "slot_values": {
            "restaurant-pricerange": "expensive",
            "restaurant-area": "centre",
            "attraction-area": "centre"
        },
        "turn_slot_values": {
            "restaurant-pricerange": "expensive",
            "restaurant-area": "centre"
        },
        "last_slot_values": {
            "attraction-area": "centre"
        }
    },

]


def run(test_set, turn=-1, use_gold=False):
    # turn and use_gold are for analysis purpose
    # turn = -1 means evalute all dialogues
    # turn = 0 means evaluate single-turn dialogues
    # turn = 1 means evalute two-turn dialogues... etc.
    # when use_gold = True, the context are gold context (for analysis purpose)

    # openai limitation 20 queries/min
    timer = SpeedLimitTimer(second_per_step=3.1)

    result_dict = defaultdict(list)  # use to record the accuracy

    selected_set = test_set
    # if needed, only evaluate on particular turns (analysis purpose)
    if turn >= 0:
        if not use_gold:
            raise ValueError(
                "can only evaluate particular turn when using gold context")
        selected_set = [d for d in test_set if len(
            d['dialog']['usr']) == turn + 1]

    prediction_recorder = PreviousStateRecorder()  # state recorder

    # start experiment
    all_result = []
    n_total = 0
    n_correct = 0
    n_correct_tlb = 0
    total_acc = 0
    total_dst_f1 = 0
    total_tlb_f1 = 0

    if args.lm == "gpt4":
        lm_completion = chatgpt_completion
    else:
        LM_instance = LM(args.lm)
        lm_completion = LM_instance.completion

    for data_item in tqdm(selected_set):
        n_total += 1

        completion = ""
        if use_gold:
            prompt_text = get_icl_prompt(
                data_item, examples=demonstration_examples)
        else:
            predicted_context = prediction_recorder.state_retrieval(data_item)
            modified_item = copy.deepcopy(data_item)
            modified_item['last_slot_values'] = predicted_context
            examples = demonstration_examples
            prompt_text = get_icl_prompt(
                data_item, examples=examples, given_context=predicted_context)

        # print the retrieved examples (without the sql table)
        print("input: ", prompt_text.replace(conversion(table_prompt), ""))

        # record the prompt
        data_item['prompt'] = prompt_text

        # LLM completion
        complete_flag = False
        parse_error_count = 0
        while not complete_flag:
            try:
                completion = lm_completion(prompt_text)
                # convert back the sql completion result
                completion = conversion(completion, reverse=True)
                completion = completion.replace("```sql", "").strip()
                completion = completion.replace("```", "").strip()
                
            except Exception as e:
                if e.user_message.startswith("This model's maximum context length"):
                    print("prompt overlength")
                    examples = examples[1:]
                    prompt_text = get_icl_prompt(
                        data_item, examples=examples, given_context=predicted_context)
                else:
                    # throughput too high
                    timer.sleep(10)
            else:
                try:
                    # check if CODEX is crazy
                    temp_parse = sql_pred_parse(completion)
                except:
                    parse_error_count += 1
                    if parse_error_count >= 5:
                        complete_flag = True
                else:
                    complete_flag = True
            # limit query speed
            timer.step()

        # aggregate the prediction and the history states
        predicted_slot_values = {}
        try:
            predicted_slot_values = sql_pred_parse(completion)  # a dictionary
        except:
            print("the output is not a valid SQL query")
            data_item['not_valid'] = 1
        predicted_slot_values = typo_fix(
            predicted_slot_values, ontology=ontology, version=args.mwz_ver)

        predicted_slot_values = {k:v for k,v in predicted_slot_values.items() if k in ontology}

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
        all_slot_values = {k: v.split('|')[0]
                           for k, v in all_slot_values.items()}

        # record current turn prediction
        prediction_recorder.add_state(data_item, all_slot_values)


        print("completion: ", completion)


        # record the predictions
        data_item['pred_turn_change'] = final_predicted_slot_values
        data_item['pred'] = all_slot_values
        data_item['ontology_path'] = ontology_path
        data_item['completion'] = completion
        all_result.append(data_item)

        # print the result
        print(
            f"this is the {n_total - 1}th example. {data_item['ID']}_turn_{data_item['turn_id']}")
        print(
            f"pred turn change: {sv_dict_to_string(final_predicted_slot_values, sep='-')}")
        print(
            f"gold turn change: {sv_dict_to_string(data_item['turn_slot_values'], sep='-')}")
        print(f"pred states: {sv_dict_to_string(all_slot_values, sep='-')}")
        print(
            f"gold states: {sv_dict_to_string(data_item['slot_values'], sep='-')}")

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

    print(f"DST JGA {n_correct}/{n_total}  =  {n_correct / n_total}")
    print(f"TLB JGA {n_correct_tlb}/{n_total}  =  {n_correct_tlb / n_total}")
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
