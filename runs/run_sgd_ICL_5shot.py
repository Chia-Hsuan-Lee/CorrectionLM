import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
import copy
from collections import defaultdict
from tqdm import tqdm
from config import CONFIG

from utils.helper import SpeedLimitTimer, PreviousStateRecorder, clean_sql_completion
from utils.gpt4o_completion import chatgpt_completion
from utils.llama_quantization_completion_class import LM
from utils.sql_sgd import sql_pred_parse, sv_dict_to_string

from prompts.prompting_sgd_vanilla_icl import get_icl_prompt
from eval.evaluate_metrics import evaluate

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default="./expts/debug", help="directory to save running log and configs")
parser.add_argument('--test_fn', type=str, default='', help="file to evaluate on, empty means use the test set")
parser.add_argument('--lm', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct',
                    help="langauge model to use for zeroshot prompting. Input should be gpt4 or a valid HF model name")
args = parser.parse_args()

# create the output folder
os.makedirs(args.output_dir, exist_ok=True)

with open(os.path.join(args.output_dir, "exp_config.json"), 'w') as f:
    json.dump(vars(args), f, indent=4)

NUM_EXAMPLE=5

# evaluate on some other file
if args.test_fn:
    test_set_path = args.test_fn

print("Running ICL on: ", test_set_path)

with open(test_set_path) as f:
    test_set = json.load(f)



demonstration_examples = [
    {
        "dialogue_id": "104_00112",
        "turn_id": 8,
        "dialog": {
            "sys": [
                "",
                "What is your desired category of event and desired City to search for?",
                "Found 4 events and one of which is Padres Vs Brewers at Petco park. What about your opinion on this event.",
                "What about an event like padres Vs Diamondbacks at Petco Park, Is that okay for you?",
                "Shall I purchase tickets to this event now?",
                "How can I help you further?",
                "I got 10 results. What do you think about a Hotel name Bahia Resort hotel with 3 stars? Is that okay for you.",
                "Shall I book the rooms in that hotel?",
                "How many days you need the reservation for the rooms and from which date you want me to book the rooms?"
            ],
            "usr": [
                "I want to find some nice events for which I need your help to find the best one.",
                "Search for something like Sports in San Diego.",
                "Search for some other events that should perfectly match my requirements.",
                "Yes that is perfect.",
                "No, not now, may be later.",
                "Please search for some nice hotels to stay there for a while.",
                "Yes, I like it.",
                "Yes, do it for me.",
                "Book the rooms for three days from 3rd of March."
                ]
            },
        "slot_values": {
            "Hotels_1-check_in_date": "3rd of March",
            "Hotels_1-destination": "San Diego",
            "Hotels_1-hotel_name": "Bahia Resort hotel",
            "Hotels_1-number_of_days": "three"
        },
        "turn_slot_values": {
            "Hotels_1-check_in_date": "3rd of March",
            "Hotels_1-number_of_days": "three"
        },
        "last_slot_values": {
            "Hotels_1-destination": "San Diego",
            "Hotels_1-hotel_name": "Bahia Resort hotel"
        }
    },
    {
        "dialogue_id": "37_00098",
        "turn_id": 0,
        "dialogue": " [user] Can you find me a one way flight for a trip I have coming up? I'd like to get an Economy ticket with American Airlines.",
        "turn_pair": " [user] Can you find me a one way flight for a trip I have coming up? I'd like to get an Economy ticket with American Airlines.",
        "dst": "Flights_1-airlines=American Airlines, Flights_1-seating_class=Economy, ",
        "t5_dst": "Flights_1-airlines=American Airlines, Flights_1-seating_class=Economy, ",
        "tlb": "Flights_1-airlines=American Airlines, Flights_1-seating_class=Economy, ",
        "prev_dst": "",
        "t5_prev_dst": "",
        "prev_dst_turn_pair": "[context]   [user] Can you find me a one way flight for a trip I have coming up? I'd like to get an Economy ticket with American Airlines.",
        "ID": "37|00098",
        "domains": [
            "Flights_1"
        ],
        "dialog": {
            "sys": [
                ""
            ],
            "usr": [
                "Can you find me a one way flight for a trip I have coming up? I'd like to get an Economy ticket with American Airlines."
            ]
        },
        "slot_values": {
            "Flights_1-airlines": "American Airlines",
            "Flights_1-seating_class": "Economy"
        },
        "turn_slot_values": {
            "Flights_1-airlines": "American Airlines",
            "Flights_1-seating_class": "Economy"
        },
        "last_slot_values": {}
    },
    {
        "dialogue_id": "31_00028",
        "turn_id": 5,
        "dialogue": " [user] I am looking to move out of my share-house. Can you help me find a new apartment? [system] Yeah, I'm sure I could help, how many bedrooms are we talking? [user] I'm interested in a 4 bedroom unit [system] Okay, which part of town are you interested in? [user] I'm interested in Cupertino. [system] Great, I have found a good property that matches your criteria. The property is at Forge Homestead Apartments, 20667 Forge Way, it is a 4 bedroom, 3 bath unit renting at 4800 dollars a month. [user] That sounds good, what's the phone number I can contact to inquire? [system] You can call 408-739-0870 [user] Thank you. Any other apartments in the area? [system] Yes, there's another one at Gardens Of Fontainbleu Apartments, 10200 Miller Avenue, it is a 4 bedroom, 3 bath unit renting at 4950 per month. No pets are allowed in this particular unit though. [user] Thank you. Can you find me other apartments that also allow pets?",
        "turn_pair": " [system] Yes, there's another one at Gardens Of Fontainbleu Apartments, 10200 Miller Avenue, it is a 4 bedroom, 3 bath unit renting at 4950 per month. No pets are allowed in this particular unit though. [user] Thank you. Can you find me other apartments that also allow pets?",
        "dst": "Homes_1-area=Cupertino, Homes_1-number_of_beds=4, Homes_1-pets_allowed=True, ",
        "t5_dst": "Homes_1-area=Cupertino, Homes_1-number_of_beds=4, Homes_1-pets_allowed=True, ",
        "tlb": "Homes_1-pets_allowed=True, ",
        "prev_dst": "Homes_1-area=Cupertino, Homes_1-number_of_beds=4, ",
        "t5_prev_dst": "Homes_1-area=Cupertino, Homes_1-number_of_beds=4, ",
        "prev_dst_turn_pair": "[context] Homes_1-area=Cupertino, Homes_1-number_of_beds=4,   [system] Yes, there's another one at Gardens Of Fontainbleu Apartments, 10200 Miller Avenue, it is a 4 bedroom, 3 bath unit renting at 4950 per month. No pets are allowed in this particular unit though. [user] Thank you. Can you find me other apartments that also allow pets?",
        "ID": "31|00028",
        "domains": [
            "Homes_1"
        ],
        "dialog": {
            "sys": [
                "",
                "Yeah, I'm sure I could help, how many bedrooms are we talking?",
                "Okay, which part of town are you interested in?",
                "Great, I have found a good property that matches your criteria. The property is at Forge Homestead Apartments, 20667 Forge Way, it is a 4 bedroom, 3 bath unit renting at 4800 dollars a month.",
                "You can call 408-739-0870",
                "Yes, there's another one at Gardens Of Fontainbleu Apartments, 10200 Miller Avenue, it is a 4 bedroom, 3 bath unit renting at 4950 per month. No pets are allowed in this particular unit though."
            ],
            "usr": [
                "I am looking to move out of my share-house. Can you help me find a new apartment?",
                "I'm interested in a 4 bedroom unit",
                "I'm interested in Cupertino.",
                "That sounds good, what's the phone number I can contact to inquire?",
                "Thank you. Any other apartments in the area?",
                "Thank you. Can you find me other apartments that also allow pets?"
            ]
        },
        "slot_values": {
            "Homes_1-area": "Cupertino",
            "Homes_1-number_of_beds": "4",
            "Homes_1-pets_allowed": "True"
        },
        "turn_slot_values": {
            "Homes_1-pets_allowed": "True"
        },
        "last_slot_values": {
            "Homes_1-area": "Cupertino",
            "Homes_1-number_of_beds": "4"
        }
    },
    {
        "dialogue_id": "105_00010",
        "turn_id": 3,
        "dialogue": " [user] I want to go to new york city from washington on a bus [system] When are you leaving and how many tickets do you need [user] i need one tickets leaving on 13th of march [system] What time are you leaving [user] I want to leave at 11:30 [system] Please confirm your booking for 1 ticket from washington to new york on march 13th at 11:30 am [user] Yes that works",
        "turn_pair": " [system] Please confirm your booking for 1 ticket from washington to new york on march 13th at 11:30 am [user] Yes that works",
        "dst": "Buses_1-from_location=washington, Buses_1-leaving_date=13th of march|march 13th, Buses_1-leaving_time=11:30|11:30 am, Buses_1-to_location=new york|new york city, Buses_1-travelers=1, ",
        "t5_dst": "Buses_1-from_location=washington, Buses_1-leaving_date=march 13th, Buses_1-leaving_time=11:30 am, Buses_1-to_location=new york city, Buses_1-travelers=1, ",
        "tlb": "Buses_1-leaving_date=march 13th, Buses_1-leaving_time=11:30 am, Buses_1-to_location=new york, ",
        "prev_dst": "Buses_1-from_location=washington, Buses_1-leaving_date=13th of march, Buses_1-leaving_time=11:30, Buses_1-to_location=new york city, Buses_1-travelers=1, ",
        "t5_prev_dst": "Buses_1-from_location=washington, Buses_1-leaving_date=13th of march, Buses_1-leaving_time=11:30, Buses_1-to_location=new york city, Buses_1-travelers=1, ",
        "prev_dst_turn_pair": "[context] Buses_1-from_location=washington, Buses_1-leaving_date=13th of march, Buses_1-leaving_time=11:30, Buses_1-to_location=new york city, Buses_1-travelers=1,   [system] Please confirm your booking for 1 ticket from washington to new york on march 13th at 11:30 am [user] Yes that works",
        "ID": "105|00010",
        "domains": [
            "Buses_1",
            "Hotels_1"
        ],
        "dialog": {
            "sys": [
                "",
                "When are you leaving and how many tickets do you need",
                "What time are you leaving",
                "Please confirm your booking for 1 ticket from washington to new york on march 13th at 11:30 am"
            ],
            "usr": [
                "I want to go to new york city from washington on a bus",
                "i need one tickets leaving on 13th of march",
                "I want to leave at 11:30",
                "Yes that works"
            ]
        },
        "slot_values": {
            "Buses_1-from_location": "washington",
            "Buses_1-leaving_date": "13th of march|march 13th",
            "Buses_1-leaving_time": "11:30|11:30 am",
            "Buses_1-to_location": "new york|new york city",
            "Buses_1-travelers": "1"
        },
        "turn_slot_values": {
            "Buses_1-leaving_date": "march 13th",
            "Buses_1-leaving_time": "11:30 am",
            "Buses_1-to_location": "new york"
        },
        "last_slot_values": {
            "Buses_1-from_location": "washington",
            "Buses_1-leaving_date": "13th of march",
            "Buses_1-leaving_time": "11:30",
            "Buses_1-to_location": "new york city",
            "Buses_1-travelers": "1"
        }
    },
    {
        "dialogue_id": "119_00043",
        "turn_id": 0,
        "dialogue": " [user] I need a spacious rental car to pick up half past 1 in the afternoon on the 13th of March.",
        "turn_pair": " [user] I need a spacious rental car to pick up half past 1 in the afternoon on the 13th of March.",
        "dst": "RentalCars_2-car_type=Full-size, RentalCars_2-dropoff_date=13th of March, RentalCars_2-pickup_time=half past 1 in the afternoon, ",
        "t5_dst": "RentalCars_2-car_type=Full-size, RentalCars_2-dropoff_date=13th of March, RentalCars_2-pickup_time=half past 1 in the afternoon, ",
        "tlb": "RentalCars_2-car_type=Full-size, RentalCars_2-dropoff_date=13th of March, RentalCars_2-pickup_time=half past 1 in the afternoon, ",
        "prev_dst": "",
        "t5_prev_dst": "",
        "prev_dst_turn_pair": "[context]   [user] I need a spacious rental car to pick up half past 1 in the afternoon on the 13th of March.",
        "ID": "119|00043",
        "domains": [
            "RentalCars_2",
            "Homes_1"
        ],
        "dialog": {
            "sys": [
                ""
            ],
            "usr": [
                "I need a spacious rental car to pick up half past 1 in the afternoon on the 13th of March."
            ]
        },
        "slot_values": {
            "RentalCars_2-car_type": "Full-size",
            "RentalCars_2-dropoff_date": "13th of March",
            "RentalCars_2-pickup_time": "half past 1 in the afternoon"
        },
        "turn_slot_values": {
            "RentalCars_2-car_type": "Full-size",
            "RentalCars_2-dropoff_date": "13th of March",
            "RentalCars_2-pickup_time": "half past 1 in the afternoon"
        },
        "last_slot_values": {}
    }

]

examples = demonstration_examples


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
     
       
        # record the prompt
        data_item['prompt'] = prompt_text

        # chatgpt completion
        complete_flag = False
        parse_error_count = 0
        while not complete_flag:
            try:
                completion = lm_completion(prompt_text)
                completion = clean_sql_completion(completion)

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
                    # check if chatgpt is crazy
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
            print(predicted_slot_values)
        except:
            print("the output is not a valid SQL query")
            data_item['not_valid'] = 1
        print("predicted_slot_values:    ", predicted_slot_values)

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

        # record the predictions
        data_item['pred_turn_change'] = final_predicted_slot_values
        data_item['pred'] = all_slot_values
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
