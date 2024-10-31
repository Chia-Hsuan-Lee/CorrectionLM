import json
import random
import copy
from glob import glob
import argparse
from tqdm import tqdm

def load_schema(schema_path):
    with open(schema_path, 'r') as f:
        return json.load(f)

def load_dialogues(dialogue_path):
    dialogues = []
    for file in glob(dialogue_path):
        with open(file, 'r') as f:
            dialogues.extend(json.load(f))
    return dialogues

def conversion(string):
    return "none" if not string else string

def process_dialogues(dialogues, schema, percentage):
    num_dials = len(dialogues)
    sampled_num = int(num_dials * percentage)
    sampled_dialogues = random.sample(dialogues, sampled_num)
    print("Sampled dialogues:", len(sampled_dialogues))

    out_dials = []
    for dial in tqdm(sampled_dialogues):
        prev_dst = {}
        turn_id = -1
        cur_dial, cur_turn = "", ""
        domains = dial["services"]

        schema_str = "".join(schema[domain] for domain in domains)
        dial_id = dial["dialogue_id"]
        sys_uttrs = [""]
        usr_uttrs = []

        for turn in dial["turns"]:
            cur_dst = copy.deepcopy(prev_dst)
            tlb = {}
            speaker = turn["speaker"].lower()
            uttr = turn["utterance"]
            cur_dial += f" [{speaker}] {uttr}"
            cur_turn += f" [{speaker}] {uttr}"
            if speaker == "user":
                usr_uttrs.append(uttr)
            else:
                sys_uttrs.append(uttr)

            if speaker == "user":
                turn_id += 1
                for frame in turn["frames"]:
                    domain = frame["service"]
                    cur_slot_values = copy.deepcopy(frame["state"]["slot_values"])
                    for slot, values in cur_slot_values.items():
                        if domain + "-" + slot not in prev_dst:
                            value = random.sample(values, 1)[0]
                            tlb[domain + "-" + slot] = value
                            cur_dst[domain + "-" + slot] = values
                        else:
                            for value in values:
                                if value not in prev_dst[domain + "-" + slot]:
                                    tlb[domain + "-" + slot] = value
                                    cur_dst[domain + "-" + slot].append(value)
                                    break

                tlb_target_string = conversion(" [sep] ".join(f"{ds}=={value}" for ds, value in tlb.items()))
                dst_target_string = conversion(" [sep] ".join(f"{ds}=={'|'.join(values)}" for ds, values in cur_dst.items()))
                t5_dst_target_string = conversion(" [sep] ".join(f"{ds}=={random.sample(values, 1)[0]}" for ds, values in cur_dst.items()))
                prev_target_string = conversion(" [sep] ".join(f"{ds}=={'|'.join(values)}" for ds, values in prev_dst.items()))
                t5_prev_target_string = conversion(" [sep] ".join(f"{ds}=={random.sample(values, 1)[0]}" for ds, values in prev_dst.items()))

                line = {
                    "dialogue_id": dial_id,
                    "turn_id": turn_id,
                    "dialogue": cur_dial,
                    "turn_pair": cur_turn,
                    "dst": dst_target_string,
                    "t5_dst": t5_dst_target_string,
                    "tlb": tlb_target_string,
                    "prev_dst": prev_target_string,
                    "t5_prev_dst": t5_prev_target_string,
                    "prev_dst_turn_pair": f"[context] {t5_prev_target_string} {cur_turn}",
                    "schema_prev_dst_turn_pair": f"[schema] {schema_str} [context] {t5_prev_target_string} {cur_turn}",
                    "schema_prev_dst_turn_pair_reversed": f"[context] {t5_prev_target_string} {cur_turn} [schema] {schema_str}",
                    "ID": dial_id.replace("_", "|"),
                    "domains": copy.deepcopy(domains),
                    "dialog": {"sys": copy.deepcopy(sys_uttrs), "usr": copy.deepcopy(usr_uttrs)},
                    "slot_values": copy.deepcopy({ds: "|".join(values) for ds, values in cur_dst.items()}),
                    "turn_slot_values": copy.deepcopy(tlb),
                    "last_slot_values": copy.deepcopy({ds: "|".join(values) for ds, values in prev_dst.items()})
                }

                out_dials.append(line)
                prev_dst = copy.deepcopy(cur_dst)
                cur_turn = ""

    return out_dials

def main():
    parser = argparse.ArgumentParser(description="Process and sample dialogues from a dataset.")
    parser.add_argument('--schema_path', type=str, required=True, help="Path to the schema file.")
    parser.add_argument('--dialogue_path', type=str, required=True, help="Glob path to the dialogue files.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the output JSON file.")
    parser.add_argument('--percentage', type=float, default=0.05, help="Percentage of dialogues to sample.")
    args = parser.parse_args()

    random.seed(0)
    schema = load_schema(args.schema_path)
    dialogues = load_dialogues(args.dialogue_path)
    out_dials = process_dialogues(dialogues, schema, args.percentage)

    with open(args.output_path, 'w') as out_file:
        json.dump(out_dials, out_file, indent=4)

if __name__ == "__main__":
    main()
