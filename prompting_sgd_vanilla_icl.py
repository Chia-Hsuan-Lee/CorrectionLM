import json
from utils.sql import slot_values_to_seq_sql

test_ontology_path = "./data/sgd/ontology/sgd_ontology_3rows.json"
test_ontology = json.load(open(test_ontology_path))


def get_icl_prompt(data_item, examples, given_context=None, n_examples=None):
    """
    You can try different prompt in here.
    """
    question_item = data_item
   
    prompt_text = f""

    max_n_examples = len(examples)
    if n_examples is not None:
        max_n_examples = n_examples

    # in case for zero-shot learning
    if max_n_examples > 0:
        for example_id, example in enumerate(examples[-max_n_examples:]):
            prompt_text += f"Example #{example_id + 1}\n"

            sys_list = example['dialog']['sys']
            n = len(sys_list)
            if n >= 2:
                few_sys_utts = sys_list[-min(3, n):-1]
                usr_list = example['dialog']['usr']
                few_usr_utts = usr_list[-min(3, n):-1]

                for sys_utt, usr_utt in zip(few_sys_utts, few_usr_utts):
                    if sys_utt == 'none':
                        sys_utt = ''
                    if usr_utt == 'none':
                        usr_utt = ''
                    prompt_text += f"[system] {sys_utt}\n"
                    prompt_text += f"[user] {usr_utt}\n"

            # remove multiple choice in last slot values
            last_slot_values = {s: v.split(
                '|')[0] for s, v in example['last_slot_values'].items()}
            prompt_text += f"[context] {', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()})}\n"

            last_sys_utt = example['dialog']['sys'][-1]
            if last_sys_utt == 'none':
                last_sys_utt = ''
            prompt_text += f"[system] {last_sys_utt}\n"
            prompt_text += f"Q: [user] {example['dialog']['usr'][-1]}\n"

            prompt_text += f"SQL: {slot_values_to_seq_sql(example['turn_slot_values'])};\n"
            prompt_text += "\n\n"


    question_domains = question_item["domains"]
    for domain in question_domains:
        prompt_text += f"{(test_ontology[domain])}\n"


    prompt_text += f"-- Using valid SQLite, answer the following multi-turn conversational questions for the tables provided above.\n"
    prompt_text += f"Example #{max_n_examples + 1}\n"

    sys_list = question_item['dialog']['sys']
    n = len(sys_list)
    if n >= 2:
        few_sys_utts = sys_list[-min(3, n):-1]

        usr_list = question_item['dialog']['usr']
        few_usr_utts = usr_list[-min(3, n):-1]

        for sys_utt, usr_utt in zip(few_sys_utts, few_usr_utts):
            if sys_utt == 'none':
                sys_utt = ''
            if usr_utt == 'none':
                usr_utt = ''
            prompt_text += f"[system] {sys_utt}\n"
            prompt_text += f"[user] {usr_utt}\n"

    if given_context is None:
        last_slot_values = {s: v.split(
            '|')[0] for s, v in question_item['last_slot_values'].items()}
    else:
        last_slot_values = given_context
    prompt_text += f"[context] {', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()})}\n"

    last_sys_utt = question_item['dialog']['sys'][-1]
    if last_sys_utt == 'none':
        last_sys_utt = ''
    prompt_text += f"[system] {last_sys_utt}\n"
    prompt_text += f"Q: [user] {question_item['dialog']['usr'][-1]}\n"
    prompt_text += "SQL: SELECT * FROM"

    return prompt_text