from utils.sql import slot_values_to_seq_sql
import json

correction_instruction = """
-- You are an advanced reasoning agent that can improve based on self refection. Your job here is to perform dialogue state tracking task by outputing valid SQLite.
You are given a couple of things in the following order:
(1) Schema: A predifined structure that grounds the dialogue states. It is shown in the format of SQL tables.
(2) Dialogue state tracking task demonstrations: Each demonstration contains a dialogue between a user and a human agent. Each demonstration also includes a Correct SQL and Predicted SQL. Correct SQL is the ground truth dialogue states and Predicted SQL is an incorrect prediction produced by an ineffective dialogue state tracking system.
Reason on the paired Correct SQL and Predicted SQL and learn from the mistakes. Then produce the Correct SQL (in SQLite) for a newly provided example.
"""


ontology_path = "data/sgd/ontology/sgd_ontology_3rows.json"
ontology = json.load(open(ontology_path))



def get_zeroshot_correction_prompt(data_item, examples, given_context=None, n_examples=None):
    """
    You can try different prompt in here.
    """
    question_item = data_item

    prompt_text = f"{correction_instruction}\n"

    max_n_examples = len(examples)
    if n_examples is not None:
        max_n_examples = n_examples

    prompt_text += "Here are some examples:\n"


    # in case for zero-shot learning
    if max_n_examples > 0:
        for example_id, example in enumerate(examples[-max_n_examples:]):
            prompt_text += f"Example #{example_id + 1}\n"

            # remove multiple choice in last slot values
            last_slot_values = {s: v.split(
                '|')[0] for s, v in example['last_slot_values'].items()}
            prompt_text += f"[context] {', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()})}\n"

            last_sys_utt = example['dialog']['sys'][-1]
            if last_sys_utt == 'none':
                last_sys_utt = ''
            prompt_text += f"[human agent] {last_sys_utt}\n"
            prompt_text += f"[user] {example['dialog']['usr'][-1]}\n"

            prompt_text += f"Predicted SQL: {slot_values_to_seq_sql(example['pred_turn_change'])};\n"
            prompt_text += f"Correct SQL: {slot_values_to_seq_sql(example['turn_slot_values'])};\n"
         
            prompt_text += "\n\n"

    question_domains = question_item["domains"]
    for domain in question_domains:
      prompt_text += f"{(ontology[domain])}\n"

    prompt_text += """Produce the Correct SQL for this new example. Output only the SQLite command.\n\n"""

    if given_context is None:
        last_slot_values = {s: v.split(
            '|')[0] for s, v in question_item['last_slot_values'].items()}
    else:
        last_slot_values = given_context
    prompt_text += f"[context] {', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()})}\n"

    last_sys_utt = question_item['dialog']['sys'][-1]
    if last_sys_utt == 'none':
        last_sys_utt = ''
    prompt_text += f"[human agent] {last_sys_utt}\n"
    prompt_text += f"[user] {question_item['dialog']['usr'][-1]}\n"
    prompt_text += f"Predicted SQL: {slot_values_to_seq_sql(question_item['pred_turn_change'])};\n"
    prompt_text += "Correct SQL: SELECT * FROM"

    return prompt_text


def get_correctionlm_prompt(data_item, examples, given_context=None, n_examples=None):
    """
    You can try different prompt in here.
    """
    question_item = data_item

    max_n_examples = len(examples)
    if n_examples is not None:
        max_n_examples = n_examples

    prompt_text = "Here are some examples:\n"

    # in case for zero-shot learning
    if max_n_examples > 0:
        for example_id, example in enumerate(examples[-(max_n_examples-1):]):
            prompt_text += f"Example #{example_id + 1}\n"
            example_domains = example["domains"]
            for domain in example_domains:
                prompt_text += f"{(ontology[domain])}\n"

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
            prompt_text += f"[human agent] {last_sys_utt}\n"
            prompt_text += f"[user] {example['dialog']['usr'][-1]}\n"

            prompt_text += f"Predicted SQL: {slot_values_to_seq_sql(example['pred_turn_change'])};\n"
            prompt_text += f"Correct SQL: {slot_values_to_seq_sql(example['turn_slot_values'])};\n"
         
            prompt_text += "\n\n"


    question_domains = question_item["domains"]
    for domain in question_domains:
        prompt_text += f"{(ontology[domain])}\n"

    prompt_text += """Produce the Correct SQL for this new example. Output only the SQLite command.\n\n"""


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

    last_slot_values = {s: v.split(
        '|')[0] for s, v in question_item['last_slot_values'].items()}
    
    prompt_text += f"[context] {', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()})}\n"

    last_sys_utt = question_item['dialog']['sys'][-1]
    if last_sys_utt == 'none':
        last_sys_utt = ''
    prompt_text += f"[human agent] {last_sys_utt}\n"
    prompt_text += f"[user] {question_item['dialog']['usr'][-1]}\n"
    prompt_text += f"Predicted SQL: {slot_values_to_seq_sql(question_item['pred_turn_change'])};\n"
    completion = f"Correct SQL: {slot_values_to_seq_sql(question_item['turn_slot_values'])};\n"

    return prompt_text, completion




def get_llama_SFT_prompt(data_item, examples, given_context=None, n_examples=None):
    """
    You can try different prompt in here.
    """
    question_item = examples[-2]

    max_n_examples = len(examples)
    if n_examples is not None:
        max_n_examples = n_examples

    prompt_text = "Here are some examples:\n"

    # in case for zero-shot learning
    if max_n_examples > 0:
        for example_id, example in enumerate(examples[-max_n_examples:-2]):
            prompt_text += f"Example #{example_id + 1}\n"

            example_domains = example["domains"]
            for domain in example_domains:
                prompt_text += f"{(ontology[domain])}\n"

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
            prompt_text += f"[human agent] {last_sys_utt}\n"
            prompt_text += f"[user] {example['dialog']['usr'][-1]}\n"

            prompt_text += f"Predicted SQL: {slot_values_to_seq_sql(example['pred_turn_change'])};\n"
            prompt_text += f"Correct SQL: {slot_values_to_seq_sql(example['turn_slot_values'])};\n"
         
            prompt_text += "\n\n"

    question_domains = question_item["domains"]
    for domain in question_domains:
        prompt_text += f"{(ontology[domain])}\n"

    prompt_text += """Produce the Correct SQL for this new example. Output only the SQLite command.\n\n"""

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

    last_slot_values = {s: v.split(
        '|')[0] for s, v in question_item['last_slot_values'].items()}

    prompt_text += f"[context] {', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()})}\n"

    last_sys_utt = question_item['dialog']['sys'][-1]
    if last_sys_utt == 'none':
        last_sys_utt = ''
    prompt_text += f"[human agent] {last_sys_utt}\n"
    prompt_text += f"[user] {question_item['dialog']['usr'][-1]}\n"
    prompt_text += f"Predicted SQL: {slot_values_to_seq_sql(question_item['pred_turn_change'])};\n"
    completion = f"Correct SQL: {slot_values_to_seq_sql(question_item['turn_slot_values'])};\n"

    return prompt_text, completion
