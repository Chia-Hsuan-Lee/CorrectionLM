from utils.sql import slot_values_to_seq_sql


correction_instruction = """
-- You are an advanced reasoning agent that can improve based on self refection. Your job here is to perform dialogue state tracking task by outputing valid SQLite.
You are given a couple of things in the following order:
(1) Schema: A predifined structure that grounds the dialogue states. It is shown in the format of SQL tables.
(2) Dialogue state tracking task demonstrations: Each demonstration contains a dialogue between a user and a human agent. Each demonstration also includes a Correct SQL and Predicted SQL. Correct SQL is the ground truth dialogue states and Predicted SQL is an incorrect prediction produced by an ineffective dialogue state tracking system.
Reason on the paired Correct SQL and Predicted SQL and learn from the mistakes. Then produce the Correct SQL (in SQLite) for a newly provided example.
"""

pure_table_prompt = """
CREATE TABLE hotel(
  name text,
  pricerange text CHECK (pricerange IN (dontcare, cheap, moderate, expensive)),
  type text CHECK (type IN (hotel, guest house)),
  parking text CHECK (parking IN (dontcare, yes, no)),
  book_stay int,
  book_day text,
  book_people int,
  area text CHECK (area IN (dontcare, centre, east, north, south, west)),
  stars int CHECK (stars IN (dontcare, 0, 1, 2, 3, 4, 5)),
  internet text CHECK (internet IN (dontcare, yes, no))
)
/*
4 example rows:
SELECT * FROM hotel LIMIT 4;
name  pricerange  type  parking book_stay book_day  book_people area  stars internet
a and b guest house moderate  guest house  dontcare  3 friday  5 east  4 yes
ashley hotel  expensive hotel yes 2 thursday  5 north 5 yes
el shaddia guest house  cheap guest house  yes 5 friday  2 centre  dontcare  no
express by holiday inn cambridge  dontcare  guest house yes 3 monday  2 east  dontcare  no
*/

CREATE TABLE train(
  destination text,
  departure text,
  day text,
  book_people int,
  leaveat text,
  arriveby text
)
/*
3 example rows:
SELECT * FROM train LIMIT 3;
destination departure day book_people leaveat arriveby
london kings cross  cambridge monday  6 dontcare 05:51
cambridge stansted airport  dontcare  1 20:24 20:52
peterborough  cambridge saturday  2  12:06  12:56
*/

CREATE TABLE attraction(
  name text,
  area text CHECK (area IN (dontcare, centre, east, north, south, west)),
  type text CHECK (type IN (architecture, boat, church, cinema, college, concert hall, entertainment, hotspot, multiple sports, museum, nightclub, park, special, swimming pool, theatre))
)
/*
4 example rows:
SELECT * FROM attraction LIMIT 4;
name area type
abbey pool and astroturf pitch  centre  swimming pool
adc theatre centre  theatre
all saints church dontcare  architecture
castle galleries  centre  museum
*/

CREATE TABLE restaurant(
  name text,
  food text,
  pricerange text CHECK (pricerange IN (dontcare, cheap, moderate, expensive)),
  area text CHECK (area IN (centre, east, north, south, west)),
  book_time text,
  book_day text,
  book_people int
)
/*
5 example rows:
SELECT * FROM restaurant LIMIT 5;
name  food  pricerange  area  book_time book_day  book_people
pizza hut city centre italian dontcare centre  13:30 wednesday 7
the missing sock  international moderate  east  dontcare dontcare  2
golden wok chinese moderate north 17:11 friday 4
cambridge chop house  dontcare  expensive  center 08:43 monday  5
darrys cookhouse and wine shop  modern european expensive center  11:20 saturday  8
*/

CREATE TABLE taxi(
  destination text,
  departure text,
  leaveat text,
  arriveby text
)
/*
3 example rows:
SELECT * FROM taxi LIMIT 3;
destination departure leaveat arriveby
copper kettle royal spice 14:45 15:30
magdalene college  university arms hotel dontcare  15:45
lovell lodge  da vinci pizzeria 11:45 dontcare
*/
"""

def conversion(prompt, reverse=False):
    conversion_dict = {"leaveat": "depart_time", "arriveby": "arrive_by_time",
                       "book_stay": "book_number_of_days",
                       "food": "food_type"}
    reverse_conversion_dict = {v: k for k, v in conversion_dict.items()}
    used_dict = reverse_conversion_dict if reverse else conversion_dict

    for k, v in used_dict.items():
        prompt = prompt.replace(k, v)
    return prompt



def get_zeroshot_correction_prompt(data_item, examples, given_context=None, n_examples=None):
    """
    You can try different prompt in here.
    """
    question_item = data_item


    prompt_text = f"{correction_instruction}\n"

    prompt_text += "Here is the Schema\n"
    prompt_text += f"{conversion(pure_table_prompt)}\n"

    max_n_examples = len(examples)
    if n_examples is not None:
        max_n_examples = n_examples

    prompt_text += "Here are some examples:\n"

        
    if max_n_examples > 0:
        for example_id, example in enumerate(examples[-max_n_examples:]):
            prompt_text += f"Example #{example_id + 1}\n"

            # remove multiple choice in last slot values
            last_slot_values = {s: v.split(
                '|')[0] for s, v in example['last_slot_values'].items()}
            prompt_text += f"[context] {conversion(', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()}))}\n"

            last_sys_utt = example['dialog']['sys'][-1]
            if last_sys_utt == 'none':
                last_sys_utt = ''
            prompt_text += f"[human agent] {last_sys_utt}\n"
            prompt_text += f"[user] {example['dialog']['usr'][-1]}\n"

            prompt_text += f"Predicted SQL: {conversion(slot_values_to_seq_sql(example['pred_turn_change']))};\n"
            prompt_text += f"Correct SQL: {conversion(slot_values_to_seq_sql(example['turn_slot_values']))};\n"
         
            prompt_text += "\n\n"


    prompt_text += """Produce the Correct SQL for this new example. Output only the SQLite command.\n\n"""

    if given_context is None:
        last_slot_values = {s: v.split(
            '|')[0] for s, v in question_item['last_slot_values'].items()}
    else:
        last_slot_values = given_context
    prompt_text += f"[context] {conversion(', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()}))}\n"

    last_sys_utt = question_item['dialog']['sys'][-1]
    if last_sys_utt == 'none':
        last_sys_utt = ''
    prompt_text += f"[human agent] {last_sys_utt}\n"
    prompt_text += f"[user] {question_item['dialog']['usr'][-1]}\n"
    prompt_text += f"Predicted SQL: {conversion(slot_values_to_seq_sql(question_item['pred_turn_change']))};\n"
    prompt_text += "Correct SQL: SELECT * FROM"

    return prompt_text




def get_correctionlm_prompt(data_item, examples, given_context=None, n_examples=None):
    """
    You can try different prompt in here.
    """
    question_item = data_item

    # for MWOZ, we omit the instruction and schema table to save tokens for examples
    prompt_text = f"{conversion(pure_table_prompt)}\n"

    max_n_examples = len(examples)
    if n_examples is not None:
        max_n_examples = n_examples

    prompt_text += "Here are some examples:\n"

    for example_id, example in enumerate(examples[-(max_n_examples-1):]):
        prompt_text += f"Example #{example_id + 1}\n"

        # remove multiple choice in last slot values
        last_slot_values = {s: v.split(
            '|')[0] for s, v in example['last_slot_values'].items()}
        prompt_text += f"[context] {conversion(', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()}))}\n"

        last_sys_utt = example['dialog']['sys'][-1]
        if last_sys_utt == 'none':
            last_sys_utt = ''
        prompt_text += f"[human agent] {last_sys_utt}\n"
        prompt_text += f"[user] {example['dialog']['usr'][-1]}\n"

        prompt_text += f"Predicted SQL: {conversion(slot_values_to_seq_sql(example['pred_turn_change']))};\n"
        prompt_text += f"Correct SQL: {conversion(slot_values_to_seq_sql(example['turn_slot_values']))};\n"
        prompt_text += "\n\n"

    prompt_text += """Produce the Correct SQL for this new example. Output only the SQLite command.\n\n"""

    last_slot_values = {s: v.split(
        '|')[0] for s, v in question_item['last_slot_values'].items()}
    
    prompt_text += f"[context] {conversion(', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()}))}\n"

    last_sys_utt = question_item['dialog']['sys'][-1]
    if last_sys_utt == 'none':
        last_sys_utt = ''
    prompt_text += f"[human agent] {last_sys_utt}\n"
    prompt_text += f"[user] {question_item['dialog']['usr'][-1]}\n"
    prompt_text += f"Predicted SQL: {conversion(slot_values_to_seq_sql(question_item['pred_turn_change']))};\n"
    completion = f"Correct SQL: {conversion(slot_values_to_seq_sql(question_item['turn_slot_values']))};\n"

    return prompt_text, completion




def get_llama_SFT_prompt(data_item, examples, given_context=None, n_examples=None):
    """
    # use training set as inference target to sort the in-context exemplars for training! 
    """
    question_item = examples[-2] # exclude the selected inference target 

    prompt_text = f"{conversion(pure_table_prompt)}\n"

    max_n_examples = len(examples)
    if n_examples is not None:
        max_n_examples = n_examples

    prompt_text += "Here are some examples:\n"

    # in case for zero-shot learning
    if max_n_examples > 0:
        for example_id, example in enumerate(examples[-max_n_examples:-2]):
            prompt_text += f"Example #{example_id + 1}\n"

            # remove multiple choice in last slot values
            last_slot_values = {s: v.split(
                '|')[0] for s, v in example['last_slot_values'].items()}
            prompt_text += f"[context] {conversion(', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()}))}\n"

            last_sys_utt = example['dialog']['sys'][-1]
            if last_sys_utt == 'none':
                last_sys_utt = ''
            prompt_text += f"[human agent] {last_sys_utt}\n"
            prompt_text += f"[user] {example['dialog']['usr'][-1]}\n"

            prompt_text += f"Predicted SQL: {conversion(slot_values_to_seq_sql(example['pred_turn_change']))};\n"
            prompt_text += f"Correct SQL: {conversion(slot_values_to_seq_sql(example['turn_slot_values']))};\n"
         
            prompt_text += "\n\n"


    prompt_text += """Produce the Correct SQL for this new example. Output only the SQLite command.\n\n"""


    last_slot_values = {s: v.split(
        '|')[0] for s, v in question_item['last_slot_values'].items()}
   

    prompt_text += f"[context] {', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()})}\n"

    last_sys_utt = question_item['dialog']['sys'][-1]
    if last_sys_utt == 'none':
        last_sys_utt = ''
    prompt_text += f"[human agent] {last_sys_utt}\n"
    prompt_text += f"[user] {question_item['dialog']['usr'][-1]}\n"
    prompt_text += f"Predicted SQL: {conversion(slot_values_to_seq_sql(question_item['pred_turn_change']))};\n"
    completion = f"Correct SQL: {conversion(slot_values_to_seq_sql(question_item['turn_slot_values']))};\n"


    return prompt_text, completion

