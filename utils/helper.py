# from https://github.com/Yushi-Hu/IC-DST/tree/main/utils
import time

# time to restrict query speed
class SpeedLimitTimer:
    def __init__(self, second_per_step=3.1):
        self.record_time = time.time()
        self.second_per_step = second_per_step

    def step(self):
        time_div = time.time() - self.record_time
        if time_div <= self.second_per_step:
            time.sleep(self.second_per_step - time_div)
        self.record_time = time.time()

    def sleep(self, s):
        time.sleep(s)


class PreviousStateRecorder:

    def __init__(self):
        self.states = {}

    def add_state(self, data_item, slot_values):
        dialog_ID = data_item['ID']
        turn_id = data_item['turn_id']
        if dialog_ID not in self.states:
            self.states[dialog_ID] = {}
        self.states[dialog_ID][turn_id] = slot_values

    def state_retrieval(self, data_item):
        dialog_ID = data_item['ID']
        turn_id = data_item['turn_id']
        if turn_id == 0:
            return {}
        else:
            return self.states[dialog_ID][turn_id - 1]

def clean_sql_completion(completion):
    if completion.endswith(";"):
        completion = completion[:-1]

    unwanted_phrases = [
        "```sql", "```SQL", "SELECT * FROM", "SELECT", "```", "SQL:", "\n"
    ]

    for phrase in unwanted_phrases:
        completion = completion.replace(phrase, "").strip()

    if "FROM" in completion:
        completion = completion.split("FROM")[1].strip()

    return completion