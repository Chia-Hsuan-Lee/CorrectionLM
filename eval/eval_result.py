from evaluate_metrics import evaluate
import json
import argparse

 

parser = argparse.ArgumentParser()
parser.add_argument('--eval_fn', type=str, help="file to evaluate on", required=True)  
parser.add_argument('--eval_mode', type=str, help="eval on first pass or corrected second pass", required=True)  
args = parser.parse_args()


n_total = 0
n_correct = 0
n_correct_tlb = 0
total_acc = 0
total_dst_f1 = 0
total_tlb_f1 = 0

items = json.load(open(args.eval_fn))
for item in items: 
    n_total += 1
    if args.eval_mode == "first_pass":
        this_jga, this_acc, dst_f1 = evaluate(item['pred'] , item['slot_values'])
    elif args.eval_mode == "second_pass":
        this_jga, this_acc, dst_f1 = evaluate(item['pred_second_pass'] , item['slot_values'])
    if args.eval_mode == "first_pass":
        this_tlb, _, tlb_f1 = evaluate(item['pred_turn_change'], item['turn_slot_values'])
    elif args.eval_mode == "second_pass":
        this_tlb, _, tlb_f1 = evaluate(item['pred_turn_change_second_pass'], item['turn_slot_values'])

    total_acc += this_acc
    total_dst_f1 += dst_f1
    total_tlb_f1 += tlb_f1

    if this_jga:
        n_correct += 1
    if this_tlb:
        n_correct_tlb += 1
    
print(f"DST JGA {n_correct}/{n_total}  =  {n_correct / n_total}")
print(f"TLB JGA {n_correct_tlb}/{n_total}  =  {n_correct_tlb / n_total}")
print(f"Slot Acc {total_acc/n_total}")
print(f"DST Joint F1 {total_dst_f1/n_total}")
print(f"TLB Joint F1 {total_tlb_f1/n_total}")
print()
