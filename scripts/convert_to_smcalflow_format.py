"""
This script takes files in the jsonlines in our released format for the contextual utterances, combines them with the non-contextual utterances
and converts it to the format of SMCalFlow. 
"""
import argparse
import jsonlines
from tqdm import tqdm

def convert_files(original_smcal_train_file,
                  original_smcal_valid_file,
                  find_event_train_file,
                  find_event_valid_file,
                  find_event_revise_train_file,
                  find_event_revise_valid_file,
                  find_event_revise_train_output,
                  find_event_revise_valid_output):
    dialogue_id_to_turns = {}
    with jsonlines.open(original_smcal_train_file, 'r') as reader:
        for obj in reader:
            dialogue_id_to_turns[obj['dialogue_id']] = obj['turns']

    with jsonlines.open(original_smcal_valid_file, 'r') as reader:
        for obj in reader:
            dialogue_id_to_turns[obj['dialogue_id']] = obj['turns']


    with jsonlines.open(find_event_revise_train_output, 'w') as writer:
        with jsonlines.open(find_event_revise_train_file, 'r') as reader:
            for obj in tqdm(reader):
                if type(obj['dialogue_id']) == str:
                    turn = dialogue_id_to_turns[obj['dialogue_id']][int(obj['turn_part_id'])]
                    turn['lispress'] = obj['plan']
                    turn['fully_typed_lispress'] = obj['typed_plan']
                    writer.write({'dialogue_id': obj['dialogue_id'],
                                'turns': [turn]})
        with jsonlines.open(find_event_train_file, 'r') as reader:
            for obj in tqdm(reader):
                if type(obj['dialogue_id']) == str:
                    turn = dialogue_id_to_turns[obj['dialogue_id']][int(obj['turn_part_id'])]
                    turn['lispress'] = obj['plan']
                    turn['fully_typed_lispress'] = obj['typed_plan']
                    writer.write({'dialogue_id': obj['dialogue_id'],
                                'turns': [turn]})


    with jsonlines.open(find_event_revise_valid_output, 'w') as writer:
        with jsonlines.open(find_event_revise_valid_file, 'r') as reader:
            for obj in tqdm(reader):
                if type(obj['dialogue_id']) == str:
                    turn = dialogue_id_to_turns[obj['dialogue_id']][int(obj['turn_part_id'])]
                    turn['lispress'] = obj['plan']
                    turn['fully_typed_lispress'] = obj['typed_plan']
                    writer.write({'dialogue_id': obj['dialogue_id'],
                                'turns': [turn]})
        with jsonlines.open(find_event_valid_file, 'r') as reader:
            for obj in tqdm(reader):
                if type(obj['dialogue_id']) == str:
                    turn = dialogue_id_to_turns[obj['dialogue_id']][int(obj['turn_part_id'])]
                    turn['lispress'] = obj['plan']
                    turn['fully_typed_lispress'] = obj['typed_plan']
                    writer.write({'dialogue_id': obj['dialogue_id'],
                                'turns': [turn]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_smcal_train_file', type=str, required=True)
    parser.add_argument('--original_smcal_valid_file', type=str, required=True)
    parser.add_argument('--find_event_train_file', type=str, required=True)
    parser.add_argument('--find_event_valid_file', type=str, required=True)
    parser.add_argument('--find_event_revise_train_file', type=str, required=True)
    parser.add_argument('--find_event_revise_valid_file', type=str, required=True)
    parser.add_argument('--find_event_revise_edit_fragment_train_file', type=str, required=True)
    parser.add_argument('--find_event_revise_edit_fragment_valid_file', type=str, required=True)
    parser.add_argument('--find_event_revise_train_output', type=str, required=True)
    parser.add_argument('--find_event_revise_valid_output', type=str, required=True)
    parser.add_argument('--find_event_revise_edit_fragment_train_output', type=str, required=True)
    parser.add_argument('--find_event_revise_edit_fragment_valid_output', type=str, required=True)

    args = parser.parse_args()
    convert_files(args.original_smcal_train_file,
                    args.original_smcal_valid_file,
                    args.find_event_train_file,
                    args.find_event_valid_file,
                    args.find_event_revise_train_file,
                    args.find_event_revise_valid_file,
                    args.find_event_revise_train_output,
                    args.find_event_revise_valid_output)

    convert_files(args.original_smcal_train_file,
                    args.original_smcal_valid_file,
                    args.find_event_train_file,
                    args.find_event_valid_file,
                    args.find_event_revise_edit_fragment_train_file,
                    args.find_event_revise_edit_fragment_valid_file,
                    args.find_event_revise_edit_fragment_train_output,
                    args.find_event_revise_edit_fragment_valid_output)
