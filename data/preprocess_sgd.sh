git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue

# sample 5% from training set
python preprocess_sgd.py --schema_path sgd/ontology/schema_prefix.json --dialogue_path "dstc8-schema-guided-dialogue/train/dialogues_*.json" --output_path data/sgd/sgd_train_5p.json --percentage 0.05

# sample 100% from testing set
python preprocess_sgd.py --schema_path sgd/ontology/schema_prefix.json --dialogue_path "dstc8-schema-guided-dialogue/test/dialogues_*.json" --output_path data/sgd/sgd_test_100p.json --percentage 1