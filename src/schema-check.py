import json 
import os 
from tqdm import tqdm 

SUBMISSION_DIR = "public_submission"
BTC_DIR = "/home/hiraki/Downloads/public_test/lyrics"

submission_files = os.listdir(SUBMISSION_DIR)
btc_files = os.listdir(BTC_DIR)
json_filter = lambda x: x.endswith("json")

submission_files = list(filter(json_filter, submission_files))
btc_files = list(filter(json_filter, btc_files))

len(submission_files), len(btc_files)
set(btc_files) - set(submission_files)

for file in tqdm(submission_files):
    with open(os.path.join(SUBMISSION_DIR, file)) as f:
        submission_data = json.load(f)

    with open(os.path.join(BTC_DIR, file)) as f:
        btc_data = json.load(f)
    
    # Check number of sentences
    assert len(submission_data) == len(btc_data)

    # For each sentence
    for (sent_sub, sent_b) in zip(submission_data, btc_data):
        # Verify schema
        assert sent_sub.keys() == sent_b.keys()
        
        # Equal sentence length
        assert len(sent_sub["l"]) == len(sent_b["l"])

        # Each token in sentences, verify schema and data equality
        for (tok_sub, tok_b) in zip(sent_sub["l"], sent_b["l"]):
            assert tok_b.keys() == tok_sub.keys()
            assert tok_b["d"] == tok_sub["d"]

print("All tests passed")