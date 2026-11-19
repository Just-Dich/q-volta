import sys, json
from tokenizers import Tokenizer
t = Tokenizer.from_file("/workspace/q-volta-models/pi-adapter/tokenizer.json")
print(t.decode(json.loads(sys.stdin.read().strip().splitlines()[-1])["ids"], skip_special_tokens=False))
