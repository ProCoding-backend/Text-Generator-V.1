# Text-Generator-V.1 Without Rules
## Tiny Transformer Text Generator (no hand-written rules)
-------------------------------------------------------
####  This script trains a very small Transformer language model on your own text
#### and then generates new text. The model *learns from data* end‑to‑end—no
#### if/else expert rules. It's a minimal, transparent example of "building your
#### own AI" with gradient descent.

###### Quick start
-----------
###### 1) Put some plain text into a file named `data.txt` in the same folder.
   More data => better results. A few hundred KB is a nice start.

###### 2) Install deps (Python 3.9+ recommended):
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   # Or your CUDA wheel if you have a GPU

###### 3) Train for a few minutes:
   python tiny_transformer_textgen.py --epochs 5 --device auto

###### 4) Generate text:
  python tiny_transformer_textgen.py --gen-only --generate "Once upon a time" --max-new-tokens 200


###### Notes
-----
- This is purposely tiny to keep it readable. Scale dims/layers and train
  longer for better quality.
- Safe to tinker: change `d_model`, `n_heads`, `n_layers`, `block_size` etc.
- You can swap the dataset to *anything texty*: code, chats, poetry, logs…

###### License: MIT

Takes about 5-7 minutes to be trained completely from data.txt

Try creating your own data file

###### If hard, run the create_data.py script
###### It will auto generate files for you

## Made by ❤️ Sohan Shaw ❤️, Pro-Coding Backend 💕😊

### Give Proper Credits if uploaded on a social media platform ©️ Sohan Shaw

