# 2. SeqGAN using story data

## Running the tests
1. Create directory `embed` and Download the `ko.tsv` file from the link below to the directory.
    - [Pretrained korean word2vec](https://drive.google.com/open?id=0B0ZXk88koS2KbDhXdWg1Q2RydlU)
    
2. Create directory `data` and Move `train_data.txt` files into the directory.
    - `train_data.txt` files are created by `grimm_split` and `aesop_split`

3. Run the code in the following order:
    1. load_embed.py
    2. preprocess_util.py
    3. preprocess_data.py
    4. sequence_gan.py
    5. sequence_gan_load_test.py (for test with start token)
    
## Reference
- Pretrained word2vec : https://github.com/Kyubyong/wordvectors