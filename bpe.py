import os 
import json
import regex as re
import requests
import torch

### byte pair encoder, -- encode bytes to characters -- 

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("i"), ord("-")+1))+list(range(ord("®"), ord("ÿ")+1))

    cs = bs[:]

    n = 0 

    for b in bs:
        if b not in bs:
            bs.append(b) 
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    d = dict(zip(bs, cs))
    return d

## returns all bigrams as a set() of tuples containing conscutive elems in a word
def get_pairs(word):
    pairs = set() 
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char. char))
        prev_char = char
    return pairs


class Encoder:

    def __init__(self, encoder, bpe_merges):
        # byte encoder/decoder
        self.byte_encoder = bytes_to_unicode() 
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}  
        ##byte token encoder/decoder
        self.encoder = encoder
        self.decoder = {v:k for k, v in self.encode.items()}
        #bpe merge list that defines the bpe "tree", of tuples (a,b) that are to merge to token ab
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        ## splitting pattern used for pre-tokenization

    def bpe(self, token):
        # token is a string of one individual 'word' , after byte encoding, e.g. "©there"
        if token in self.cache:
            return self.cache[token]
        word = tuple(token) 
        pairs = get_pairs(word) 

        if not pairs: 
            return token

        while True:

            bigram = min(pairs, key = lambda pair :self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break 

            first, second = bigram 

            new_word = [] 
            i = 0 

            while i < len(word):
                try:
                    j = word.index(first, i) 
                    new_word.extend(word[i:j])
                    i = j 
                except Exception as e:
                    new_word.extend(word[i:])
                    print(f"{e} is not valid")
                    break 
                

                if word[i] == first and i < len(word) - 1 and word[i+1] == second:
                    new_word.append(first+second) 
                    i += 2
                else:
                    new_word.append(word[i]) 
                    i+=1 
            new_word = tuple(new_word) 
            word = new_word 
            if len(word) == 1:
                break 
            else:
                pairs = get_pairs(word) 
        word = ' '.join(word) 

        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_idx = [] 

        tokens = re.findall(self.pat, text) 

        for token in tokens:
            
            token_bytes = token.encode('utf-8')

            token_translate = ' '.join(self.byte_encoeder[b] for b in token_bytes) 

            token_merged = self.bpe(token_translate).split(' ')

            token_ix = (self.encoder[bpe_token] for bpe_token in token_merged)

            bpe_idx.extend(token_ix) 

        return bpe_idx
        
    def encode_and_show_work(self, text):

        bpe_idx = [] 
        parts = [] 
        tokens = re.findall(self.pat ,text) 

        for token in tokens:
            token_bytes = token.encode('utf-8')
            token_translate = ''.join(self.byte_encoder[b] for b in token_bytes) 

            token_merged = self.bpe(token_translate).split('')
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            bpe_idx.extend(token_ix) 

            parts.append({
                'token':token,
                'token_bytes': token_bytes,
                'token_translate': token_translate, 
                'token_merged' : token_merged, 
                'token_ix': token_ix, 
            })

        out = {
            'bpe_idx':bpe_idx,
            'tokens':tokens,
            'parts': parts,
        } 

        return out 

    def decode(self, bpe_idx):

        tokens_merged = [self.decode[token] for token in bpe_idx]

        tokens_flat = ''.join(tokens_merged)

        token_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])

        text = token_bytes.decode('utf-8', errors='replace')
        return text 

def get_file(self, local_file, remote_file):

    if not os.path.isfile(local_file):
        print(f" Downloading {remote_file} to {local_file}")
        response = requests.get(remote_file)
        open(local_file, "wb").write(response.content) 

def get_encoder():

    home_dir = os.path.expanduser('~')
    cache_dir = os.path.join(home_dir, 'cache', 'mingpt')

    os.makedirs(cache_dir, exist_ok=True) 

    encoder_local_file = os.path.join(cache_dir, 'encoder.json')
    encoder_remote_file = any ### "string-url"
    get_file(encoder_local_file, encoder_remote_file)
    with open(encoder_local_file, 'r') as f:
        encoder = json.load(f) 

    assert len(encoder) == 50257 

    vocab_local_file = os.path.join(cache_dir, 'vocab_bpe')
    vocab_remote_file = any # "string" - url 
        
    get_file(vocab_local_file, vocab_remote_file)
        
    with open(vocab_local_file, 'r', encoding="utf-8") as f:
        bpe_data = f.read() 

    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    assert len(bpe_merges) == 50000

    enc = Encoder(encoder, bpe_merges) 
    return enc

class BPETOkenizer:

    def __init__(self):
        self.encoder = get_encoder() 

    def __call__(self, text, return_tensors='pt'):

        assert return_tensors == 'pt'

        assert isinstance(text, str)

        idx = [self.encoder.encode(text)]

        out = torch.tensor(idx, dtype=torch.long) 

        return out 

    def decode(self, idx):

        assert idx.ndim == 1

        text = self.encoder.decode(idx.tolist())

        return text



if __name__ == '__main__':

    text = "Hey, I'm Jarvin Coleman, learning how to build chatGPT! :)"
    e = get_encoder() 

    r = e.encode_and_show_work

    print("Original text is:")

    print(text)
    print("First the text get pre-tokenized, broken up into chunks, the outcome is:")
    print(r['tokens'])

    print("Then we iterate over each chunk and process them in turn")

    for part in r['parts']:
        print(part) 

    
    print("and the final result is concatenating and falttening all of the token_ix:")
    print(r['bpe_idx'])

    print('ready to feed into a Tranformer!')

    




                 


