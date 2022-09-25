import jax.numpy as jnp
import sentencepiece as spm
from tqdm import tqdm
from argparse import ArgumentParser

def pad(x, length, pad_token):
    return jnp.pad(x, (0, length - len(x)), 'constant', constant_values=pad_token) 

def get_sp(model_file):
    sp = spm.SentencePieceProcessor(model_file=model_file)
    return sp

def read_data(src_path, target_path, sp: spm.SentencePieceProcessor, max_length=100):
    with open(src_path, 'r') as f:
        src = f.read().splitlines()
        src = sp.EncodeAsIds(src, add_bos=True, add_eos=True)
    
    with open(target_path, 'r') as f:
        target = f.read().splitlines()
        target = sp.EncodeAsIds(target, add_bos=True, add_eos=True)
        
    xy = list(filter(lambda x: len(x[0]) <= 100 and len(x[1]) <= 100, zip(src, target)))
    del src, target
    
    for i in tqdm(range(len(xy))):
        xy[i] = (pad(jnp.array(xy[i][0], copy=False), 100, sp.pad_id()), pad(jnp.array(xy[i][1], copy=False), 100, sp.pad_id()))
        
    x, y = zip(*xy)
    del xy
    x, y = jnp.vstack(x), jnp.vstack(y)
    return x, y
        



if __name__ == '__main__':
    parser = ArgumentParser('Preprocess data')
    parser.add_argument('--src_path', type=str, required=True, help='Path to source file')
    parser.add_argument('--tgt_path', type=str, required=True, help='Path to target file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to sentencepiece model file')
    args = parser.parse_args()
    
    sp = get_sp(args.model_path)
    src, tgt = read_data(args.src_path, args.tgt_path, sp)
    
    jnp.save(f'{args.src_path}.npy', src)
    jnp.save(f'{args.tgt_path}.npy', tgt)