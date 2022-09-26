import jax.numpy as jnp
import sentencepiece as spm
from tqdm import tqdm
from argparse import ArgumentParser

def pad(x, length, pad_token):
    return jnp.pad(x, (0, length - len(x)), 'constant', constant_values=pad_token) 

def get_sp(model_file):
    sp = spm.SentencePieceProcessor(model_file=model_file)
    return sp

def encode(sp, f):
    return map(lambda x: sp.EncodeAsIds(x, add_bos=True, add_eos=False), f)

def read_data(src_path, target_path, sp: spm.SentencePieceProcessor, max_length=100):
    f_src = open(src_path, 'r') 
    f_tgt = open(target_path, 'r')
    num_lines = sum(1 for _ in open(src_path, 'r'))
    xy = filter(lambda x: len(x[0]) <= 100 and len(x[1]) <= max_length, zip(encode(sp, f_src), encode(sp, f_tgt)))

    pad_id = sp.pad_id()
    
    xy_bins = [([], []) for _ in range(10)]
    print('Padding...')
    for pair in tqdm(xy, total=num_lines):
        for b in range(10, 101, 10):
            if len(pair[0]) <= b and len(pair[1]) < b:
                xy_bins[b//10 - 1][0].append(pad(jnp.array(pair[0], copy=False), b, pad_id))
                xy_bins[b//10 - 1][1].append(pad(jnp.array(pair[1], copy=False), b, pad_id)) 
                break

    f_src.close()
    f_tgt.close()
            
    return xy_bins
        

if __name__ == '__main__':
    parser = ArgumentParser('Preprocess data')
    parser.add_argument('--src_path', type=str, required=True, help='Path to source file')
    parser.add_argument('--tgt_path', type=str, required=True, help='Path to target file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to sentencepiece model file')
    args = parser.parse_args()
    
    sp = get_sp(args.model_path)
    xy_bins = read_data(args.src_path, args.tgt_path, sp)
    
    print('Saving...')
    for i in tqdm(range(len(xy_bins))):
        jnp.save(f'{args.src_path}_{(i+1)*10}.npy', xy_bins[i][0])
        jnp.save(f'{args.tgt_path}_{(i+1)*10}.npy', xy_bins[i][1])
    
    # jnp.save(f'{args.src_path}.npy', src)
    # jnp.save(f'{args.tgt_path}.npy', tgt)