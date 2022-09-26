from model import gen_transformer, gen_loss
from jax import jit, grad, vmap, nn
import jax.numpy as jnp
import jax
from tqdm import tqdm
import optax
from argparse import ArgumentParser

def load_data(src_path, target_path):
    x = [jnp.load(f"{src_path}_{i}.npy") for i in range(10, 101, 10)]
    y = [jnp.load(f"{target_path}_{i}.npy") for i in range(10, 101, 10)]
    return x, y

def prepare_data(x, y, pad_token, vocab_size):
    x_mask = get_pad_mask(x, pad_token)
    y_mask = get_pad_mask(y, pad_token)
    
    # x = nn.one_hot(x, vocab_size)
    # y = nn.one_hot(y, vocab_size)
    # shape (batch_size, seq_len, vocab_size)
    
    return x, y, x_mask, y_mask

def get_batches(x, y, num_tokens, pad_token, vocab_size, shuffle=True, key=None):
    for x_, y_ in zip(x, y):
        # x_, y_ are sets of same length padded sentences
        if shuffle:
            key, subkey = jax.random.split(key, 2)
            idx = jax.random.permutation(subkey, len(x_))
            x_, y_ = x_[idx], y_[idx]
            
        batch_size = num_tokens // x_.shape[-1]
        for i in range(0, len(x_), batch_size):
            yield prepare_data(x_[i:i+batch_size], y_[i:i+batch_size], pad_token, vocab_size)

@jax.jit
def get_pad_mask(x: jnp.ndarray, pad_token: int) -> jnp.ndarray:
    """
    Create pad mask for x.
    
    Args:
        x (jnp.ndarray): Input tensor of shape (batch_size, seq_len)
    """
    partial = -1e9 * (x == pad_token).astype(jnp.float32)
    # shape = (batch_size, seq_len)
    out = jnp.tile(partial[:, None, :], (1, partial.shape[1], 1))
    # shape = (batch_size, seq_len, seq_len)
    return out

# def loss_fn(params, x, y, x_mask, y_mask, is_training, key):
#     y_pred = transformer(x, y, x_mask, y_mask, params, is_training, key)
#     loss = softmax_cross_entropy(logits=y_pred, labels=y)
#     return loss.mean()
    
def main(args):
    
    config = {
        'd_model': 512,
        'dk': 64,
        'dv': 64,
        'dff': 2048,
        'enc_layers': 6,
        'dec_layers': 6,
        'heads': 8,
        'dropout_rate': 0.1,
        'vocab_size': 37000,
        'num_tokens': args.num_tokens, # in number of tokens (approx)
        'epochs': 10,
        'eps_label_smoothing': 0.1,
        'learning_rate': 0.0001,
    }
    
    key = jax.random.PRNGKey(42)
    
    transformer, gen_params = gen_transformer(config)
    # transformer = vmap(transformer, in_axes=(None, 0, 0, 0, 0, None, None), out_axes=0)
    # transformer = jit(transformer, static_argnames='is_training')
    
    loss = gen_loss(config)
    
    loss_and_grad = jax.value_and_grad(loss)
    
    key, subkey = jax.random.split(key)
    
    params = gen_params(subkey)
    
    optimizer = optax.adam(config['learning_rate'])
    opt_state = optimizer.init(params)
    
    en, de = load_data(args.src_path, args.tgt_path)
    
    for epoch in range(config['epochs']):
        key, subkey = jax.random.split(key)
        batches = get_batches(en, 
                              de, 
                              num_tokens=config['num_tokens'], 
                              pad_token=-1, 
                              vocab_size=config['vocab_size'], 
                              shuffle=False, key=subkey)
        
        total_sentences = sum(map(len, en))
        with tqdm(total=total_sentences) as pbar:
            for x, y, x_mask, y_mask in batches:
                key, subkey = jax.random.split(key)
                # print(transformer(params, x, y, x_mask, y_mask, True, subkey))
                l, grads = loss_and_grad(params, transformer, x, y, x_mask, y_mask, True, subkey)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                pbar.update(x.shape[0])
                pbar.set_description(f'Loss: {l:.4f}')
            
        
if __name__ == '__main__':
    parser = ArgumentParser('Train Transformer')
    parser.add_argument('--src_path', type=str, default='./data/dev.en')
    parser.add_argument('--tgt_path', type=str, default='./data/dev.de')
    parser.add_argument('--num_tokens', type=int, default=5000)
    args = parser.parse_args()
    main(args)