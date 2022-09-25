from model import gen_transformer, gen_loss
from jax import jit, grad, vmap, nn
import jax.numpy as jnp
import jax
from tqdm import tqdm
import optax
from argparse import ArgumentParser

def load_data(src_path, target_path):
    x = jnp.load(src_path)
    y = jnp.load(target_path)
    return x, y

def prepare_data(x, y, pad_token, vocab_size):
    x_mask = get_pad_mask(x, pad_token)
    y_mask = get_pad_mask(y, pad_token)
    
    x = nn.one_hot(x, vocab_size)
    y = nn.one_hot(y, vocab_size)
    
    return x, y, x_mask, y_mask

def get_batches(x, y, batch_size, pad_token, vocab_size, shuffle=True, key=None):
    if shuffle:
        idx = jax.random.permutation(key, len(x))
        x, y = x[idx], y[idx]
    for i in range(0, len(x), batch_size):
        yield prepare_data(x[i:i+batch_size], y[i:i+batch_size], pad_token, vocab_size)
        
def get_pad_mask(x: jnp.ndarray, pad_token):
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
        'batch_size': args.batch_size,
        'epochs': 10,
        'eps_label_smoothing': 0.1,
        'learning_rate': 0.0001,
    }
    
    key = jax.random.PRNGKey(42)
    
    transformer, gen_params = gen_transformer(config)
    transformer = vmap(transformer, in_axes=(None, 0, 0, 0, 0, None, None), out_axes=0)
    transformer = jit(transformer, static_argnames='is_training')
    
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
                              batch_size=config['batch_size'], 
                              pad_token=-1, 
                              vocab_size=config['vocab_size'], 
                              shuffle=False, key=subkey)
        
        for x, y, x_mask, y_mask in (pbar := tqdm(batches, total=len(en)//config['batch_size'], unit_scale=config['batch_size'])):
            key, subkey = jax.random.split(key)
            # print(transformer(params, x, y, x_mask, y_mask, True, subkey))
            loss, grads = loss_and_grad(params, transformer, x, y, x_mask, y_mask, True, subkey)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            pbar.set_description(f'Loss: {loss:.4f}')
            
        
    
    
if __name__ == '__main__':
    parser = ArgumentParser('Train Transformer')
    parser.add_argument('--src_path', type=str, default='./data/train.en.npy')
    parser.add_argument('--tgt_path', type=str, default='./data/train.de.npy')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    main(args)