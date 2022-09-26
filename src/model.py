from jax import numpy as jnp
from jax import nn
from jax import jit, vmap
from jax.tree_util import Partial
import jax
from utils import custom_put_along_axis
import optax

def self_partial(*args):
    return lambda x: Partial(x, *args)

def gen_add_positional_encoding(config: dict):
    d_model = config['d_model']
    
    @jit
    @self_partial(d_model)
    def add_positional_encoding(d_model, x):
        positions = jnp.arange(x.shape[0])[:, None]
        enc = jnp.zeros((positions.shape[0], d_model), dtype=jnp.float32)
        enc = enc.at[:, 0::2].set(jnp.sin(positions / (10000 ** (2 * jnp.arange(0, d_model, 2) / d_model).reshape(1, -1))))
        enc = enc.at[:, 1::2].set(jnp.cos(positions / (10000 ** (2 * jnp.arange(1, d_model, 2) / d_model).reshape(1, -1))))
        return x + enc
   
    return add_positional_encoding


def gen_attention_func(d_model: int):
    
    def attention(q, k, v, mask=None):
        """Calculate attention.

        Args:
            q (jax.numpy.ndarray): query, shape = (input, dk)
            k (jax.numpy.ndarray): key, shape = (input, dk)
            v (jax.numpy.ndarray): value, shape = (input, dv)
            mask (jax.numpy.ndarray, optional): mask. Defaults to None, shape = (input, input)

        Returns:
            jax.numpy.ndarray: attention
        """
        
        num = jnp.matmul(q, k.T) 
        # shape = (input, input)
        scaled = num / jnp.sqrt(d_model) 
        # shape = (input, input)
        if mask is not None:
            scaled = scaled + mask 
            # shape = (input, input)
        softmax_out = nn.softmax(scaled, axis=-1)
        # shape = (input, input)
        attention = jnp.matmul(softmax_out, v)
        # shape = (input, dv)
        return attention
    
    return attention

def gen_multihead_attention_func(config: dict):
    """Generate multihead attention function.

    Args:
        config (dict): configuration

    Returns:
        function: multihead attention function
    """
    d_model = config['d_model']
    heads = config['heads']
    dk = config['dk']
    dv = config['dv']
    
    @jit
    @self_partial(d_model, heads, dk, dv)
    def gen_params(d_model, heads, dk, dv, key):
        keys = jax.random.split(key, 4)
        initializer = nn.initializers.glorot_uniform()
        Wq = initializer(keys[0], (heads, d_model, dk), jnp.float32)
        Wk = initializer(keys[1], (heads, d_model, dk), jnp.float32)
        Wv = initializer(keys[2], (heads, d_model, dv), jnp.float32)
        Wo = initializer(keys[3], (heads * dv, d_model), jnp.float32)
        params = {'Wq': Wq, 'Wk': Wk, 'Wv': Wv, 'Wo': Wo}
        return params
    
    @jit
    @self_partial(d_model)
    def multihead_attention(d_model, params, q, k, v, mask=None):
        """Calculate multihead attention.

        Args:
            q (jax.numpy.ndarray): query, shape = (input, d_model)
            k (jax.numpy.ndarray): key, shape = (input, d_model)
            v (jax.numpy.ndarray): value, shape = (input, d_model)
            params (dict): parameters
                Wq (jax.numpy.ndarray): weight matrix for query, shape = (heads, d_model, dk)
                Wk (jax.numpy.ndarray): weight matrix for key, shape = (heads, d_model, dk)
                Wv (jax.numpy.ndarray): weight matrix for value, shape = (heads, d_model, dv))
                Wo (jax.numpy.ndarray): weight matrix for output, shape = (heads * dv, d_model)
            mask (jax.numpy.ndarray, optional): mask. Defaults to None, shape = (input, input)

        Returns:
            jax.numpy.ndarray: multihead attention
        """
        Wq, Wk, Wv, Wo = params['Wq'], params['Wk'], params['Wv'], params['Wo']
        
        # Project to multiheaded queries, keys, and values
        q = jnp.matmul(q, Wq)
        # shape = (heads, input, dk)
        k = jnp.matmul(k, Wk)
        # shape = (heads, input, dk)
        v = jnp.matmul(v, Wv)
        # shape = (heads, input, dv)
        
        # Compute attention 
        num = jnp.matmul(q, jnp.transpose(k, (0, 2, 1))) 
        # shape = (heads, input, input)
        scaled = num / jnp.sqrt(d_model) 
        # shape = (heads, input, input)
        if mask is not None:
            scaled = scaled + mask[None, :, :] 
            # shape = (heads, input, input)
        softmax_out = nn.softmax(scaled, axis=-1)
        # shape = (heads, input, input)
        multihead_attention = jnp.matmul(softmax_out, v)
        # shape = (heads, input, dv)
        
        concatenated = jnp.concatenate(multihead_attention, axis=1)
        out = jnp.matmul(concatenated, Wo)
        
        return out
    
    return multihead_attention, gen_params

def gen_feedforward(config: dict):
    """Generate feedforward function.

    Args:
        d_model (int): dimension of model
        dff (int): dimension of feedforward

    Returns:
        function: feedforward function
    """
    d_model = config['d_model']
    dff = config['dff']
    
    @jit
    @self_partial(d_model, dff)
    def gen_params(d_model, dff, key):
        keys = jax.random.split(key, 2)
        initializer = nn.initializers.glorot_uniform()
        W1 = initializer(keys[0], (d_model, dff), jnp.float32)
        W2 = initializer(keys[1], (dff, d_model), jnp.float32)
        b1 = jnp.zeros((dff,))
        b2 = jnp.zeros((d_model,))
        params = {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}
        return params
    
    @jit
    def feedforward(params, x):
        """Feedforward.
        
        Args:
            x (jax.numpy.ndarray): input, shape = (input, d_model)
            params (dict): parameters
                W1 (jax.numpy.ndarray): weight matrix for first layer, shape = (d_model, dff)
                b1 (jax.numpy.ndarray): bias for first layer, shape = (dff,)
                W2 (jax.numpy.ndarray): weight matrix for second layer, shape = (dff, d_model)
                b2 (jax.numpy.ndarray): bias for second layer, shape = (d_model,)
            
        Returns:
            jax.numpy.ndarray: output, shape = (input, d_model)
            
        """
        W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
        return jnp.matmul(nn.relu(jnp.matmul(x, W1) + b1), W2) + b2
    
    return feedforward, gen_params

def gen_layer_norm(config: dict):
    d_model = config['d_model']
    
    @jit
    @self_partial(d_model)
    def gen_params(d_model, key):
        keys = jax.random.split(key, 2) 
        initializer = nn.initializers.glorot_uniform()
        gamma = initializer(keys[0], (1, d_model), jnp.float32)
        beta = initializer(keys[1], (1, d_model), jnp.float32)
        params = {'gamma': gamma, 'beta': beta}
        return params
    
    @jit
    def layer_norm(params, x, eps=1e-6):
        """
        Layer Normalization.
        
        Args:
            x (jax.numpy.ndarray): input, shape = (input, d_model)
            params (dict): parameters
                gamma (jax.numpy.ndarray): scale, shape = (d_model,)
                beta (jax.numpy.ndarray): shift, shape = (d_model,)
            eps (float, optional): epsilon. Defaults to 1e-6.
            
        Returns:
            jnp.ndarray: output, shape = (input, d_model)
        """
        gamma, beta = params['gamma'], params['beta']
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / jnp.sqrt(var + eps) + beta
    
    return layer_norm, gen_params

def gen_dropout(config: dict):
    p = 1 - config['dropout_rate']
    
    @jit
    @self_partial(p)
    def dropout(p:float, x, key:jax.random.PRNGKey):
        """
        Dropout.
        
        Args:
            x (jax.numpy.ndarray): input, shape = (input, d_model)
            params (dict): parameters
            
        Returns:
            jax.numpy.ndarray: output, shape = (input, d_model)
        """
        
        x = x * jax.random.bernoulli(key, p, x.shape)
        
        return x
    
    return dropout

def gen_encoder_layer(config: dict):
    multi_head, gen_mh_params = gen_multihead_attention_func(config)
    layer_norm, gen_norm_params = gen_layer_norm(config)
    ffw, gen_ffw_params = gen_feedforward(config)
    dropout = gen_dropout(config)
    
    @jit
    def gen_params(key):
        keys = jax.random.split(key, 4)
        mh_params = gen_mh_params(keys[0])
        norm1_params = gen_norm_params(keys[1])
        ffw_params = gen_ffw_params(keys[2])
        norm2_params = gen_norm_params(keys[3])
        params = {'mh': mh_params, 'norm1': norm1_params, 'ffw': ffw_params, 'norm2': norm2_params}
        return params
    
    @Partial(jit, static_argnames=('is_training',))
    def encoder_layer(params, x, mask=None, is_training=True, key:jax.random.PRNGKey=None):
        """
        Encoder Layer in Transformer Model.
        
        Args:
            x (jax.numpy.ndarray): input, shape = (input, d_model)
            params (dict): parameters
                attention (dict): parameters for attention
                layer_norm1 (dict): parameters for frist layer normalization
                layer_norm2 (dict): parameters for second layer normalization
                ffw (dict): parameters for feedforward
            mask (jax.numpy.ndarray, optional): mask. Defaults to None, shape = (input, input)
            is_training (bool, optional): whether training. Defaults to True.
            key (jax.random.PRNGKey, optional): key for dropout. Must be given when is_training is True. Defaults to None.
            
        Returns:
            jax.numpy.ndarray: output, shape = (input, d_model)
            
        """
        mh_params, norm1_params, ffw_params, norm2_params = params['mh'], params['norm1'], params['ffw'], params['norm2']
        
        keys = jax.random.split(key, 2) if is_training else None # Split keys for dropout
        
        # Multihead attention
        attention = multi_head(mh_params, x, x, x, mask=mask)
        # shape = (input, d_model)
        if is_training:
            attention = dropout(attention, keys[0])
        out1 = layer_norm(norm1_params, x + attention)
        # shape = (input, d_model)
        
        # Feedforward
        feedforward_out = ffw(ffw_params, out1)
        # shape = (input, d_model)
        if is_training:
            feedforward_out = dropout(feedforward_out, keys[1])
        out2 = layer_norm(norm2_params, out1 + feedforward_out)
        # shape = (input, d_model)
        
        return out2

    return encoder_layer, gen_params

def gen_decoder_layer(config: dict):
     
    multi_head, gen_mh_params = gen_multihead_attention_func(config)
    layer_norm, gen_norm_params = gen_layer_norm(config)
    ffw, gen_ffw_params = gen_feedforward(config)
    dropout = gen_dropout(config)
    
    @jit
    def gen_params(key):
        keys = jax.random.split(key, 6)
        mh1_params = gen_mh_params(keys[0])
        norm1_params = gen_norm_params(keys[1])
        mh2_params = gen_mh_params(keys[2])
        norm2_params = gen_norm_params(keys[3])
        ffw_params = gen_ffw_params(keys[4])
        norm3_params = gen_norm_params(keys[5])
        params = {'mh1': mh1_params, 'norm1': norm1_params, 'mh2': mh2_params, 'norm2': norm2_params, 'ffw': ffw_params, 'norm3': norm3_params}
        return params
    
    @Partial(jit, static_argnames=('is_training',))
    def decoder_layer(params, enc_out, dec_in, src_mask=None, target_mask=None, is_training=True, key:jax.random.PRNGKey=None):
        """
        Decoder Layer in Transformer Model.
        
        Args:
            enc_out (jax.numpy.ndarray): encoder output, shape = (input, d_model)
            dec_in (jax.numpy.ndarray): decoder input, shape = (input, d_model)
            params (dict): parameters
                attention1 (dict): parameters for attention
                attention2 (dict): parameters for attention
                layer_norm1 (dict): parameters for frist layer normalization
                layer_norm2 (dict): parameters for second layer normalization
                layer_norm3 (dict): parameters for third layer normalization
                ffw (dict): parameters for feedforward
            src_mask (jax.numpy.ndarray, optional): source mask (padding). Defaults to None, shape = (input, input)
            target_mask (jax.numpy.ndarray, optional): target mask (padding + future tokens). Defaults to None, shape = (input, input)
            is_training (bool, optional): whether training. Defaults to True.
            key (jax.random.PRNGKey, optional): key for dropout. Must be given when is_training is True. Defaults to None.
            
        Returns:
            jax.numpy.ndarray: output, shape = (input, d_model)
            
        """
        mh1_params, norm1_params, mh2_params, norm2_params, ffw_params, norm3_params = params['mh1'], params['norm1'], params['mh2'], params['norm2'], params['ffw'], params['norm3']
        
        keys = jax.random.split(key, 3) if is_training else None # Split keys for dropout
        
        # Multihead attention
        attention1 = multi_head(mh1_params, dec_in, dec_in, dec_in, mask=target_mask)
        # shape = (input, d_model)
        if is_training:
            attention1 = dropout(attention1, keys[0])
        out1 = layer_norm(norm1_params, dec_in + attention1)
        # shape = (input, d_model)
        
        # Multihead attention
        attention2 = multi_head(mh2_params, out1, enc_out, enc_out, mask=src_mask)
        # shape = (input, d_model)
        if is_training:
            attention2 = dropout(attention2, keys[1])
        out2 = layer_norm(norm2_params, out1 + attention2)
        # shape = (input, d_model)
        
        # Feedforward
        feedforward_out = ffw(ffw_params, out2)
        # shape = (input, d_model)
        if is_training:
            feedforward_out = dropout(feedforward_out, keys[2])
        out3 = layer_norm(norm3_params, out2 + feedforward_out)
        # shape = (input, d_model)
        
        return out3
    
    return decoder_layer, gen_params

def gen_encoder(config):
    encoder_layer, gen_encoder_layer_params = gen_encoder_layer(config)
    
    enc_layers = config['enc_layers']
    
    @jit
    @self_partial(enc_layers)
    def gen_params(enc_layers, key):
        keys = jax.random.split(key, enc_layers)
        params = [gen_encoder_layer_params(keys[i]) for i in range(enc_layers)]
        return params
    
    @Partial(jit, static_argnames=('is_training',))
    @self_partial(enc_layers)
    def encoder(enc_layers, params, x, mask=None, is_training=True, key:jax.random.PRNGKey=None):
        """
        Encoder in Transformer Model.
        
        Args:
            x (jax.numpy.ndarray): input, shape = (input, d_model)
            params (dict): parameters
                layer_i (dict): parameters for layer i
            mask (jax.numpy.ndarray, optional): mask. Defaults to None, shape = (input, input)
            is_training (bool, optional): whether training. Defaults to True.
            key (jax.random.PRNGKey, optional): key for dropout. Must be given when is_training is True. Defaults to None.
            
        Returns:
            jax.numpy.ndarray: output, shape = (input, d_model)
            
        """
        
        keys = jax.random.split(key, enc_layers) if is_training else [None] * enc_layers # Split keys for dropout
        
        for i in range(enc_layers): 
            x = encoder_layer(params[i], x, mask, is_training, keys[i])
        return x
    
    
    return encoder, gen_params

def gen_decoder(config: dict):
    decoder_layer, gen_decoder_layer_params = gen_decoder_layer(config)
    dec_layers = config['dec_layers']
    
    @jit
    @self_partial(dec_layers)
    def gen_params(dec_layers, key):
        keys = jax.random.split(key, dec_layers)
        params = [gen_decoder_layer_params(keys[i]) for i in range(dec_layers)]
        return params
    
    @Partial(jit, static_argnames=('is_training',))
    @self_partial(dec_layers)
    def decoder(dec_layers, params, enc_out, dec_in, src_mask=None, target_mask=None, is_training=True, key:jax.random.PRNGKey=None):
        """
        Decoder in Transformer Model.
        
        Args:
            enc_out (jax.numpy.ndarray): encoder output, shape = (input, d_model)
            dec_in (jax.numpy.ndarray): decoder input, shape = (input, d_model)
            params (dict): parameters
                layer_i (dict): parameters for layer i
            src_mask (jax.numpy.ndarray, optional): source mask (padding). Defaults to None, shape = (input, input)
            target_mask (jax.numpy.ndarray, optional): target mask (padding + future tokens). Defaults to None, shape = (input, input)
            key (jax.random.PRNGKey, optional): key for dropout. Must be given when is_training is True. Defaults to None.
            
        Returns:
            jax.numpy.ndarray: output, shape = (input, d_model)
        """
        keys = jax.random.split(key, dec_layers) if is_training else [None] * dec_layers # Split keys for dropout
        x = dec_in
        for i in range(dec_layers): 
            x = decoder_layer(params[i], enc_out, x, src_mask, target_mask, is_training, keys[i])
        return x 
    
    return decoder, gen_params

def gen_transformer(config):
    encoder, gen_encoder_params = gen_encoder(config)
    decoder, gen_decoder_params = gen_decoder(config)
    dropout = gen_dropout(config)
    add_positional_encoding = gen_add_positional_encoding(config)
    
    d_model = config['d_model']
    vocab_size = config['vocab_size']
    
    @jit
    @self_partial(d_model, vocab_size)
    def gen_params(d_model, vocab_size, key):
        keys = jax.random.split(key, 3)
        encoder_params = gen_encoder_params(keys[0])
        decoder_params = gen_decoder_params(keys[1])
        initializer = nn.initializers.glorot_uniform()
        U = initializer(keys[2], (d_model, vocab_size), jnp.float32)
        params = {'encoder': encoder_params, 'decoder': decoder_params, 'U': U}
        return params
    
    @Partial(jit, static_argnames=('is_training',))
    @Partial(vmap, in_axes=(None, 0, 0, 0, 0, None, None), out_axes=0)
    @self_partial(d_model)
    def transformer(d_model, params, src, target, src_mask, target_mask, is_training=True, key:jax.random.PRNGKey=None):
        """
        Transformer Model.
        
        Args:
            src (jax.numpy.ndarray): source, shape = (input)
            target (jax.numpy.ndarray): target, shape = (input)
            src_mask (jax.numpy.ndarray): source mask, shape = (input, input)
            target_mask (jax.numpy.ndarray): target mask, shape = (input, input)
            params (dict): parameters
                U (jax.numpy.ndarray): embedding matrix, shape = (d_model, vocab_size)
                encoder (dict): parameters for encoder
                decoder (dict): parameters for decoder
            is_training (bool, optional): whether training. Defaults to True.
            key (jax.random.PRNGKey, optional): key for dropout. Must be given when is_training is True. Defaults to None.
        
        Returns:
            jax.numpy.ndarray: output, shape = (input, d_model)
            
        """
        U, encoder_params, decoder_params = params['U'], params['encoder'], params['decoder']
        keys = jax.random.split(key, 4) if is_training else [None] * 4 # Split keys for dropout
        
        src_embeddings = jnp.take(U, src, axis=1).T * jnp.sqrt(d_model)
        # shape = (input, d_model)
        # src_embeddings = jnp.matmul(src, U.T) * jnp.sqrt(d_model) 
        # shape = (input, d_model)
        positioned_src_embeddings = add_positional_encoding(src_embeddings)
        # shape = (input, d_model)
        if is_training:
            positioned_src_embeddings = dropout(positioned_src_embeddings, keys[0])
        enc_out = encoder(encoder_params, positioned_src_embeddings, mask=src_mask, is_training=is_training, key=keys[1])
        # shape = (input, d_model)
        
        # target_embeddings = jnp.matmul(target, U.T) * jnp.sqrt(d_model)
        target_embeddings = jnp.take(U, target, axis=1).T * jnp.sqrt(d_model)
        # shape = (input, d_model)
        positioned_target_embeddings = add_positional_encoding(target_embeddings)
        # shape = (input, d_model)
        if is_training:
            positioned_target_embeddings = dropout(positioned_target_embeddings, keys[2])
        dec_out = decoder(decoder_params, enc_out, positioned_target_embeddings, src_mask=src_mask, target_mask=target_mask, is_training=is_training, key=keys[3])
        # shape = (input, d_model)
        out = jnp.matmul(dec_out, U)
        # shape = (input, vocab_size)
        if not is_training:
            out = nn.softmax(out)
        return out
    
    return transformer, gen_params 


def gen_loss(config, testing=False):
    # @self_partial(eps_label_smoothing)
    # def label_smoothing(eps_label_smoothing, labels):
    #     """
    #     Label smoothing.
        
    #     Args: 
    #         labels (jax.numpy.ndarray): labels, shape = (batch, input, vocab_size)
    #     """
    #     # labels.at[labels == 1].set(1 - eps_label_smoothing)
    #     # labels.at[labels == 0].set(eps_label_smoothing / (labels.shape[-1] - 1))
    #     smoothed_amount = eps_label_smoothing / (labels.shape[-1] - 1)
    #     labels = labels * (1 - eps_label_smoothing - smoothed_amount) + smoothed_amount
    #     return labels
    
    eps_label_smoothing = config['eps_label_smoothing']
    vocab_size = config['vocab_size']
    
    smooth_value = eps_label_smoothing / (vocab_size - 1)
    label_value_mult = (1 - eps_label_smoothing) 
    
    @jit
    @self_partial(vocab_size, smooth_value, label_value_mult)
    def one_hot_compare(vocab_size, smooth_value, label_value_mult, logits, labels):
        labels = nn.one_hot(labels, vocab_size)
        labels = labels * (label_value_mult - smooth_value) + smooth_value
        return optax.softmax_cross_entropy(logits, labels)
    
    @jit
    @self_partial(smooth_value, label_value_mult)
    def label_smoothed_softmax_cross_entropy(smooth_value, label_value_mult, logits, labels):
        """
        Softmax cross entropy. Attempt at memory efficient implementation.
        
        Args:
            logits (jax.numpy.ndarray): logits, shape = (batch, input, vocab_size)
            labels (jax.numpy.ndarray): labels (not one_hot encoded), shape = (batch, input)
        """
        # Equivalent to:
        #         -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
        # Strategy:
        # Multiply all values by smooth_value
        # Multiply at label index by (1 - eps_label_smoothing) / smooth_value
        logits = nn.log_softmax(logits, axis=-1)
        temp = jnp.take_along_axis(logits, labels[:, :, None], axis=2) * label_value_mult
        logits = smooth_value * logits
        logits = custom_put_along_axis(logits, labels[:, :, None], temp, axis=2)
        return -jnp.sum(logits, axis=-1)
    
    @Partial(jit, static_argnames=('is_training', 'model'))
    def loss(params, model, x, y, x_mask, y_mask, is_training=True, key:jax.random.PRNGKey=None):
        """
        Loss function. Note: x and y are not one_hot encoded for memory efficiency.
        
        Args:
            params (dict): parameters for model
            model (function): model
            x (jax.numpy.ndarray): source, shape = (batch, input)
            y (jax.numpy.ndarray): target, shape = (batch, input)
            x_mask (jax.numpy.ndarray): source mask, shape = (batch, input, input)
            y_mask (jax.numpy.ndarray): target mask, shape = (batch, input, input)
            is_training (bool, optional): whether training. Defaults to True.
            key (jax.random.PRNGKey, optional): key for dropout. Must be given when is_training is True. Defaults to None.
        """
        
        y_pred = model(params, x, y, x_mask, y_mask, is_training, key)
        
        return jnp.mean(label_smoothed_softmax_cross_entropy(y_pred, y))
    
    if testing:
        return label_smoothed_softmax_cross_entropy, one_hot_compare
    return loss
        
        
        
        
        
        
        
        
        