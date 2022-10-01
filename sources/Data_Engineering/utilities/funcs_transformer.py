import logging
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential

logger = logging.getLogger(__name__)

class TransformerBlock_Dense(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(TransformerBlock_Dense, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // 2)
        self.ffn = Sequential([
            layers.Dense(embed_dim * 2),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, attn_mask):
        attn_output = self.att(inputs, inputs, attention_mask=attn_mask)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        return config


class TransformerBlock_LSTM(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(TransformerBlock_LSTM, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // 2)
        self.ffn = Sequential([
            layers.LSTM(embed_dim * 2, return_sequences=True),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, attn_mask):
        attn_output = self.att(inputs, inputs, attention_mask=attn_mask)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        return config


class TransformerBlock_BIDLSTM(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(TransformerBlock_BIDLSTM, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // 2)
        self.ffn = Sequential([
            layers.Bidirectional(
                layers.LSTM(embed_dim, return_sequences=True)
            ),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, attn_mask):
        attn_output = self.att(inputs, inputs, attention_mask=attn_mask)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        return config


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        return config


class ExtractAttentionMask(tf.keras.layers.Layer):
    def __init__(self):
        super(ExtractAttentionMask, self).__init__()

    def call(self, x):
        return tf.math.not_equal(x, tf.zeros(shape=tf.shape(x), dtype=tf.int64))

    def get_config(self):
        config = super().get_config()
        return config

# example
# embed = TokenAndPositionEmbedding(MAX_LEN, len(vectorizer.get_vocabulary(True)), embed_dim)
# # layer_getAttn = ExtractAttentionMask()
# # layer_reshape = layers.Reshape((-1,1))
# # transformer_block1 = TransformerBlock(embed_dim, num_heads, 0.125)
# # transformer_block2 = TransformerBlock(embed_dim, num_heads, 0.125)
# avgPooling = layers.GlobalAveragePooling1D()
#
# x = vectorizer(input1)
# # attn_mask = layer_reshape(layer_getAttn(x))
# x = embed(x)
# # x = transformer_block1(x, attn_mask)
# # x = transformer_block2(x, attn_mask)
# input1_embed = avgPooling(x)
#
# x = vectorizer(input2)
# # attn_mask = layer_reshape(layer_getAttn(x))
# x = embed(x)
# # x = transformer_block1(x, attn_mask)
# # x = transformer_block2(x, attn_mask)
# input2_embed = avgPooling(x)