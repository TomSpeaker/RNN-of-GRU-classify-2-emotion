import tensorflow as tf

def build_model(input_dim=10000, output_dim=128, input_length=200):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length),
        tf.keras.layers.GRU(128, return_sequences=False),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # 二分类输出
    ])
    return model
