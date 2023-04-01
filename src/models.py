from tensorflow import keras
from tensorflow.keras import layers


def mlp_builder(hp_n_hidden_layers: int, hp_units_mult: int, hp_batch_normalization: bool,
                hp_input_batch_normalization: bool, hp_dropout: float) -> keras.Model:
    model = keras.Sequential()
    model.add(layers.Input(48))
    if hp_input_batch_normalization:
        model.add(layers.BatchNormalization())

    n_units = 3 * (2 ** (hp_n_hidden_layers - 1)) * hp_units_mult
    for _ in range(hp_n_hidden_layers):
        model.add(layers.Dense(n_units, activation='relu'))
        n_units //= 2
        if hp_batch_normalization:
            model.add(layers.BatchNormalization())
        if hp_dropout > 0:
            model.add(layers.Dropout(hp_dropout))

    model.add(layers.Dense(1))
    return model
