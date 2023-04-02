from tensorflow import keras
from tensorflow.keras import layers


def mlp_builder(hp_n_hidden_layers: int, hp_units_mult: int, hp_unit_decrease_factor: float,
                hp_batch_normalization: bool, hp_input_batch_normalization: bool, hp_dropout: float) -> keras.Model:
    model = keras.Sequential()
    model.add(layers.Input(48))
    if hp_input_batch_normalization:
        model.add(layers.BatchNormalization())

    n_units_list = [4 * hp_units_mult]
    for _ in range(hp_n_hidden_layers - 1):
        n_units_list.append(int(n_units_list[-1] * hp_unit_decrease_factor))

    for n_units in reversed(n_units_list):
        model.add(layers.Dense(n_units, activation='relu'))
        if hp_batch_normalization:
            model.add(layers.BatchNormalization())
        if hp_dropout > 0:
            model.add(layers.Dropout(hp_dropout))

    model.add(layers.Dense(1))
    return model


def convnet_builder(hp_n_conv_blocks: int, hp_n_conv_layers: int, hp_filters_mult: int, hp_conv_spatial_dropout: float,
                    hp_mlp_n_hidden_layers: int, hp_mlp_units_mult: int, hp_mlp_dropout: float,
                    hp_batch_normalization: bool, hp_input_batch_normalization: bool) -> keras.Model:
    model = keras.Sequential()
    model.add(layers.Input(48))
    if hp_input_batch_normalization:
        model.add(layers.BatchNormalization())

    model.add(layers.Reshape((-1, 1)))

    # Convolutional network
    n_filters = 8 * hp_filters_mult
    for _ in range(hp_n_conv_blocks):  # block
        for _ in range(hp_n_conv_layers):  # layer
            model.add(layers.Conv1D(n_filters, 3, padding='same', activation='relu'))
            if hp_batch_normalization:
                model.add(layers.BatchNormalization())
            if hp_conv_spatial_dropout:
                model.add(layers.SpatialDropout1D(hp_conv_spatial_dropout))

        model.add(layers.MaxPooling1D())
        n_filters *= 2

    model.add(layers.Flatten())

    # MLP at the end
    if hp_mlp_n_hidden_layers > 0:
        n_units = 4 * (2 ** (hp_mlp_n_hidden_layers - 1)) * hp_mlp_units_mult
        for _ in range(hp_mlp_n_hidden_layers):
            model.add(layers.Dense(n_units, activation='relu'))
            n_units //= 2
            if hp_batch_normalization:
                model.add(layers.BatchNormalization())
            if hp_mlp_dropout > 0:
                model.add(layers.Dropout(hp_mlp_dropout))

    model.add(layers.Dense(1))

    return model
