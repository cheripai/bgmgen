from keras.models import Model
from keras.layers import Input, Dropout, TimeDistributed, Dense
from keras.layers import BatchNormalization, Embedding, Activation, Reshape
from keras.layers.merge import Add
from keras.layers.recurrent import GRU
from keras.regularizers import l2

NUM_NOTES = 128


def build(max_token_length,
          num_image_features=2048,
          hidden_size=512,
          embedding_size=512,
          regularizer=1e-8):

    notes_input = Input(shape=(max_token_length, NUM_NOTES), name="notes")
    notes_to_embedding = TimeDistributed(
        Dense(
            units=embedding_size,
            kernel_regularizer=l2(regularizer),
            name="notes_embedding"))(notes_input)
    notes_dropout = Dropout(0.5, name="notes_dropout")(notes_to_embedding)

    image_input = Input(
        shape=(max_token_length, num_image_features), name="image")
    image_embedding = TimeDistributed(
        Dense(
            units=embedding_size,
            kernel_regularizer=l2(regularizer),
            name="image_embedding"))(image_input)
    image_dropout = Dropout(0.5, name="image_dropout")(image_embedding)

    recurrent_inputs = [notes_dropout, image_dropout]
    merged_input = Add()(recurrent_inputs)

    recurrent_network = GRU(
        units=hidden_size,
        recurrent_regularizer=l2(regularizer),
        kernel_regularizer=l2(regularizer),
        bias_regularizer=l2(regularizer),
        return_sequences=True,
        name='recurrent_network')(merged_input)

    output = TimeDistributed(
        Dense(
            units=NUM_NOTES,
            kernel_regularizer=l2(regularizer),
            activation="sigmoid"),
        name="output")(recurrent_network)

    inputs = [notes_input, image_input]
    model = Model(inputs=inputs, outputs=output)
    return model


if __name__ == "__main__":
    model = build(128)
