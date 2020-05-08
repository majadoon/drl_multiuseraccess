"""
Deep Q network agent with Dueling
"""
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Multiply, Lambda, Add, merge
from keras.optimizers import Adam
from keras import backend as K
import numpy as np


class DQNAgent:

    def __init__(self, step_size, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.step_size = step_size
        self.model = self.build_model()    # original model

    # creating a deep neural Q-network model
    def build_model(self, dueling=True):
        x_in = Input(shape=(self.step_size, self.state_size))
        lstm_layer = LSTM(100)(x_in)
        hidden_layer1 = Dense(10, activation='relu')(lstm_layer)

        if dueling:
            print("Dueling Mode")
            hidden_layer2 = Dense(10, kernel_initializer='he_uniform', activation='relu')(lstm_layer)
            state_value = Dense(1, activation='linear')(hidden_layer2)
            # state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(self.action_size, ))(hidden_layer1)
            print('svalue:', state_value)
            action_adv = Dense(self.action_size)(hidden_layer1)
            action_adv = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
                                output_shape=(self.action_size,))(action_adv)

            output_y = Add()([state_value, action_adv])

        else:

            output_y = Dense(self.action_size, activation='linear')(hidden_layer1)

        model = Model(inputs=x_in, outputs=output_y, name="DQN")
        model.compile(loss=['mse'], optimizer=Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999),
                      metrics=['accuracy'])

        return model

    def load_weights(self, name):
        self.model.load_weights(name)

    def save_weights(self, name):
        self.model.save_weights(name)
