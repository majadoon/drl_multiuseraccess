"""
Deep Q network agent with Dueling
"""
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Multiply, Lambda, Add
from keras.optimizers import Adam, RMSprop, SGD
from keras import backend as K


class DQNAgent:

    def __init__(self, step_size, state_size, action_size, dueling, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.dueling = dueling
        self.model = self.build_model()    # original model
        self.target_model = self.build_model()  # target model

        
    # creating a deep neural Q-network model
    def build_model(self):
        x_in = Input(shape=(self.step_size, self.state_size))
        lstm_layer = LSTM(100, activation="tanh", recurrent_activation="tanh", 
                          kernel_initializer="glorot_uniform")(x_in)
        hidden_layer1 = Dense(10, activation='relu', 
                              kernel_initializer="he_uniform")(lstm_layer)

        if self.dueling:
            print("Dueling Mode")

            # way 1
            # '''
            hidden_layer2 = Dense(10, activation='relu', 
                                  kernel_initializer="he_uniform")(lstm_layer)
            state_value = Dense(1)(hidden_layer2)
            # state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(self.action_size, ))(hidden_layer1)
            action_adv = Dense(self.action_size)(hidden_layer1)
            action_adv = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
                                output_shape=(self.action_size,))(action_adv)
            output_y = Add()([state_value, action_adv])
            # '''
            '''
            # way 2
            x = Dense(self.action_size + 1, activation='linear')(hidden_layer1)
            output_y = Lambda(lambda i: K.expand_dims(i[:, 0], -1) + i[:, 1:] - 
                              K.mean(i[:, 1:], keepdims=True), output_shape=(self.action_size,))(x)
            '''
        else:
            print("No dueling")
            output_y = Dense(self.action_size, activation='linear')(hidden_layer1)
        #epochs = 50
        #decay_rate = self.learning_rate / 2*epochs
        #opt = keras.optimizers.RMSprop(lr=self.learning_rate, decay = decay_rate)
        model = Model(inputs=x_in, outputs=output_y, name="DQN")
        #model.compile(loss=['mse'], optimizer=opt)
        model.compile(loss=['mse'], optimizer=Adam(lr=self.learning_rate))

        return model

    def load_weights(self, name):
        self.model.load_weights(name)

    def save_weights(self, name):
        self.model.save_weights(name)
