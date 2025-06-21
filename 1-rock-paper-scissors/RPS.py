import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical

# Global Variables
model = None
opponent_history = []
X_data = []
y_data = []
window_size = 5  # Determine the Last no of moves

# Encoding Dictionary
move_to_index = {'R':0, 'P':1, 'S':2}
index_to_move = ['R', 'P', 'S']
counter_move  = {'R':'P', 'P':'S', 'S':'R'}

# Encode to Vectors (One-Hot Encoding)
def encode_move(move):
    return to_categorical(move_to_index[move], num_classes=3)

# Initialize a RNN model
def build_model():
    m = Sequential()
    m.add(SimpleRNN(16, input_shape=(window_size, 3), activation='relu'))
    m.add(Dense(3, activation='softmax'))
    m.compile(optimizer='adam', loss='categorical_crossentropy')
    return m

def player(prev_play, model_built=[False]):
    global model, opponent_history, X_data, y_data, max_training_moves

    # First move
    if prev_play == "":
        return "R"

    # Update history
    opponent_history.append(prev_play)
    
    # Train after enough data
    if len(opponent_history) >= window_size+1:
        seq = opponent_history[-(window_size + 1):-1]
        label = opponent_history[-1]

        X_data.append([encode_move(m) for m in seq])
        y_data.append(encode_move(label))

        X = np.array(X_data)
        y = np.array(y_data)

        if not model_built[0]:
            model = build_model()
            model_built[0] = True
        
        model.fit(X, y, epochs=5, verbose=0)

        last_seq = np.array([[encode_move(m) for m in opponent_history[-window_size:]]])
        prediction = model.predict(last_seq, verbose=0)[0]
        predicted_index = np.argmax(prediction)
        predicted_move = index_to_move[predicted_index]

        return counter_move[predicted_move]
        
    return np.random.choice(['R', 'P', 'S'])