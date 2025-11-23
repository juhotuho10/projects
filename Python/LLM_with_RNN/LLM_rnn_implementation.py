import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader
import random


def letters_to_oneshot(word, char_to_int):
    # converts characters into onshot based on char to int dictionary
    word_oneshot = []
    vocab_len = len(char_to_int)
    for letter in word:
        letter_oneshot = np.zeros(vocab_len)
        index = char_to_int[letter]
        letter_oneshot[index] = 1
        word_oneshot.append(letter_oneshot)
    return np.array(word_oneshot)


def file_to_oneshot(filename):
    # converts a file of strings into a list of oneshot arrays
    # also generates a char to int and int to char dictionaries
    file = open(filename, "r").read()

    unique_chars = set(file)

    char_to_int = {char: i for i, char in enumerate(unique_chars)}
    int_to_char = {i: char for i, char in enumerate(unique_chars)}

    file_oneshot = letters_to_oneshot(file, char_to_int)

    return file_oneshot, len(char_to_int), char_to_int, int_to_char


class Net(nn.Module):
    def __init__(self, vocab_len):
        super(Net, self).__init__()

        ########################## new state ###################################

        self.dense_1 = nn.Linear(in_features=(vocab_len * 2), out_features=32)
        self.bn1 = nn.BatchNorm1d(num_features=32)

        self.dense_2 = nn.Linear(in_features=32, out_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=64)

        self.dense_3 = nn.Linear(in_features=64, out_features=32)
        self.bn3 = nn.BatchNorm1d(num_features=32)

        self.dense_4 = nn.Linear(in_features=32, out_features=32)
        self.bn4 = nn.BatchNorm1d(num_features=32)

        self.dense_5 = nn.Linear(in_features=32, out_features=(vocab_len))

        ########################## new character ###################################

        self.dense_6 = nn.Linear(in_features=(vocab_len), out_features=32)
        self.bn6 = nn.BatchNorm1d(num_features=32)

        self.dense_7 = nn.Linear(in_features=32, out_features=64)
        self.bn7 = nn.BatchNorm1d(num_features=64)

        self.dense_8 = nn.Linear(in_features=64, out_features=32)
        self.bn8 = nn.BatchNorm1d(num_features=32)

        self.dense_9 = nn.Linear(in_features=32, out_features=32)
        self.bn9 = nn.BatchNorm1d(num_features=32)

        self.dense_10 = nn.Linear(in_features=32, out_features=(vocab_len))

    def forward(self, x_curr, prev_state):
        x = torch.concatenate((x_curr, prev_state), dim=-1).unsqueeze(0)
        x = F.relu(self.bn1(self.dense_1(x)))
        x = F.relu(self.bn2(self.dense_2(x)))
        x = F.relu(self.bn3(self.dense_3(x)))
        x = F.relu(self.bn4(self.dense_4(x)))
        new_state = self.dense_5(x)

        x = F.relu(self.bn6(self.dense_6(new_state)))
        x = F.relu(self.bn7(self.dense_7(x)))
        x = F.relu(self.bn8(self.dense_8(x)))
        x = F.relu(self.bn9(self.dense_9(x)))
        new_character = self.dense_10(x)

        return new_character.squeeze(), new_state.squeeze()


class CustomDataset(Dataset):
    # dataloader
    def __init__(self, X_curr, y):
        self.X_curr = X_curr
        self.y = y

    def __len__(self):
        return len(self.X_curr)

    def __getitem__(self, index):
        return self.X_curr[index], self.y[index]


def make_dataloader(oneshot_encodings, batch_size):
    X_curr = []
    y = []

    # only using 1/10 of the data because getting predictions takes AGES when using a for loop
    for i in range(1, int(len(oneshot_encodings) / 10) - 1):
        curr = oneshot_encodings[i]
        next = oneshot_encodings[i + 1]

        X_curr.append(torch.tensor(curr, dtype=torch.float32))
        y.append(torch.tensor(next, dtype=torch.float32))

    X_curr = torch.stack(X_curr)
    y = torch.stack(y)

    dataset = CustomDataset(X_curr, y)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_expected_predictions(file_string):
    # gets a ranfom index and a 50 long string after that
    starting_index = random.randrange(0, len(file_string) - 50)

    expected = file_string[starting_index : starting_index + 50]

    return expected


def predict_next_char(model, curr_char, prev_state, int_to_char, char_to_int):
    # predicts the next character using the mode

    oneshot_curr = torch.tensor(letters_to_oneshot(curr_char, char_to_int), dtype=torch.float32).squeeze()

    with torch.no_grad():
        prediction, new_state = model(oneshot_curr, prev_state)

    num = torch.argmax(prediction, dim=0)

    predicted_char = int_to_char[num.item()]

    return predicted_char, new_state


def eval_function(model, file_name, int_to_char, char_to_int):
    # evaluates the model by generating a 50 char long string using the model
    file_string = open(file_name, "r").read()

    target_string = get_expected_predictions(file_string)

    vocab_len = len(int_to_char)

    predicted_string = target_string[0]

    curr_char = predicted_string[0]
    prev_state = torch.tensor([0] * vocab_len)

    model.eval()

    for _ in range(49):
        next_char, next_state = predict_next_char(model, curr_char, prev_state, int_to_char, char_to_int)

        predicted_string += next_char

        curr_char = next_char
        prev_state = next_state

    return predicted_string, target_string


def get_prediction(model, X_curr_batch, vocab_len):
    predictions = []
    prev_state = torch.tensor([0] * vocab_len)
    model.eval()
    for X_curr in X_curr_batch:
        predicted_char, new_state = model(X_curr, prev_state)

        predictions.append(predicted_char)

        prev_state = new_state

    return torch.stack(predictions, dim=0)


def train_model(file_name):
    print(f"training for file: {file_name}")
    oneshot_encodings, len_unique_chars, char_to_int, int_to_char = file_to_oneshot(file_name)

    dataloader = make_dataloader(oneshot_encodings, 64)

    model = Net(len_unique_chars)

    num_epochs = 5
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # decaying LR
    scheduler = ExponentialLR(optimizer, gamma=0.5)

    for epoch in range(num_epochs):
        total_loss = 0
        total_samples = 0
        for X_curr_batch, y_batch in dataloader:
            y_pred = get_prediction(model, X_curr_batch, len_unique_chars)

            model.train()
            loss = loss_fn(y_pred, y_batch)
            current_loss = loss.item() * X_curr_batch.size(0)
            total_loss += current_loss
            total_samples += X_curr_batch.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # update scheduled learning rate
        scheduler.step()

        average_loss = total_loss / total_samples
        print(f"Epoch {epoch + 1}, Loss: {average_loss}")
        predicted_string, target_string = eval_function(model, "abcde_edcba.txt", int_to_char, char_to_int)
        print(f"starting character: '{target_string[0]}'")
        print(f"predicted string: \n{predicted_string}")
        print()

    return model, int_to_char, char_to_int


def prediction_accuracy(predicted_string, target_string):
    # gets the accuracy of the strings by comparing how many characters are equal to eachother
    pred_array = np.array([*predicted_string])
    target_array = np.array([*target_string])
    assert len(pred_array) == len(target_array)

    return (pred_array == target_array).mean()


def print_evaluations(file_name, target_string, predicted_string):
    # printing function for quickly printing accuracy statistics
    print(f"predicting the {file_name} file")
    print(f"staring characters: '{target_string[0]}'")
    print(f"target string: \n{target_string}")
    print()
    print(f"predicted string: \n{predicted_string}")
    acc = prediction_accuracy(predicted_string, target_string)
    print(f"prediction accuracy for {file_name}: {acc * 100:.2f}%")
    print()


print("model that takes into account the previous 1 character and models own state")
file_name = "abcde.txt"

model, int_to_char, char_to_int = train_model(file_name)
predicted_string, target_string = eval_function(model, file_name, int_to_char, char_to_int)
print_evaluations(file_name, target_string, predicted_string)


file_name = "abcde_edcba.txt"
model, int_to_char, char_to_int = train_model(file_name)
predicted_string, target_string = eval_function(model, file_name, int_to_char, char_to_int)
print_evaluations(file_name, target_string, predicted_string)
