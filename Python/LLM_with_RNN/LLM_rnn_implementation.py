import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader
import random


def read_file(filename):
    # file into a continuous string, lines separated by space
    with open(filename, "r") as file:
        line_list = [line.strip() for line in file]

    return " ".join(line_list)


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
    all_chars = read_file(filename)

    unique_chars = set(all_chars)

    char_to_int = {char: i for i, char in enumerate(unique_chars)}
    int_to_char = {i: char for i, char in enumerate(unique_chars)}

    file_oneshot = letters_to_oneshot(all_chars, char_to_int)

    return file_oneshot, len(char_to_int), char_to_int, int_to_char


class Net(nn.Module):
    def __init__(self, vocab_len):
        super(Net, self).__init__()
        self.dense_1 = nn.Linear(in_features=(vocab_len * 2), out_features=32)
        self.bn1 = nn.BatchNorm1d(num_features=32)

        self.dense_2 = nn.Linear(in_features=32, out_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=64)

        self.dense_3 = nn.Linear(in_features=64, out_features=32)
        self.bn3 = nn.BatchNorm1d(num_features=32)

        self.dense_4 = nn.Linear(in_features=32, out_features=16)
        self.bn4 = nn.BatchNorm1d(num_features=16)

        self.dense_5 = nn.Linear(in_features=16, out_features=vocab_len)

    def forward(self, x_prev, x_curr):
        x = torch.concatenate((x_prev, x_curr), dim=1)
        x = F.relu(self.bn1(self.dense_1(x)))
        x = F.relu(self.bn2(self.dense_2(x)))
        x = F.relu(self.bn3(self.dense_3(x)))
        x = F.relu(self.bn4(self.dense_4(x)))
        return self.dense_5(x)


class CustomDataset(Dataset):
    # dataloader
    def __init__(self, X_prev, X_curr, y):
        self.X_prev = X_prev
        self.X_curr = X_curr
        self.y = y

    def __len__(self):
        return len(self.X_prev)

    def __getitem__(self, index):
        return self.X_prev[index], self.X_curr[index], self.y[index]


def make_dataloader(oneshot_encodings):
    X_prev = []
    X_curr = []
    y = []

    for i in range(1, len(oneshot_encodings) - 1):
        prev = oneshot_encodings[i - 1]
        curr = oneshot_encodings[i]
        next = oneshot_encodings[i + 1]

        X_prev.append(torch.tensor(prev, dtype=torch.float32))
        X_curr.append(torch.tensor(curr, dtype=torch.float32))
        y.append(torch.tensor(next, dtype=torch.float32))

    X_prev = torch.stack(X_prev)
    X_curr = torch.stack(X_curr)
    y = torch.stack(y)

    dataset = CustomDataset(X_prev, X_curr, y)

    return DataLoader(dataset, batch_size=1024, shuffle=True)


def get_expected_predictions(file_string):
    # gets a ranfom index and a 50 long string after that
    starting_index = random.randrange(0, len(file_string) - 50)

    expected = file_string[starting_index : starting_index + 50]

    return expected


def predict_next_char(model, prev_char, curr_char, int_to_char, char_to_int):
    # predicts the next character using the mode

    oneshot_prev = torch.tensor(letters_to_oneshot(prev_char, char_to_int), dtype=torch.float32)
    oneshot_curr = torch.tensor(letters_to_oneshot(curr_char, char_to_int), dtype=torch.float32)

    with torch.no_grad():
        predicted = model(oneshot_prev, oneshot_curr)
    num = torch.argmax(predicted, dim=1)[0]

    predicted = int_to_char[num.item()]

    return predicted


def eval_function(model, file_name, int_to_char, char_to_int):
    # evaluates the model by generating a 50 char long string using the model
    file_string = read_file(file_name)

    target_string = get_expected_predictions(file_string)

    predicted_string = target_string[:2]

    prev_char = predicted_string[0]
    curr_char = predicted_string[1]

    model.eval()

    for _ in range(48):
        next_char = predict_next_char(model, prev_char, curr_char, int_to_char, char_to_int)

        predicted_string += next_char

        prev_char = curr_char
        curr_char = next_char

    return predicted_string, target_string


oneshot_encodings, len_unique_chars, char_to_int, int_to_char = file_to_oneshot("abcde_edcba.txt")

dataloader = make_dataloader(oneshot_encodings)

print("model that takes into account the previous 2 characters")
model = Net(len_unique_chars)
model.train()
num_epochs = 10
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# decaying LR
scheduler = ExponentialLR(optimizer, gamma=0.9)

for epoch in range(num_epochs):
    total_loss = 0
    total_samples = 0

    for X_prev_batch, X_curr_batch, y_batch in dataloader:
        y_pred = model(X_prev_batch, X_curr_batch)

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
    print(f"starting characters: '{target_string[:2]}'")
    print(f"predicted string: {predicted_string}")
    print()


def prediction_accuracy(predicted_string, target_string):
    # gets the accuracy of the strings by comparing how many characters are equal to eachother
    pred_array = np.array([*predicted_string])
    target_array = np.array([*target_string])
    assert len(pred_array) == len(target_array)

    return (pred_array == target_array).mean()


def print_evaluations(file_name, target_string, predicte_string):
    # printing function for quickly printing accuracy statistics
    print(f"predicting the {file_name} file")
    print(f"staring characters: '{target_string[:2]}'")
    print(f"target string: {target_string}")
    print(f"predicted string: {predicte_string}")
    acc = prediction_accuracy(predicte_string, target_string)
    print(f"prediction accuracy: {acc * 100:.2f}%")
    print()


file_name = "abcde_edcba.txt"
predicte_string, target_string = eval_function(model, file_name, int_to_char, char_to_int)
print_evaluations(file_name, target_string, predicte_string)
