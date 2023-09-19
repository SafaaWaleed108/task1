# task1
import nltk
import random
import numpy as np
import tensorflow as tf
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample training data
training_data = [
    ("What is your name?", "My name is ChatBot."),
    ("How are you?", "I'm good. How about you?"),
    ("What are you doing?", "I'm chatting with you."),
    ("Who created you?", "I was created by OpenAI."),
    ("Goodbye", "Goodbye!"),
]

# Preprocess training data
corpus = []
labels = []
for data in training_data:
    corpus.append(data[0])
    labels.append(data[1])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
y = np.array(labels)

# Train a classifier
classifier = tf.keras.Sequential([
    tf.keras.layers.Dense(256, input_shape=(X.shape[1],), activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(y.shape[1], activation='softmax')
])
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.fit(X.toarray(), tf.keras.utils.to_categorical(y), epochs=10)

# Sequence-to-sequence model
class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Seq2SeqModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size)
        self.decoder = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.encoder(output, hidden)
        output, hidden = self.decoder(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

# Train a sequence-to-sequence model
seq2seq_model = Seq2SeqModel(X.shape[1], y.shape[1], 128)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(seq2seq_model.parameters(), lr=0.01)

def train(input_tensor, target_tensor):
    hidden = seq2seq_model.init_hidden()
    optimizer.zero_grad()
    loss = 0

    for i in range(input_tensor.size(0)):
        output, hidden = seq2seq_model(input_tensor[i], hidden)
        loss += criterion(output, target_tensor[i])

    loss.backward()
    optimizer.step()
    return loss.item() / input_tensor.size(0)

def evaluate_seq(input_tensor):
    with torch.no_grad():
        hidden = seq2seq_model.init_hidden()
        output_string = ""

        for i in range(input_tensor.size(0)):
            output, hidden = seq2seq_model(input_tensor[i], hidden)
            _, topi = output.topk(1)
            output_string += vectorizer.inverse_transform(topi.numpy())[0][0] + " "

        return output_string

X_seq = []
for sentence in corpus:
    tokens = nltk.word_tokenize(sentence.lower())
    X_seq.append([vectorizer.vocabulary_.get(token, 0) for token in tokens])

X_seq = torch.tensor(X_seq, dtype=torch.long)
y_seq = torch.tensor(y, dtype=torch.long)

for epoch in range(10):
    total_loss = 0
    for i in range(X_seq.size(0)):
        loss = train(X_seq[i], y_seq[i])
        total_loss += loss

    print(f'Epoch: {epoch + 1}, Loss: {total_loss / X_seq.size(0)}')

# Chatbot function
def chat():
    print("ChatBot: Hello! How can I assist you?")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            print("ChatBot: Goodbye!")
            break

        # Rule-based model
        input_vector = vectorizer.transform([user_input])
        predicted_label = classifier.predict(input_vector.toarray())
        response = predicted_label[0]

        # Sequence-to-sequence model
        input_seq = nltk.word_tokenize(user_input.lower())
        input_seq = torch.tensor([vectorizer.vocabulary_.get(token, 0) for token in input_seq], dtype=torch.long)
        response_seq = evaluate_seq(input_seq)

        print("ChatBot (Rule-based):", response)
        print("ChatBot (Seq2Seq):", response_seq)

# Start the chatbot
chat()
