import openai
import os
from openai import OpenAI
import json
from rank_bm25 import BM25Okapi
from transformers import BertModel, BertTokenizer, AdamW
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# get the answer from the chatgpt
def chatgpt_answer(question, gpt_model="gpt-3.5-turbo"):
    os.environ["OPENAI_API_KEY"] = "..."
    openai.api_key = os.environ["OPENAI_API_KEY"]

    client = OpenAI()
    completion = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "user", "content": question}
        ]
    )
    return completion.choices[0].message.content


# input all the previous memory, and format them in the way "question-answer", store them in a list
def allSentences(memorybase_file_path):
    IO_list = []  # Initialize the empty list
    with open(memorybase_file_path, 'r', encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)  # Load JSON object from each line
            # Check if there are at least two messages to combine
            if len(data["messages"]) >= 2:
                # Extract sentences from the first and second messages' "content" and combine them
                combined_sentence = data["messages"][0]["content"] + "->" + data["messages"][1]["content"]
                IO_list.append(combined_sentence)  # Add the combined sentence to the list A
    return IO_list


# use some non-self learning method to select some related expamples according to the question first - BM25.
# "n" in the funcution get_top_n is the number of examples we want to get
def retrieval_first_BM25(question_asked, IO_list):
    tokenized_IO = [doc.split(" ") for doc in IO_list]
    bm25 = BM25Okapi(tokenized_IO)
    tokenized_query = question_asked.split(" ")
    return bm25.get_top_n(tokenized_query, IO_list, n=10)


def grade_and_select_forMemory(prompt, train_list):
    before_dash, after_dash = prompt.split('<->')
    dic_score = {}
    for example in train_list:
        question = f"In terms of the question-{before_dash}, by giving you this prompt-{example}, What is the " \
                   f"probability " \
                   f"that the ansewr-{after_dash} will be gave?. Your answer should be just a number, a decimal " \
                   f"between 0 and 1, no other words "
        try:
            example_score = float(chatgpt_answer(question))  # Attempt to convert
            dic_score[example] = example_score
        except ValueError:
            continue
    sorted_dict = {k: v for k, v in sorted(dic_score.items(), key=lambda item: item[1])}  # 分数是从低到高排的
    items_list = list(sorted_dict.items())
    contrastive_examples = []
    contrastive_symbol = []

    # In here, you can determine how many unrelated examples you want(value 0)
    for C, D in items_list[:3]:
        contrastive_examples.append((prompt, C))
        contrastive_symbol.append(0)

    # In here, you can determine how manny relate examples you want(value 1)
    # If the dictionary has 10 or fewer items, this will not add any new items
    for C, D in items_list[-3:]:
        contrastive_examples.append((prompt, C))
        contrastive_symbol.append(1)

    # Ensure only 10 items are returned for each list
    return contrastive_examples, contrastive_symbol


# this function will return two list
# for first five couples are the most unrelated one with the prompt, the last most are most related
# first list will contain the contrastive couple()
# second list will contain the sigal + or -
def grade_and_select(prompt, train_list):
    before_dash, after_dash = prompt.split('->')
    dic_score = {}
    for example in train_list:
        question = f"In terms of the question-{before_dash}, by giving you this prompt-{example}, What is the " \
                   f"probability " \
                   f"that the ansewr-{after_dash} will be gave?. Your answer should be just a number, a decimal " \
                   f"between 0 and 1, no other words "
        try:
            example_score = float(chatgpt_answer(question))  # Attempt to convert
            dic_score[example] = example_score
        except ValueError:
            continue
    sorted_dict = {k: v for k, v in sorted(dic_score.items(), key=lambda item: item[1])}  # 分数是从低到高排的
    items_list = list(sorted_dict.items())
    contrastive_examples = []
    contrastive_symbol = []

    # In here, you can determine how many unrelated examples you want(value 0)
    for C, D in items_list[:2]:
        contrastive_examples.append((prompt, C))
        contrastive_symbol.append(0)

    # In here, you can determine how manny relate examples you want(value 1)
    # If the dictionary has 10 or fewer items, this will not add any new items
    for C, D in items_list[-2:]:
        contrastive_examples.append((prompt, C))
        contrastive_symbol.append(1)

    # Ensure only 10 items are returned for each list
    return contrastive_examples, contrastive_symbol


# Define the Dataset
class SentencePairDataset(Dataset):
    def __init__(self, sentence_pairs_ss, labels_ss):
        self.sentence_pairs = sentence_pairs_ss
        self.labels = labels_ss

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'sentence_pair': self.sentence_pairs[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }


# Model Definition with a classification layer
class SentenceSimilarityModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(SentenceSimilarityModel, self).__init__()
        self.encoder = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size * 2, 1)

    def forward(self, sentence_pairs_s):
        flattened_sentences = [sentence_e for pair in sentence_pairs_s for sentence_e in pair]
        inputs = self.tokenizer(flattened_sentences, padding=True, truncation=True, return_tensors="pt", max_length=512)
        outputs = self.encoder(**inputs)
        sentence_embeddings = outputs.pooler_output.view(-1, 2, self.encoder.config.hidden_size)
        combined_embeddings = torch.cat((sentence_embeddings[:, 0], sentence_embeddings[:, 1]), dim=1)
        similarity_scores = self.classifier(combined_embeddings).squeeze(-1)
        return similarity_scores, sentence_embeddings


def save_model_and_optimizer(trained_model, trained_optimizer, model_path="model.pth", optimizer_path="optimizer.pth"):
    torch.save(trained_model.state_dict(), model_path)
    torch.save(trained_optimizer.state_dict(), optimizer_path)
    # print(f"Saved model to {model_path} and optimizer state to {optimizer_path}")


def load_model_and_optimizer(trained_model, trained_optimizer, model_path="model.pth", optimizer_path="optimizer.pth"):
    trained_model.load_state_dict(torch.load(model_path))
    trained_optimizer.load_state_dict(torch.load(optimizer_path))
    # print(f"Loaded model from {model_path} and optimizer state from {optimizer_path}")


def training_model(model_l, new_sentence_pairs, new_labels, optimizer_r, epochs=3, batch_size=8):
    """
    Continue training the model with new examples and labels.

    Args:
    - model: The SentenceSimilarityModel instance.
    - new_sentence_pairs: A list of tuples, where each tuple is a pair of sentences (new training data).
    - new_labels: A list of labels corresponding to the sentence pairs (1 for similar, 0 for not similar).
    - optimizer: The optimizer to use for training.
    - epochs: Number of epochs to train for.
    - batch_size: The batch size for training.
    """
    # Create a new dataset and dataloader with the new examples
    criterion = nn.BCEWithLogitsLoss()
    new_dataset = SentencePairDataset(new_sentence_pairs, new_labels)
    new_loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=True)

    # Training Loop
    model_l.train()
    for epoch in range(epochs):
        for batch in new_loader:
            optimizer_r.zero_grad()
            sentence_pairs_s = batch['sentence_pair']
            labels_s = batch['label']
            outputs, _ = model_l(sentence_pairs_s)
            loss = criterion(outputs, labels_s)
            loss.backward()
            optimizer_r.step()
        #print(f"Epoch {epoch}, Loss: {loss.item()}")


def main(file_name):
    all_sentences = allSentences(file_name)
    model = SentenceSimilarityModel()
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    start = 0
    for x in tqdm(all_sentences):
        start = start + 1
        if start > 1:
            model = SentenceSimilarityModel()
            optimizer = AdamW(model.parameters(), lr=5e-5)
            # Load the saved states
            load_model_and_optimizer(model, optimizer, "model.pth", "optimizer.pth")
            model = model.to(device)
        first_list = retrieval_first_BM25(x, all_sentences)
        sentence_pairs, labels = grade_and_select(x, first_list)
        training_model(model, sentence_pairs, labels, optimizer, epochs=2)
        save_model_and_optimizer(model, optimizer, "model.pth", "optimizer.pth")


if __name__ == '__main__':
    main()
