import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
# Define environment variable, path of data, model name and device
os.environ["HF_HOME"] = "./huggingface"  # Replace with your desired directory
print("Please replace it with your hf access token:")
os.environ["HF_HOME_TOKEN"] = "Please_replace_it_with_your_hf_access_token"

result_dir = './results'
data_file_path = './data/my_company_info.json'

model_name = "gpt2-medium"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
# Write a python file to google driver
# Sample of json datasets
# You can also directly upload this code to your google driver
# The code write here in this way is for better understanding of whole project


from torch.utils.data import Dataset
import json

class ChatData(Dataset):
    def __init__(self, path: str, tokenizer):
        self.data = json.load(open(path, "r"))

        self.X = []
        for i in self.data:
            for j in i['dialog']:
                self.X.append(j['text'])

        for idx, i in enumerate(self.X):
            try:
                self.X[idx] = "<startofstring> " + i + " <bot>: " + self.X[idx + 1] + " <endofstring>"
            except:
                break

        for i in self.data:
            for j in i['dialog']:
                self.X.append(j['text'])

        total_samples = len(self.X)  # Calculate the total number of samples
        print("total_samples", total_samples)
        # define samples amount
        self.X = self.X[:500]
        print("Here is the self.X[0] i wanna check:")
        print(self.X[0])

        self.X_encoded = tokenizer(self.X, return_tensors="pt", max_length=30, padding="max_length", truncation=True)
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx]
# Download model, save model and tokernize to harddisk
## prepare tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"pad_token": "<pad>",
                            "bos_token": "<startofstring>",
                            "eos_token": "<endofstring>"})
tokenizer.add_tokens(["<bot>:"])

## prepare model
### Specify the desired embedding size (must be a multiple of 8)
desired_embedding_size = 50264  # Change this to the desired size
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
### Resize the embedding layer to the desired size
model.resize_token_embeddings(len(tokenizer), desired_embedding_size)
model = model.to(device)

## save tokenizer and model to harddisk
tokenizer.save_pretrained(result_dir)
model.save_pretrained(result_dir)
## load model and tokenizer from harddisk
### Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(result_dir)

### Load the GPT-2 model from the local folder
model = GPT2LMHeadModel.from_pretrained(result_dir)
model.to(device)
# Define infer and train function
def infer(inp):
    inp = "<startofstring> " + inp + " <bot>: "
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)  # Use .to(device) method to move the tensor to the specified device
    a = inp["attention_mask"].to(device)  # Use .to(device) method here as well

    output = model.generate(X, attention_mask=a, max_length=100, num_return_sequences=1)

    output = tokenizer.decode(output[0])

    return output

def train(chatData, model, optim):

    epochs = 12

    for _ in tqdm.tqdm(range(epochs)):  # Use range() to iterate through epochs
        for X, a in chatData:
            print(X)
            X = X.to(device)
            a = a.to(device)
            optim.zero_grad()
            loss = model(input_ids=X, attention_mask=a, labels=X).loss
            loss.backward()
            optim.step()

    # Save the model's state dictionary after training is complete
    torch.save(model.state_dict(), "model_state.pt")
    print(infer("How do you see the integration of holographic technology in education?"))
    
from chat_data import ChatData

#Load ChatData, train model and optimizer
chatData = ChatData(data_file_path, tokenizer)
chatData = DataLoader(chatData, batch_size=64)

model.train()

optim = Adam(model.parameters())
# train 10 times
epochs = 10  # You can adjust the number of epochs as needed
for epoch in range(epochs):
    print("Round: ", epoch)
    train(chatData, model, optim)

inp = ""
while True:
    inp = input("Enter your input (press Enter when done): " + " " * 20)
    print(infer(inp))
    

