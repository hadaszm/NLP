from transformers import BertTokenizer, BertForPreTraining
import torch
import pandas as pd
import random

class OurDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)


def train_bert(data_filename, epochs=2):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForPreTraining.from_pretrained('bert-base-uncased')

    df = pd.read_csv(data_filename)
    df['tokenized_sentences'] = df['tokenized_sentences'].apply(eval)
    text = df[df['Type'] == 'a']['tokenized_sentences'].to_list()
    bag = sum(text, [])
    bag_size = len(bag)
    
    
    sentence_a = []
    sentence_b = []
    label = []

    for paragraph in text:
        sentences = [
            sentence for sentence in paragraph if sentence != ''
        ]
        num_sentences = len(sentences)
        if num_sentences > 1:
            start = random.randint(0, num_sentences-2)
            # 50/50 whether is IsNextSentence or NotNextSentence
            if random.random() >= 0.5:
                # this is IsNextSentence
                sentence_a.append(sentences[start])
                sentence_b.append(sentences[start+1])
                label.append(0)
            else:
                index = random.randint(0, bag_size-1)
                # this is NotNextSentence
                sentence_a.append(sentences[start])
                sentence_b.append(bag[index])
                label.append(1)
    inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt',
                   max_length=512, truncation=True, padding='max_length')
    
    inputs['next_sentence_label'] = torch.LongTensor([label]).T

    inputs['labels'] = inputs.input_ids.detach().clone()

    # create random array of floats with equal dimensions to input_ids tensor
    rand = torch.rand(inputs.input_ids.shape)
    # create mask array
    mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
               (inputs.input_ids != 102) * (inputs.input_ids != 0)
    
    selection = []

    for i in range(inputs.input_ids.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )
    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103

    dataset = OurDataset(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()
    from transformers import AdamW 
    optim = AdamW(model.parameters(), lr=5e-5)

    from tqdm import tqdm  # for our progress bar

    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, leave=True)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            next_sentence_label = batch['next_sentence_label'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = model(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            next_sentence_label=next_sentence_label,
                            labels=labels)
            # extract loss
            loss = outputs.loss
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
    return model 