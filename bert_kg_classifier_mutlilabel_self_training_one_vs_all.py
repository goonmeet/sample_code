import os
import sys
import json
import copy
import time
import torch
import random
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from transformers import BertTokenizer
from torch.nn import BCEWithLogitsLoss, BCELoss
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, multilabel_confusion_matrix


########### Resources ###########
# https://mccormickml.com/2019/07/22/BERT-fine-tuning/
# https://github.com/huggingface/transformers/issues/5816
# https://blog.usejournal.com/part1-bert-for-advance-nlp-with-transformers-in-pytorch-357579d63512
# https://medium.com/swlh/painless-fine-tuning-of-bert-in-pytorch-b91c14912caa
# https://colab.research.google.com/drive/1jocViLorbwWIkTXKwxCOV9HLTaDDgCaw?usp=sharing
# https://colab.research.google.com/github/joeddav/blog/blob/master/_notebooks/2020-05-29-ZSL.ipynb#scrollTo=La_ga8KvSFYd


def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--KG_FOR_TRAINING", default="Attribute", type=str)
    parser.add_argument("--STEPS_FOR_TARGET", default=10000, type=int)
    parser.add_argument("--ACCPECTANCE_THRESHOLD", default=0.95, type=float)
    parser.add_argument("--NUM_LABELS", default=2, type=int)
    parser.add_argument("--EPOCHS", default=2, type=int)
    parser.add_argument("--OPTIMIZER_LR", default=0.00002, type=float)
    parser.add_argument("--OPTIMIZER_EPSILON", default=0.0000001, type=float)

    return parser.parse_args()

parser = parseArgs()

print("\n------------------ Training for {} ------------------".format(parser.KG_FOR_TRAINING))

# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

torch.backends.cudnn.benchmark = True

#### READ GQA DATA:
print("\n------------------ READ GQA DATA ------------------")
TRAIN_QUESTIONS_TO_KG = "/scratch/bajaj.32/gqa_data/Train_Questions_to_KG_Nov_20.json"
train_questions_to_kg = json.load(open(TRAIN_QUESTIONS_TO_KG, "r"))
print(len(train_questions_to_kg))

selected_data = []
df = pd.DataFrame(train_questions_to_kg).T

mlb = MultiLabelBinarizer()
knowledge_gaps = [['Activity', 'Attribute', 'Direction', 'EntityResolution', 'Location', 'Material', 'Reasoning', 'Sentiment', 'Size', 'State'],]

mlb_transformation = mlb.fit_transform(df["Knowledge_Gaps"])

for class_index in range(0, len(mlb.classes_)):
    df[mlb.classes_[class_index]] = mlb_transformation[:,class_index]
print(df.head())

postive_examples = df.loc[df[parser.KG_FOR_TRAINING] == 1]
negtive_examples = df.loc[df[parser.KG_FOR_TRAINING] == 0]

if len(negtive_examples) > len(postive_examples):
    negtive_examples = negtive_examples.sample(n=len(postive_examples), random_state=1)

df = pd.concat([postive_examples, negtive_examples])
df = shuffle(df)
questions = df.Question_Text.values#[:100]
gqa_labels = df[parser.KG_FOR_TRAINING].values#[:100]


print("GQA questions", questions[:10])
print("GQA Labels", gqa_labels[:10])
assert len(questions) == len(gqa_labels)

print('Number of GQA questions: {:,}\n'.format(len(questions)))

#### READ MCAN DATA:
print("\n------------------ READ MCAN DATA ------------------")
mcan_json = json.load(open("/scratch/bajaj.32/mcan_data/vqa/v2_OpenEnded_mscoco_train2014_questions.json", "r"))["questions"]
print(len(mcan_json))
mcan_data_for_self_training = []
for x in mcan_json:
    if "How many" not in x["question"]:
        mcan_data_for_self_training.append([x["image_id"], x["question_id"], x["question"], ])
mcan_df = pd.DataFrame(mcan_data_for_self_training, columns = ["image_id", "question_id", "question"])
print(mcan_df.head())
mcan_df = mcan_df[:30000]
print('Number of MCAN questions: {:,}\n'.format(mcan_df.shape[0]))
mcan_qids = mcan_df.question_id.values
mcan_questions = mcan_df.question.values

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Print the original question.
print('Original: ', questions[0])

# Print the question split into tokens.
print('Tokenized: ', tokenizer.tokenize(questions[0]))

# Print the question mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(questions[0])))

def get_max_question_len(tokenizer, questions):
    max_len = 0
    # For every question...
    for question in questions:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(question, add_special_tokens=True)
        # Update the maximum question length.
        max_len = max(max_len, len(input_ids))
    return max_len

NUM_QUESTIONS = 10000
max_len_gqa = get_max_question_len(tokenizer, questions)
max_len_mcan = get_max_question_len(tokenizer, mcan_questions)
max_len = max(max_len_gqa, max_len_mcan)
print('Max question length: ', max_len)


def tokenize_input_and_collect_input_ids_and_masks(tokenizer, questions):
    # Tokenize all of the questions and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []
    # For every question...
    for question in questions:
        encoded_dict = tokenizer.encode_plus(
                            question,                      # question to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 64,           # Pad & truncate all questions.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )
        # Add the encoded question to the list.
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # Print question 0, now as a list of IDs.
    print('Original: ', questions[0])
    print('Token IDs:', input_ids[0])

    return input_ids, attention_masks

print("\n------------------ TOKENIZE GQA DATA ------------------")
gqa_input_ids, gqa_attention_masks  = tokenize_input_and_collect_input_ids_and_masks(tokenizer, questions)

print("\n------------------ TOKENIZE MCAN DATA ------------------")
mcan_input_ids, mcan_attention_masks  = tokenize_input_and_collect_input_ids_and_masks(tokenizer, mcan_questions)

# Convert the lists into tensors.
gqa_labels = torch.tensor(gqa_labels)


def created_training_tensor_dataset(input_ids, attention_masks, labels=None, split=False):

    # Combine the training inputs into a TensorDataset.
    if labels is not None:
        print(input_ids.shape)
        print(attention_masks.shape)
        print(labels.shape)
        dataset = TensorDataset(input_ids, attention_masks, labels)
    else:
        dataset = TensorDataset(input_ids, attention_masks)

    train_dataset = dataset
    val_dataset = None
    # Create a 90-10 train-validation split.
    if split:
        # Calculate the number of samples to include in each set.
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        print('{:>5,} validation samples'.format(val_size))

    print('{:>5,} training samples'.format(len(train_dataset)))

    return train_dataset, val_dataset

gqa_train_dataset, gqa_val_dataset = created_training_tensor_dataset(gqa_input_ids, gqa_attention_masks, labels=gqa_labels, split=True)
mcan_dataset, _ = created_training_tensor_dataset(mcan_input_ids, mcan_attention_masks)

# The DataLoader needs to know our batch size for training, so we specify it
# here. For fine-tuning BERT on a specific task, the authors recommend a batch
# size of 16 or 32.
batch_size = 16

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order.
train_dataloader = DataLoader(
            gqa_train_dataset,  # The training samples.
            sampler = RandomSampler(gqa_train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            gqa_val_dataset, # The validation samples.
            sampler = SequentialSampler(gqa_val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

mcan_dataloader = DataLoader(
            mcan_dataset, # The validation samples.
            sampler = SequentialSampler(mcan_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

og_mcan_dataloader = copy.deepcopy(mcan_dataloader)
# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = parser.NUM_LABELS, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.cuda()


# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
optimizer = AdamW(model.parameters(),
                  lr = parser.OPTIMIZER_LR, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = parser.OPTIMIZER_EPSILON # args.adam_epsilon  - default is 1e-8.
                )

# Number of training epochs. The BERT authors recommend between 2 and 4.

# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * parser.EPOCHS
print("\n------------------ Training Details ------------------")

print("Total number of training steps", total_steps)
print("Length of Dataloader", len(train_dataloader))

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def add_to_dataloader(origial_dataloader, examples_to_add):
    return new_dataloader
# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss,
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

total_train_loss = 0


def test_on_target(model, target_dataloader, source_dataset):
    model.eval()
    examples_to_add_input_ids = []
    examples_to_input_mask = []
    examples_to_add_labels = []

    reduced_mcan_input_ids = []
    reduced_mcan_input_masks = []
    indices_to_skip = []
    counter = 0
    for i, batch in enumerate(target_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)

        with torch.no_grad():

            # values prior to applying an activation function like the softmax.
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = logits[0]

            pred_labels = torch.sigmoid(logits).to('cpu').numpy()
            pred_bools = [pl > parser.ACCPECTANCE_THRESHOLD for pl in pred_labels]

            for example_input_ids, example_input_mask, example_bools in zip(b_input_ids, b_input_mask, pred_bools):
                if any(example_bools):
                    indices_to_skip.append(counter)
                    examples_to_add_input_ids.append(torch.tensor(example_input_ids.clone().detach()))
                    examples_to_input_mask.append(torch.tensor(example_input_mask.clone().detach()))
                    if example_bools[0]:
                        examples_to_add_labels.append(torch.tensor(1).long())
                    else:
                        examples_to_add_labels.append(torch.tensor(0).long())
                else:
                    reduced_mcan_input_ids.append(torch.tensor(example_input_ids))
                    reduced_mcan_input_masks.append(torch.tensor(example_input_mask))
                counter += 1

    if len(examples_to_add_input_ids) >= 1:
        print("len(examples_to_add_input_ids) == len(examples_to_input_mask) == len(examples_to_add_labels)")

        print("Updating data loaders!!", "len of examples_to_add_input_ids", len(examples_to_add_input_ids), len(reduced_mcan_input_ids), counter)
        examples_to_add_input_ids = torch.stack(examples_to_add_input_ids) #.to("cpu")
        examples_to_input_mask = torch.stack(examples_to_input_mask) #.to("cpu")
        examples_to_add_labels = torch.stack(examples_to_add_labels) #.to("cpu")
        new_additions_to_source_dataset, _ = created_training_tensor_dataset(examples_to_add_input_ids, examples_to_input_mask, labels=examples_to_add_labels)

        new_train_dataset = torch.utils.data.ConcatDataset([source_dataset, new_additions_to_source_dataset])

        batches = []
        for ds in new_train_dataset:
            batches.append((ds[0].to("cpu"), ds[1].to("cpu"), ds[2].to("cpu")))

        new_train_dataloader = DataLoader(
                    batches,  # The training samples.
                    sampler = RandomSampler(new_train_dataset), # Select batches randomly
                    batch_size = batch_size, # Trains with this batch size., \
                    pin_memory = True
                )
        for idx, x in enumerate(target_dataloader.dataset):
            if idx in indices_to_skip:
                continue
            reduced_mcan_input_ids.append(x[0].clone().detach().to(device))
            reduced_mcan_input_masks.append(x[1].clone().detach().to(device))

        mcan_dataloader = None
        if len(reduced_mcan_input_ids) > 0:
            reduced_mcan_input_ids = torch.stack(reduced_mcan_input_ids).to(device)
            reduced_mcan_input_masks = torch.stack(reduced_mcan_input_masks).to(device)

            mcan_dataset, _ = created_training_tensor_dataset(reduced_mcan_input_ids, reduced_mcan_input_masks)

            mcan_dataloader = DataLoader(
                        mcan_dataset, # The validation samples.
                        sampler = SequentialSampler(mcan_dataset), # Pull out batches sequentially.
                        batch_size = batch_size, # Evaluate with this batch size.
                        # pin_memory = True
                    )
        return new_train_dataloader, mcan_dataloader

    return None, None

# For each epoch...
for epoch_i in range(0, parser.EPOCHS):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, parser.EPOCHS))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()
    last_train_loss = copy.deepcopy(total_train_loss)
    # Reset the total loss for this epoch.
    total_train_loss = 0

    model.train()

    training_examples = enumerate(train_dataloader)

    # For each batch of training data...
    step = 0

    for batch in tqdm(train_dataloader):
        model.train()
        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.     Last loss: {}'.format(step, len(train_dataloader), elapsed, last_train_loss))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        output = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)

        loss = output[0]
        logits = output[1]

        total_train_loss += loss.item()

        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Update the learning rate.
        scheduler.step()

        # Test model on target domain after every 1000 batches:
        if step != 0 and step % parser.STEPS_FOR_TARGET == 0 and mcan_dataloader is not None:
            print(step, "Testing on target domain ...")
            a, b = test_on_target(model, mcan_dataloader, gqa_train_dataset)
            if a is not None:
                train_dataloader, mcan_dataloader = a, b

            model = model.to(device)
            model.train()

        step += 1

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables
    total_micro_f1_score = 0
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    accumlated_label_ids = []
    accumlated_pred_bools = []
    # Evaluate data for one epoch
    for batch in validation_dataloader:

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():

            output = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask, labels=b_labels)
            loss = output[0]
            logits = output[1]
            pred_flat = np.argmax(logits.cpu(), axis=1).flatten()

        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        total_micro_f1_score += f1_score(label_ids, pred_flat, average='micro')
        val_flat_accuracy = accuracy_score(label_ids, pred_flat)
        accumlated_label_ids.extend(label_ids)
        accumlated_pred_bools.extend(pred_flat)

        pred_bools = torch.tensor(pred_flat).cpu().numpy() * 1

    model = model.to(device)
    clf_report = classification_report(accumlated_label_ids, accumlated_pred_bools, target_names=[parser.KG_FOR_TRAINING, "Not_" + parser.KG_FOR_TRAINING])
    print(clf_report)
    print(multilabel_confusion_matrix(accumlated_label_ids, accumlated_pred_bools))

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    avg_micro_f1_score = total_micro_f1_score / len(validation_dataloader)
    print("  F1 Score: {0:.2f}".format(avg_micro_f1_score))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

# Display floats with two decimal places.
pd.set_option('precision', 2)

# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)

# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')

# Display the table.
print(df_stats)
# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

# Plot the learning curve.
plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

# Label the plot.
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks(range(parser.EPOCHS))

plt.savefig("bert_loss_self_training_{}.png".format(parser.KG_FOR_TRAINING))

output_dir = './model_save_self_training_{}/'.format(parser.KG_FOR_TRAINING)

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
