# Introduction

- Main NLP TASKS COURSE:

https://huggingface.co/course/chapter7/1?fw=pt

## Working with pipelines

 It connects a model with its necessary preprocessing and postprocessing steps, allowing us to directly input any text and get an intelligible answer:

```py
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")

# [{'label': 'POSITIVE', 'score': 0.9598047137260437}]
```

it is also possible to pass batch of text.

the zero-shot classification pipeline lets you select the labels for classification.

using transformers we can perform many different operations:
- text classification
- zero-shot-classification
- text generation
- text completion (mask filling)
- token classification
- question answering
- summarization
- translation

### Zero-shot classification

We’ll start by tackling a more challenging task where we need to classify texts that haven’t been labelled. This is a common scenario in real-world projects because annotating text is usually time-consuming and requires domain expertise. For this use case, the zero-shot-classification pipeline is very powerful: it allows you to specify which labels to use for the classification, so you don’t have to rely on the labels of the pretrained model. You’ve already seen how the model can classify a sentence as positive or negative using those two labels — but it can also classify the text using any other set of labels you like.


```py
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)

# {'sequence': 'This is a course about the Transformers library',
#  'labels': ['education', 'business', 'politics'],
#  'scores': [0.8445963859558105, 0.111976258456707, 0.043427448719739914]} 
```

### Text-generation 


```py
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)

# [{'generated_text': 'In this course, we will teach you how to manipulate the world and '
#                     'move your mental and physical capabilities to your advantage.'},
#  {'generated_text': 'In this course, we will teach you how to become an expert and '
#                     'practice realtime, and with a hands on experience on both real '
#                     'time and real'}]
```

### Mask-filling

The next pipeline you’ll try is fill-mask. The idea of this task is to fill in the blanks in a given text:


```py
from transformers import pipeline

unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)
```

### Named entity recognition
Named entity recognition (NER) is a task where the model has to find which parts of the input text correspond to entities such as persons, locations, or organizations. Let’s look at an example:


```py
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

## Question answering
The question-answering pipeline answers questions using information from a given context:


```py
from transformers import pipeline

question_answerer = pipeline("question-answering")
question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)

# {'score': 0.6385916471481323, 'start': 33, 'end': 45, 'answer': 'Hugging Face'}
```
Note that this pipeline works by extracting information from the provided context; it does not generate the answer.


## Language Models

All the Transformer models mentioned above (GPT, BERT, BART, T5, etc.) have been trained as language models. This means they have been trained on large amounts of raw text in a self-supervised fashion. Self-supervised learning is a type of training in which the objective is automatically computed from the inputs of the model. That means that humans are not needed to label the data!

This type of model develops a statistical understanding of the language it has been trained on, but it’s not very useful for specific practical tasks. Because of this, the general pretrained model then goes through a process called transfer learning. 

During this process, the model is fine-tuned in a supervised way — that is, using human-annotated labels — on a given task.

An example of a task is predicting the next word in a sentence having read the n previous words. This is called **causal language modeling** because the output depends on the past and present inputs, but not the future ones. Another example is **masked language modeling**, in which the model predicts a masked word in the sentence.

Transformes are big models, apart from a few outliers (like DistilBERT), the general strategy to achieve better performance is by increasing the models’ sizes as well as the amount of data they are pretrained on.

### Transfer Learning

the idea of transfer learning is to initialize the weights from an already trained model a fine tuning the last layer for our purposes. pretrained models are usually trained on very large amount of data and that is why they tend to work always better than models trained from scratch. 

usually, transfer learning is applied by dropping the head of the pretrained model while keeping its body. Fine-tuning, on the other hand, is the training done after a model has been pretrained. To perform fine-tuning, you first acquire a pretrained language model, then perform additional training with a dataset specific to your task.


## Transformers

### Architectures introduction

The model is primarily composed of two blocks:

- Encoder: The encoder receives an input and builds a representation of it (its features). This means that the model is optimized to acquire understanding from the input.


Each of these parts can be used independently, depending on the task:

- **Encoder-only model**s: Good for tasks that require understanding of the input, such as sentence classification and named entity recognition.

- **Decoder-only models**: Good for generative tasks such as text generation.

- **Encoder-decoder models or sequence-to-sequence models**: Good for generative tasks that require an input, such as translation or summarization.

A key feature of Transformer models is that they are built with special layers called attention layers. all you need to know is that this layer will **tell the model to pay specific attention to certain words** in the sentence you passed it (and more or less ignore the others) when dealing with the representation of each word.

The same concept applies to any task associated with natural language: a word by itself has a meaning, but that meaning is deeply affected by the context, which can be any other word (or words) before or after the word being studied.

### Architectures vs. checkpoints
As we dive into Transformer models in this course, you’ll see mentions of architectures and checkpoints as well as models. These terms all have slightly different meanings:

Architecture: This is the skeleton of the model — the definition of each layer and each operation that happens within the model.
Checkpoints: These are the weights that will be loaded in a given architecture.


### Encoder

The encoder outputs a numerical representation for each word used as input. the numerical representation is called **Feature vector**, it contains one vector per word (numerical representation), the dimension of the vector is defined by the architecture of the model itself.

Each word in the initial sequence affects every word's representation, they are **contextual vectors**. it does this thanks to the self-attention mechanism. The representation of the word is given by the others words in the prase.

encoders can use a standalone models and are particular good when we have to extract meaningful information, we are dealing with sequence classification, q&a and masked language modelling. 

if we are dealing with masked language modelling to usage of encoders, with their bi-directional context, are very good in term of accuracy. They are also good in text-classification as sentiment analysis.

Encoder models use only the encoder of a Transformer model. At each stage, the attention layers can access all the words in the initial sentence. These models are often characterized as having “bi-directional” attention, and are often called auto-encoding models.

The pretraining of these models usually revolves around somehow corrupting a given sentence (for instance, by masking random words in it) and tasking the model with finding or reconstructing the initial sentence.

Encoder models are best suited for tasks requiring an understanding of the full sentence, such as sentence classification, named entity recognition (and more generally word classification), and extractive question answering. (ALBERT, BERT, DistilBERT, ELECTRA, RoBERTa)


### Decoder 
The decoder uses the encoder’s representation (features) along with other inputs to generate a target sequence. This means that the model is optimized for generating outputs.Decoder models use only the decoder of a Transformer model. At each stage, for a given word the attention layers can only access the words positioned before it in the sentence. These models are often called auto-regressive models.

The pretraining of decoder models usually revolves around predicting the next word in the sentence. These models are best suited for tasks involving text generation.


### Encoder-Decoder

Encoder generates numerical representation of the text, in this case we pass direct the numerical representation of the encoder to the decoder as well with a start of sequence word.

the decoder decode the sequence a output a word. not as in an autoregressive model in combination with the representation of the encoder can used to generate a second word.

There is also the possibility to create an encoder-decoder assembling a custom encoder model and a decoder model

# HuggingFace Models

looking at the sentiment analysis we have three stages: the tokenizer, then this output is given to the model and finally we generate the probability from the log-odds.

## Tokenizer API

Like other neural networks, Transformer models can’t process raw text directly, so the first step of our pipeline is to convert the text inputs into numbers that the model can make sense of. To do this we use a tokenizer, which will be responsible for:

- Splitting the input into words, subwords, or symbols (like punctuation) that are called tokens

- Mapping each token to an integer

- Adding additional inputs that may be useful to the model

All this preprocessing needs to be done in exactly the same way as when the model was pretrained, so we first need to download that information from the Model Hub. o do this, we use the AutoTokenizer class and its from_pretrained() method. Using the checkpoint name of our model, it will automatically fetch the data associated with the model’s tokenizer and cache it

```py
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

Once we have the tokenizer, we can directly pass our sentences to it and we’ll get back a dictionary that’s ready to feed to our model! The only thing left to do is to convert the list of input IDs to tensors (**Transformer models only accept tensors as input**).
 
 

```py
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt") #torch, tf for tensorflow
print(inputs)

"""
{
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}
"""
```

## Going trough the model
We can download our pretrained model the same way we did with our tokenizer. 🤗 Transformers provides an AutoModel class which also has a from_pretrained() method:


```py
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```

The vector output by the Transformer module is usually large. It generally has three dimensions:

- Batch size: The number of sequences processed at a time (2 in our example).

- Sequence length: The length of the numerical representation of the sequence (16 in our example).

- Hidden size: The vector dimension of each model input.
It is said to be “high dimensional” because of the last value. The hidden size can be very large (768 is common for smaller models, and in larger models this can reach 3072 or more).


```py
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
#torch.Size([2, 16, 768])
```


## AutoModelFor*
There are many different architectures available in 🤗 Transformers, with each one designed around tackling a specific task. Here is a non-exhaustive list:

*Model (retrieve the hidden states)
*ForCausalLM
*ForMaskedLM
*ForMultipleChoice
*ForQuestionAnswering
*ForSequenceClassification
*ForTokenClassification

For our example, we will need a model with a sequence classification head (to be able to classify the sentences as positive or negative). So, we won’t actually use the AutoModel class, but AutoModelForSequenceClassification:


```py
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
```


# Models
The AutoModel class and all of its relatives are actually simple wrappers over the wide variety of models available in the library. It’s a clever wrapper as it can automatically guess the appropriate model architecture for your checkpoint, and then instantiates a model with this architecture.

## Config and fresh models
we have to also care about the **configuration of the model**!

```py
from transformers import BertConfig, BertModel

config = BertConfig() # Building the config
model = BertModel(config) # Building the model from the config
print(config)

# BertConfig {
#   [...]
#   "hidden_size": 768,
#   "intermediate_size": 3072,
#   "max_position_embeddings": 512,
#   "num_attention_heads": 12,
#   "num_hidden_layers": 12,
#   [...]
# }
```

## Load pretrained models
doing the operation above we are initiziationig the model with no pretrained weights! if instead we want that the models arrived pretrained we just have to specify it and giving the wanted checkpoint

```py
from transformers import BertModel
model = BertModel.from_pretrained("bert-base-cased")

# we could replace BertModel with the equivalent AutoModel class.
```

This model is now initialized with all the weights of the checkpoint. It can be used directly for inference on the tasks it was trained on, and it can also be fine-tuned on a new task. By training with pretrained weights rather than from scratch, we can quickly achieve good results. The weights have been downloaded and cached (so future calls to the from_pretrained() method won’t re-download them) in the cache folder, which defaults to ~/.cache/huggingface/transformers. You can customize your cache folder by setting the HF_HOME environment variable.

## Saving the model
to save the file just:

```py
# Saving a model is as easy as loading one — we use the save_pretrained() method, which is analogous to the from_pretrained() method:
model.save_pretrained("directory_on_my_computer")
```

