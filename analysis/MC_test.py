import numpy as np
from optimum.intel import INCModelForMultipleChoice
from transformers import RobertaTokenizer, RobertaForMultipleChoice, AutoTokenizer, AutoModelForMultipleChoice, \
    LongformerForMultipleChoice, LongformerTokenizer
import torch
from transformers import pipeline
def prepare_answering_input(
        tokenizer, # longformer_tokenizer
        question,  # str
        options,   # List[str]
        context,  # str
        max_seq_length=4096,
    ):
    c_plus_q = context + ' ' + tokenizer.bos_token + ' ' + question
    c_plus_q_4 = [c_plus_q] * len(options)
    tokenized_examples = tokenizer(
        c_plus_q_4,
        options,
        max_length=max_seq_length,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized_examples['input_ids'].unsqueeze(0)
    attention_mask = tokenized_examples['attention_mask'].unsqueeze(0)
    example_encoded = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    return example_encoded

#tokenizer = LongformerTokenizer.from_pretrained("potsawee/longformer-large-4096-answering-race")
tokenizer = AutoTokenizer.from_pretrained("LIAMF-USP/roberta-large-finetuned-race")
#model = LongformerForMultipleChoice.from_pretrained("potsawee/longformer-large-4096-answering-race")
#model = LongformerForMultipleChoice.from_pretrained("potsawee/longformer-large-4096-answering-race")
model = AutoModelForMultipleChoice.from_pretrained("LIAMF-USP/roberta-large-finetuned-race")
#model = INCModelForMultipleChoice.from_pretrained("INCModelForMultipleChoice/LIAMF-USP-roberta-large-finetuned-race/config")

context = r"""Chelsea's mini-revival continued with a third victory in a row as they consigned struggling Leicester City to a fifth consecutive defeat.
Buoyed by their Champions League win over Borussia Dortmund, Chelsea started brightly and Ben Chilwell volleyed in from a tight angle against his old club.
Chelsea's Joao Felix and Leicester's Kiernan Dewsbury-Hall hit the woodwork in the space of two minutes, then Felix had a goal ruled out by the video assistant referee for offside.
Patson Daka rifled home an excellent equaliser after Ricardo Pereira won the ball off the dawdling Felix outside the box.
But Kai Havertz pounced six minutes into first-half injury time with an excellent dinked finish from Enzo Fernandez's clever aerial ball.
Mykhailo Mudryk thought he had his first goal for the Blues after the break but his effort was disallowed for offside.
Mateo Kovacic sealed the win as he volleyed in from Mudryk's header.
The sliding Foxes, who ended with 10 men following Wout Faes' late dismissal for a second booking, now just sit one point outside the relegation zone.
Who had a goal ruled out for offside?
""".replace('\n', ' ')
options  = ['Ricardo Pereira', 'Ben Chilwell', 'Joao Felix', 'The Foxes']


#outputs = model(**inputs)
#prob = torch.softmax(outputs.logits, dim=-1)[0].tolist()

#selected_answer = options[np.argmax(prob)]
#print(prob)
#print(f'output index:{np.argmax(prob)}')
#print(selected_answer)

inputs = tokenizer([[context, options[0]], [context, options[1]],
                    [context, options[2]], [context, options[3]]
                    ], return_tensors="pt", padding=True)
labels = torch.tensor(0).unsqueeze(0) # ??? apparently needed, not too sure
outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels = labels)

logits = outputs.logits
predicted_class = logits.argmax().item()
print(predicted_class)