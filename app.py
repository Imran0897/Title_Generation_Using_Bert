
from transformers import BertTokenizer, BertForMaskedLM

def generate_title(input_text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors='pt')
    outputs = model.generate(input_ids, max_length=20, num_return_sequences=5)

    generated_titles = []
    for output in outputs:
        generated_title = tokenizer.decode(output, skip_special_tokens=True)
        generated_titles.append(generated_title)

    return generated_titles

# Example usage
input_text = "Abstract: This paper presents a deep learning approach for automatic image captioning. The proposed model utilizes a convolutional neural network (CNN) for image feature extraction and a recurrent neural network (RNN) with long short-term memory (LSTM) cells for generating captions. The model is trained on a large dataset of images and their corresponding captions. Experimental results demonstrate the effectiveness of the proposed approach, with the generated captions achieving high accuracy and coherence. The proposed method has potential applications in various domains such as computer vision, image understanding, and assistive technologies for the visually impaired."

generated_titles = generate_title(input_text)
for title in generated_titles:
    print(title)
