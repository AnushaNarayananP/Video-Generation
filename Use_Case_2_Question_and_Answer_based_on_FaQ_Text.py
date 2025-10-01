with open('companyProfile.txt', 'r') as file:
    file_content = file.read()
print(file_content)
import transformers
import torch
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
def chat(user_question):

    messages = [
      {"role": "system", "content": "Please answer questions just based on this information: " + file_content},
      {"role": "user", "content": user_question},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    return (outputs[0]["generated_text"][len(prompt):])
import time

while True:
    # Ask questions to chatbot
    # Do you know company Dummy-Gpt2-Datatec-Studio Inc?
    # Which products does Dummy-Gpt2-Datatec-Studio Inc have?
    question = input("Please enter your question (or 'quit' to stop): ")
    if question.lower() == 'quit':
        break
    response = chat(question)
    print(response)