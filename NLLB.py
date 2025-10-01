from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,pipeline
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if device == 0 else "cpu")


from langdetect import detect
def translate_prompt(prompt):
    is_prompt_english = detect(prompt) == 'en'
    if not is_prompt_english:
        prompt_src_lang = 'de_Latn'
        prompt_targt_lang = 'eng_Latn'
        translation_pipeline = pipeline("translation", model=model, tokenizer=tokenizer,src_lang = prompt_src_lang,tgt_lang= prompt_targt_lang)
        output = translator(prompt,max_length=100)
        prompt= output[0]['translation_text']
    print(prompt)
first_prompt = "Example 1: I am a Berliner."
second_prompt= "Beispiel 2: Ich bin ein Berliner."
translate_prompt(first_prompt)
translate_prompt(second_prompt)