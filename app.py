import streamlit as st
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import wget
import os

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
wget.download("https://cdn-129.anonfiles.com/r0O0p9W2xd/a262a624-1649855588/serviceAccountKey.json")
cred = credentials.Certificate(".\serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db=firestore.client()
docs = db.collection('code').get()

code=str(docs[0].to_dict())
len= len(code)

code=code[13:len-2]
st.write(code)


if os.path.isfile('./pytorch_model.bin'):
  print('exists')
else:
  url='https://storage.googleapis.com/sfr-codet5-data-research/finetuned_models/concode_codet5_base.bin'
  wget.download(url)
  old_name = r"./concode_codet5_base.bin"
  new_name = r"./pytorch_model.bin"
  os.rename(old_name, new_name)

  url='https://storage.googleapis.com/sfr-codet5-data-research/pretrained_models/codet5_base/config.json'
  wget.download(url)

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')


# Renaming the file

model = T5ForConditionalGeneration.from_pretrained('./')

text = "make summition of 1 and 2"
input_ids = tokenizer(text, return_tensors="pt").input_ids

# simply generate one code span
generated_ids = model.generate(input_ids, max_length=100)
code=tokenizer.decode(generated_ids[0], skip_special_tokens=True)
st.write(code)
