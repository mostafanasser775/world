import streamlit as st
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import wget
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

config={
  "type": "service_account",
  "project_id": "graduation-5ebe4",
  "private_key_id": "3fd05142e13bdb2989cc69c54f6552b41d2a8aed",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQDYCe0aztpw3/9z\nI5Z8bS1oSFt6fa2glutuJDpx+J9Q0pqnHjbeMdJ0LymvC1VuydwxsLXc6LyBE1o/\nbyYnKPpUGF4vZMeMl0cgBiV9VACtZVe9GhZ6HIPYs03OyJUl/3OVXmGPAFrwbUsr\nLAgo14hqIKEDB+iVkBdT7J1FKwYzFJRR9RHhbLq4tFR5CZGZus9WD7hzvFEAzFNu\nAInWPbI4DGIiy+geblo/WTQQY1heVO7RLnMYg4fWreBg3wSSl+QbPDYfaHN50lBe\nkoMhqbHRbWWAxU/r9qKyPst9BRIyAOgy7NgGzlcIHbXvjyFt5R3y2w0S9uni3Qy6\nWWXw+J0VAgMBAAECggEACmdwX4GvbYJ5sfDe3Hp08Bt7kLqj0k+XIkLRr+9Um6B7\nQBaZdybOyl3pSj7uImVLlF08zBkT11YYNJgn+t7pyIOzteEXpuRNkaCyJmbvMTda\n9X/frfBL6jNfL4wjekTv3k7no5EiQzqXa7Diak/puvPia5V/KwRZZA2JjABbLoaV\nygff5/DggE7/uTIRIayQE8TKDXETxkSQ0nbOxq8mO8aNy0DwBz7svZJ7Lskbj6dN\ncSrP/0wDZdg6CuGq0l44iXA7oO5FiB/XxTxm0pCZj31pc73xBTVuQlA1EQqoLmhV\nEjqMxA/MxulwgbnLvGP8PYjYf6khsz25fxgKnj2c8QKBgQDtVMUgkLvu2fd72Cw8\n5rkMtKHzoDcxdx9/HGw5bYBCKP5iQr1YVPxONWyZFKU7WLs5TgpsOJtBzoo2Wv1x\nw8+28FY3ZrLBTSdzsmZ1+Os83tYsHrCoTszHWukU/N1IufDGX0EmDLAA3uY/MJEP\neu+oAnYstm4/IPZAYayFkTfdkQKBgQDpCGIzTFHw8Xhp9deANEEh8CVpYO96tBS7\n5tWPm5lQNE1Wf2PERXwnlZ/1be6hZiJXeFL7Klm/uxbrJIUs0LeQfOJhTq9S3Fc6\ndqYcgAOZJw5M7kEGmXkMbiiD4Qds+v8B1ehb4280pAa11VWb1etQsOe5BqNtys0x\nPU0YUW0VRQKBgQC910wj44Jg6fu8Fcw1Hv2w+yB855CewcHxBIRRX5Tz1yS85tPc\nv4Ze7P8kaE5PbYe9q/5MWO9gMV1/Y0NOaCpUFGVyxXSBiTzgoDizb1yEAV/iRN5c\nk0Pcx4ygXDCJxyqhE3rie82htKsKqseuUVE43Fc5JuiDGNPB5h+BbAr68QKBgQCl\n4e3ljmKWHY4V/4bUIF5tBkHbbcGLmz1XNM65V44fZdvXwv8F7GCg3QXs88B26/lu\nQpzvZgpVTZzW7jxO1pSVKhEMK7LqTSda2rMMfqQRFZg8cy2ewQlGK/RzTHC4x0NX\nzymEn7W9xzVvotk6AWFtI0EQmQUlVsQSVQzYTr5T8QKBgQDJ8d3byzLIZR+bw4I4\niPk6g4eMXAsvdXc9RA8WjP6uoRnVd3PtI65r2vj+Mg7quy4Oa5eZ5oKyKRCviGIK\no3DB14aEMtyJzSprUnk9PrKqPoy6DniT8TVXx5t+TRl1wzmYHsyw4Ekt3MNDFrbx\neV494MUsjiEjotdOUSKNpbPs8g==\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-p6zzi@graduation-5ebe4.iam.gserviceaccount.com",
  "client_id": "106841893328540678775",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-p6zzi%40graduation-5ebe4.iam.gserviceaccount.com"
}



@st.cache(allow_output_mutation=True)
def loadmodel():
    if os.path.isfile('./pytorch_model.bin'):
        m = ""
    else:
        cred = credentials.Certificate(config)
        firebase_admin.initialize_app(cred)

        url = 'https://storage.googleapis.com/sfr-codet5-data-research/finetuned_models/concode_codet5_base.bin'
        wget.download(url)
        old_name = r"./concode_codet5_base.bin"
        new_name = r"./pytorch_model.bin"
        os.rename(old_name, new_name)

        url = 'https://storage.googleapis.com/sfr-codet5-data-research/pretrained_models/codet5_base/config.json'
        wget.download(url)
    model = T5ForConditionalGeneration.from_pretrained('./')
    return model




tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

db=firestore.client()
docs = db.collection('code').get()
code=str(docs[0].to_dict())
len= len(code)
code=code[10:len-2]
st.write(code)
# Renaming the file

model=loadmodel()
input_ids = tokenizer(code, return_tensors="pt").input_ids

# simply generate one code span
generated_ids = model.generate(input_ids, max_length=100)
code=tokenizer.decode(generated_ids[0], skip_special_tokens=True)
st.write(code)
db.collection('output').document("p1").update({"code":code}) # field already exists
