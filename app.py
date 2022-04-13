import streamlit as st
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import wget
import os

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
config={
  "type": "service_account",
  "project_id": "gpt52-88644",
  "private_key_id": "2195ec111eb069f3dbfe68b5c00f6a313d86ff5e",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDIXDK0KHSge+ZX\nNGehMufbWcPWWxWMdV5h06iHx0dyXgVeHbcypGyznk8P4U4FrLhhxS3Kqiah0buH\ntNARIAlm13/nO1k9dN1sGJDwd/rU9R3ux/Bgayt4QM+gLBsNEQw6Qdwp14ErHSCM\nB9DkMh59DY+gAI+KwWduw+Ej3+7eifSzrxUjD4PhRHVkO2R4+b28a2uNwqyeKUVv\n3q4/GrE4z+nfCM544yCYhHiUNIcpZVQqcowMOreB1WDo35OrtJlnY9DdUQ135bNV\nMXu5ISVv2FVfR/YXf5wSBwhbm/pECz0sV3AL7gmAiUE0BdhTYkmIZ4XJSH0bpm4U\n3HuVm5lhAgMBAAECggEAEqlDr7P06N2hXo3q+QCcx7lcnCJvp11nJQVptPvCk8xe\nA54Q6g6WcURVaM07Txv/MFwFH9MpNfkq1kDpAC9TsNhxeT/119uCpAbFuR/zpNIP\nr6W/pbtVmSWwMOLqwhTMZsCrmNoRlcpotaIkupxQaqVQsz5aIDpTP+XOmmDJBsD4\naiVUDbpeUEewwZPRY7RQKAGwTKrIMZCtKmDsr6rbXTPofz+BPnK3Upqpt/dcgKga\n1tO7x1vXbZVMYgwE/ksand05Bla1NdJ7jV+vIcEBoBul6qW1S2U9j33sk0jBuI+p\nTpDKQLMlIsFYephmGFgAe2jdKYvxnwUmqZVgJMWggQKBgQDq+rar9M0Cge4qdk//\nwkXeJ+mxNbIuM5q4ky86G5bxLuT2D8A4SjFV9BVxHpayIzT4M52yCxMtuJbX+Zb9\nXJIoVqTUkCDok58F+vwY6mTwC4eLB1hxhi7XBR6DYQDOhwo12a/zynL7pHJkjs/g\nytZ+nG07Z9ogPN9Uy+Ru8PNHNQKBgQDaSKqlchm8+8WkH7hFMe8ZG6Sb4X9Yp+uK\nuIeQ9+qgf2EmrQ+HK4I91leydpPqKTmba/yrPOR8Tj8fnA7rGRyzHy4WqHXIbzF7\nXcN03X3+/cLzAzuuy+y+0AiO/miiHurOW/X2Ul2ejlAZi1bRlre1+gNTVqXtBL4j\nEXBthIqS/QKBgD4Lz5pePJfx6QqLTRpymPJNRbbGP/NVKwCb1LeaO2QaBtk2VYJH\njPluRw8kjZQiGcWEE5rEs965xBLpU8Y44FsIbeO23wmqmS4CFPkbQ2XjFXpPiToI\nvWuHbYQxY/4kyDxp670K8wuhY5dL4nYv+S1bbrhl9sHWcP46DGqC8yoVAoGBAKRy\nTEUaIbPTRcFwuCVBCi0zOx7IkmTbFMNMY61eaJ+Dd1Bo3qLpr1Qgz66+UI7/gcvK\nXe8vj77qP/nzWvXY1FtJqTIetaVLx852BBNd7lcVHDJyBBuau//AwEHh/jfs7N6M\nP0/UG32hH00vQTfiwQJSmQDG7XgTs569u6J2UOONAoGBALfHSPWRDndqAJBWK34P\nXi0JvYW5IudjTgkOlxcxgi3lfArgWScTlXwShOv7YnTYJa2AP4/MItuitBOhqdH4\nHI7yOy+O8WRu/Y6htVa1tDJUaSz4uK+muc26+BsDgsnVOMUEwKzQ20ANm7qDFHZ2\n6aMXgswKWomWTLrs6xvkyXMk\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-q121d@gpt52-88644.iam.gserviceaccount.com",
  "client_id": "105148527299493746893",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-q121d%40gpt52-88644.iam.gserviceaccount.com"
}

cred = credentials.Certificate(config)
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
