import requests
import pandas as pd
#from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import os

#change model_id if diff model
model_id = "google/vit-base-patch16-224"
#hf_token unique to a write/POST request?
hf_token = "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"
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

def query(image):
    response = requests.post(api_url, headers = headers, json= {"inputs": image, "options":{"wait_for_model":True}})
    return response.json()

# f = open("listim.txt", "r")
# for x in f:
#     impath = './224/'
#     impath += x[0:-1]
#     image = Image.open(impath)
#     output = query(image)
#     embedding = pd.DataFrame(output, index = [x])
#     print(embedding)


impath = "/fast1/joya.debi/224/_400000-500000_Projects_477920_03_Elevations.jpg"
image = Image.open(impath)
output = query(image)
embedding = pd.DataFrame(output)
print(embedding)

# texts = ["How do I get a replacement Medicare card?",
#         "What is the monthly premium for Medicare Part B?",
#         "How do I terminate my Medicare Part B (medical insurance)?",
#         "How do I sign up for Medicare?",
#         "Can I sign up for Medicare Part B if I am working and have health insurance through an employer?",
#         "How do I sign up for Medicare Part B if I already have Part A?",
#         "What are Medicare late enrollment penalties?",
#         "What is Medicare and who can get it?",
#         "How can I get help with my Medicare Part A and Part B premiums?",
#         "What are the different parts of Medicare?",
#         "Will my Medicare premiums be higher because of my higher income?",
#         "What is TRICARE ?",
#         "Should I sign up for Medicare Part B if I have Veterans' Benefits?"]





#     feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
#     model = ViTForImageClassification.from_pretrained(model_name)

#     inputs = feature_extractor(images=image, return_tensors= "pt")
#     outputs = model(**inputs)
#     logits = outputs.logits
#     #model predicts one of the 1000 ImageNet classes
#     predicted_class_idx = logits.argmax(-1).item()
#     print(list_files[i], model.config.id2label[predicted_class_idx])




# api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
# headers = {"Authorization": f"Bearer {hf_token}"}

# def query(texts):
#     response = requests.post(api_url, headers = headers, json= {"inputs": texts, "options":{"wait_for_model":True}})
#     return response.json()

# texts = ["How do I get a replacement Medicare card?",
#         "What is the monthly premium for Medicare Part B?",
#         "How do I terminate my Medicare Part B (medical insurance)?",
#         "How do I sign up for Medicare?",
#         "Can I sign up for Medicare Part B if I am working and have health insurance through an employer?",
#         "How do I sign up for Medicare Part B if I already have Part A?",
#         "What are Medicare late enrollment penalties?",
#         "What is Medicare and who can get it?",
#         "How can I get help with my Medicare Part A and Part B premiums?",
#         "What are the different parts of Medicare?",
#         "Will my Medicare premiums be higher because of my higher income?",
#         "What is TRICARE ?",
#         "Should I sign up for Medicare Part B if I have Veterans' Benefits?"]

# output = query(texts)

# embeddings = pd.DataFrame(output)
# print(embeddings)