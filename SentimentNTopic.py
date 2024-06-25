import pandas as pd
import time
import random
import re
import os
import math
import tiktoken
from openai import OpenAI
from multiprocessing import Pool

client = OpenAI(api_key='')
MODELS = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k","gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125"]

OUTPUT_PATH    = "classified/"
SENTIMENT_PATH = OUTPUT_PATH + "sentiments"
TOPIC_PATH = OUTPUT_PATH + "topics"

SENTIMENT_PROMPT = "Quiero que realices un análisis de sentimientos para cada elemento y quiero me devuelvas un rango del 1 al 5, donde 1 es un comentario negativo y 5 es un comentario positivo. Solo necesito los números separados por saltos de linea y enumerados de la forma <index>.<sentimiento>, sin texto adicional, el contexto son las elecciones presidenciales de Mexico 2024, el texto fue lematizado y se redujo la variabilidad lexica, solo devuelveme la lista de elementos no me des ningun texto extra. Aquí está el texto: "
TOPIC_PROMPT     = "Voy a proporcionarte un texto lematizado y reducido. Quiero que identifiques los tópicos tratados en el texto y me los enumeres, proporcionando una lista detallada de los temas mencionados. El contexto son las elecciones en México de 2024. Detalla los topicos entendiendo el contexto que te pase. \n Texto: \n"

MAX_TOKENS = 13000
MAX_ATTEMPTS = 5

INPUT_FILES = [
    "data/1erTrimestre/Claudia.csv",
    "data/1erTrimestre/Maynez.csv",
    "data/1erTrimestre/Xochitl.csv",
    "data/2doTrimestre/Claudia.csv",
    "data/2doTrimestre/Maynez.csv" ,
    "data/2doTrimestre/Xochitl.csv"
    ]

OUTPUT_SENTIMENT_FILES = [
    "1erTrimestre_Claudia.csv",
    "1erTrimestre_Maynez.csv",
    "1erTrimestre_Xochitl.csv",
    "2doTrimestre_Claudia.csv",
    "2doTrimestre_Maynez.csv",
    "2doTrimestre_Xochitl.csv"
]

OUTPUT_TOPIC_FILES = [
    "1erTrimestre_Claudia.csv",
    "1erTrimestre_Maynez.csv",
    "1erTrimestre_Xochitl.csv",
    "2doTrimestre_Claudia.csv",
    "2doTrimestre_Maynez.csv",
    "2doTrimestre_Xochitl.csv"
]

def sentiment_analysis(text):
    try: 
        answer = client.chat.completions.create(model=random.choice(MODELS),
            messages=[
                {"role": "system", "content": "Eres un sistema de analisis de texto"},
                {"role": "user", "content": f"{SENTIMENT_PROMPT} \n {text}"}
            ],
            temperature=0)

        content = answer.choices[0].message.content
        raw_response = content.strip()

        # Validation
        try:
            response = raw_response.split("\n") # Separate in lines
            response = [re.search("\\d[.] ?(\\d)", element).group(1) for element in response] # Delete the index (1. )
        except:
            print(f"Error en el formato de GPT")

        if (len(response) == len(text.split("\n"))): # Check length of the response
            if (all([(str)(element).isdigit() for element in response])): # Check that all values are digits
                return response
    except Exception as e:
        print(f"Error during API call (RPM exceded)")
        time.sleep(30) # only 3 RPM (Request per min)
    
    return None

def processChunk(chunk, i_prev, i, sentiment=True):
    for attempt in range(MAX_ATTEMPTS): # Attempts
        if sentiment:
            text = chunk.values
            textWIndex = "\n".join([f"{index+1}. {element}" for (index, element) in enumerate(text)])
            results = sentiment_analysis(textWIndex)
        else:
            results = topic_analysis(chunk)

        if results:
            print(f"Successfully classified {i_prev} - {i}")
            return i_prev, i, results
        else:
            print(f"Failed answer from GPT [{i_prev} - {i}], attempt:{attempt}")

def sentiments(filename, output):
    df = pd.read_csv(filename)
    len_df = len(df)
    
    perIteration = 200

    print(f"<=== Start Sentiment Classification with the file: {filename}, total: {len(df)} ====>")

    chunks = [(df["0"].iloc[i:i+perIteration], i, min(i+perIteration, len_df), True) for i in range(0, len_df, perIteration)]
    
    # Sentiment
    with Pool() as pool:
        results = pool.starmap(processChunk, chunks)
    
    if (results):
        for i_prev, i, sentiments in results:
            if sentiments:
                df.loc[i_prev:i-1, "Sentiment"] = sentiments
            else:
                print(f"Failed to classify sentiment {i_prev} - {i} in file {filename}")

    # Write in file
    if not os.path.exists(SENTIMENT_PATH):
        os.makedirs(SENTIMENT_PATH)

    df.to_csv(f"{SENTIMENT_PATH}/{output}", index=False)
    print(f"<=== End Sentiment Classification with the file: {filename} ====>")

def topic_analysis(text):
    try: 
        answer = client.chat.completions.create(model=random.choice(MODELS),
            messages=[
                {"role": "system", "content": "Eres un sistema de analisis de texto"},
                {"role": "user", "content": f"{TOPIC_PROMPT} \n {text}"}
            ],
            temperature=0.8)

        content = answer.choices[0].message.content
        raw_response = content.strip()

        # Validation
        try:
            response = re.findall(r"\d+\.\s(.*?)\n", raw_response)
            return response
        except:
            print(f"Error en el formato de GPT")

    except Exception as e:
        print(f"Error during API call (RPM or TPM exceded)")
        time.sleep(30) # only 3 RPM (Request per min), 40 000 TPM (Tokens per min)
    
    return None

def getNumOfTokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def topics(filename, output):
    df = pd.read_csv(filename)
    text = df["0"].str.cat().split()

    # Get average nums of tokens & max words
    average_words_per_token = (len(text) / getNumOfTokens(" ".join(text))) - 0.01
    max_words = math.floor(MAX_TOKENS * average_words_per_token)

    # Get divider
    divider = math.ceil(len(text) / max_words)
    per_iter = math.ceil(len(text) / divider)
    
    print(f"<=== Start Sentiment Classification with the file: {filename}, total: {len(text)} words ====>")

    chunks = [(" ".join(text[i:i+per_iter]), i, min(i+per_iter, len(text)), False) for i in range(0, len(text), per_iter)] 
    # Topic
    with Pool() as pool:
        results = pool.starmap(processChunk, chunks)
    
    topics_arr = []

    if (results):
        for i_prev, i, topics in results:
            if topics:
                topics_arr+=topics
            else:
                print(f"Failed to classify sentiment {i_prev} - {i} in file {filename}")
    
    # Delete duplicates
    topics_arr = list(set(topics_arr))

    # Enumerate
    topics = "\n".join([f"{index+1}. {element}" for (index, element) in enumerate(topics_arr)])

    # Write in file
    if not os.path.exists(TOPIC_PATH):
        os.makedirs(TOPIC_PATH)

    path = f"{TOPIC_PATH}/{output}"

    # Delete file if exists
    if os.path.exists(path):
        os.remove(path)
    
    with open(path, "x") as file:
        file.write(topics)
        file.close()

    print(f"<=== End Topic Classification with the file: {filename} ====>")

# Sentiments classification
# for (index, input) in enumerate(INPUT_FILES):
#     sentiments(input, OUTPUT_SENTIMENT_FILES[index])

# Topic Modeling
for (index, input) in enumerate(INPUT_FILES):
    topics(input, OUTPUT_TOPIC_FILES[index])

