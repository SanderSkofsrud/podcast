from dotenv import load_dotenv
import os

load_dotenv()


def get_openai_key():
    return os.getenv("AZURE_OPENAI_API_KEY")

def get_openai_endpoint():
    return os.getenv("AZURE_OPENAI_ENDPOINT")

def get_openai_version():
    return os.getenv("AZURE_OPENAI_VERSION")

def get_openai_name():
    return os.getenv("AZURE_OPENAI_NAME")

def get_whisper_endpoint():
    return os.getenv("WISPER_API_URL")

def get_whisper_api_key():
    return os.getenv("WISPER_API_KEY")

def get_whisper_api_version():
    return os.getenv("WISPER_API_VERSION")

def get_whisper_api_type():
    return os.getenv("WISPER_API_TYPE")

def get_whisper_deployment_id():
    return os.getenv("WISPER_DEPLOYMENT_ID")






from openai import AzureOpenAI

# Sett opp klienten
client = AzureOpenAI(
    api_key=get_openai_key(),  
    api_version=get_openai_version(),
    azure_endpoint=get_openai_endpoint(),  
    azure_deployment=get_openai_name()  
)

"""


# Example request
completion = client.chat.completions.create(
    model=get_openai_name(),  
    messages=[
        {"role": "system", "content": "Du er en tekstklassifikator som avgjør om en tekst er en annonse eller ikke."},
        {"role": "user", "content": "Skriv en kort annonse for et sommertilbud."}
    ],
    max_tokens=50
)

print(completion.choices[0].message.content)


"""









