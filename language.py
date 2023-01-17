########################################################
#
#   Project: Language Detection FastAPI
#   Author: Maharshi Parekh
#   Contact: maharshig.parekh2014@gmail.com
#   Description: RestAPI developed in FastAPI, language model sourced from HuggingFace, predicts what Language input was provided in.
#   References: HuggingFace model card - https://huggingface.co/papluca/xlm-roberta-base-language-detection
#
########################################################

from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

# load the pre-trained language model
language_model = "papluca/xlm-roberta-base-language-detection"

# setting up pipeline for language detection
pipe = pipeline("text-classification", model = language_model)

app = FastAPI(
    title="To language or not to language, that is the question",
    version="1.0.0",
    description="Predicts which language has been inputted and outputs language and confidence score in json format."
)

# microservice - request models
class TextInput(BaseModel):
    text: str

# microservice - response models
class LanguageDetectionResponse(BaseModel):
    status: bool
    message: Optional[str]
    predicted_language: Optional[str]
    predicted_language_score: Optional[float]

    class Config:
        schema_extra = {
            "example": {
                "status": True,
                "message": None,
                "predicted_language": "en",
                "predicted_language_score": "0.9998"
            }
        }


# microservice - endpoints
@app.post("/detectLanguage", response_model=LanguageDetectionResponse, tags=["Language Operations"])
def predict_language(params:TextInput):
    """
    Detects and returns language by calling pre-trained language model
    """
    status = False
    message = None
    predicted_language = None
    predicted_language_confidence = None

    try:
        # reading input
        data_text = params.text

        # input validations
        if len(data_text) > 128 or len(data_text) == 0:
            message = "Current API handles text length of max 128 characters!"
        else:
            prediction = pipe(data_text, top_k=1, truncation=True, max_length=128)
            
            if prediction:
                predicted_language = prediction[0]["label"]
                predicted_language_confidence = prediction[0]["score"]
                status = True
    
    except Exception as ex:
        message = str(ex)
        raise HTTPException(status_code=500, detail=message)
    
    return {
        "message": message, 
        "predicted_language": predicted_language, 
        "predicted_language_score": predicted_language_confidence, 
        "status": status
    }

if __name__=="__main__":
    uvicorn.run(app,host="127.0.0.1",port=8080)