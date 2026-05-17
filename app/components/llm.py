from langchain_huggingface import HuggingFaceEndpoint
from app.config.config import HF_TOKEN,HUGGINGFACE_REPO_ID
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(hugging_face_repo_id:str=HUGGINGFACE_REPO_ID,hf_token:str = HF_TOKEN):
    try:
        logger.info('Loading LLM from Hugging Face')
        llm = HuggingFaceEndpoint(
                repo_id=hugging_face_repo_id,
                # Generation parameters are passed directly as arguments, not inside model_kwargs
                temperature=0.3,
                max_new_tokens=256,  # Use max_new_tokens instead of max_length
                return_full_text=False,
                # Note the updated parameter name for the API token
                huggingfacehub_api_token=hf_token 
            )
        logger.info('LLM Loaded Successfully')
        return llm
    except Exception as e:
        error_message = CustomException("Failed to load LLM",e)
        logger.error(str(error_message))

        #pip install -qU langchain-huggingface huggingface_hub