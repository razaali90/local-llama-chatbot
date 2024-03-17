import os
os.environ['CURL_CA_BUNDLE'] = ''

# !nvidia-smi

"""Install the libraries"""

# !pip install -q langchain transformers accelerate bitsandbytes

"""Load the libraries"""

from langchain.chains import LLMChain, SequentialChain
# from langchain.memory import ConversationBufferMemory
from langchain.memory.buffer import ConversationBufferMemory
from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain

# Trying new prompt style
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage

from transformers import pipeline
from transformers import AutoModel
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

import json
import textwrap


tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")

model = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-chat-hf",
                                            device_map='auto',
                                            torch_dtype=torch.float16,
                                            load_in_4bit=True,
                                            bnb_4bit_quant_type="nf4",
                                            bnb_4bit_compute_dtype=torch.float16)


def llama_llm():    
    
    """Define Transformers Pipeline which will be fed into Langchain"""

    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer= tokenizer,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    max_new_tokens = 512,
                    do_sample=True,
                    top_k=30,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id
                    )

    """Define the Prompt format for Llama 2 - This might change if you have a different model"""

    system_prompt = """You are a helpful assistant, you will use the provided context to answer user questions.
    Read the given context before answering questions and think step by step. If you can not answer a user question based on 
    the provided context, inform the user. Do not use any other information for answering user. Provide a detailed answer to the question."""

    def get_prompt_template(system_prompt=system_prompt, promptTemplate_type=None, history=False):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        instruction = """
        Context: {history} \n
        User: {question}"""

        prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
        prompt = PromptTemplate(input_variables=["history", "question"], template=prompt_template)

        memory = ConversationBufferMemory(input_key="question", memory_key="history", max_len=5)

        return (
            prompt,
            memory,
        )


    """Defining Langchain LLM"""

    # llm = HuggingFacePipeline(pipeline = pipe, batch_size=16, model_kwargs = {'temperature':0.7,'max_length': 256, 'top_k' :50})
    llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':0.7,'max_length': 256, 'top_k' :50})

    prompt, memory = get_prompt_template(system_prompt)

    # llm_chain = LLMChain(prompt=prompt, llm=llm, verbose = False, memory=conversation_memory)
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose = False, memory=memory)

    return llm_chain