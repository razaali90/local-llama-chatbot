
import chainlit as cl
from llama_llm import llama_llm
import torch
import gc

# llm_chain = llama_llm()


@cl.on_message
async def query_llm(message: cl.Message):
    
    llm_chain = cl.user_session.get("llm_chain")
    response = await llm_chain.acall(message.content, 
                                     callbacks=[
                                         cl.AsyncLangchainCallbackHandler()])
    
    await cl.Message(response["text"]).send()

    # to handle cuda out of memory error
    torch.cuda.empty_cache()
    gc.collect()

@cl.on_chat_start
def query_llm():
    llm_chain = llama_llm()
    cl.user_session.set("llm_chain", llm_chain)
    print(f"Current user is: {cl.user_session.get('llm_chain')}")

