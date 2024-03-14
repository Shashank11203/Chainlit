from langchain_community.llms import OpenAI
from langchain.chains import LLMChain

from langchain.memory.buffer import ConversationBufferMemory

from langchain.prompts import PromptTemplate

ice_cream_assistant_template = """
You are an food assistant chatbot named "". Your expertise is 
exclusively in providing information and advice about anything related to 
food. This includes food combinations, ice foods recipes, and general
food queries. You do not provide information outside of this 
scope. If a question is not about food, respond with, "I specialize 
only in food related queries."
Chat History: {chat_history}
Question: {question}
Answer:"""

ice_cream_assistant_prompt_template = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=ice_cream_assistant_template
)


import chainlit as cl

@cl.on_chat_start

def quey_llm():
    elements = [
    cl.Image(name="image1", display="inline", path="Logo.jpeg")
    ]
    cl.Message(content="Hello there, Welcome to AskAnyQuery related to Data!", elements=elements)

    llm = OpenAI(model='gpt-3.5-turbo-instruct',
                 temperature=0)
    
    conversation_memory = ConversationBufferMemory(memory_key="chat_history",
                                                   max_len=50,
                                                   return_messages=True
                                                   )
    llm_chain = LLMChain(llm=llm, 
                         prompt=ice_cream_assistant_prompt_template,
                         memory=conversation_memory)
    
    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def query_llm(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")
    
    response = await llm_chain.acall(message.content, 
                                     callbacks=[
                                         cl.AsyncLangchainCallbackHandler()])
    
    await cl.Message(response["text"]).send()