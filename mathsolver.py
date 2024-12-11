from langchain.chains import LLMMathChain,LLMChain
from langchain.agents import initialize_agent,AgentType,Tool
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_icon="🧮",page_title="Solving Maths With Bot")
st.title("Solving Maths with Bot 🧮")

wiki_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki_tool=Tool(
    name="Wikipedia",
    description="search the content information from Wikipedia",
    tool_type="search",
    func=wiki_wrapper.run
)

apikey=st.sidebar.text_input("Enter you Groq API Key",type="password")
if not apikey:
    st.error("Please enter your Groq API Key")

llm=ChatGroq(api_key=apikey,model="gemma2-9b-it")

mathchain=LLMMathChain.from_llm(llm=llm)
mathsolver=Tool(
    name="Math Solver",
    description=" To solve the maths problems use this tool",
    func=mathchain.run
)

template='''
you are a bot used to solve the maths problems and also used as search agent to ask question relevant to maths so give the best answer to the question,
Question:{question}
'''
prompts=PromptTemplate(input_variables=['question'],template=template)

chain=LLMChain(llm=llm,prompt=prompts)
chain_tool=Tool(
    name="LLM Chain",
    func=chain.run,
     description="A tool for answering logic-based and reasoning questions."
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[{'role':'assistant','content':'Hello,I am a chatbot used for solving maths problem and also searching agent '}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

tools=[wiki_tool,mathsolver,chain_tool]
question=st.text_area("Enter your query",placeholder="I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

solver_agent=initialize_agent(llm=llm,prompt=prompts,tools=tools,type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

if st.button("Search the Question"):
    with st.spinner("finding the answer"):
        with st.chat_message("assistant"):
            st.session_state.messages.append({'role':'user',"content":question})
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=solver_agent.run(st.session_state.messages,callbacks=[st_cb])
            st.session_state.messages.append({'role':'assistant','content':response})
            st.success(response)
        


    
