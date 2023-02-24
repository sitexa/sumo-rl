# Import things that are needed generically
from langchain.agents import initialize_agent, Tool
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper

llm = OpenAI(temperature=0)

llm_math_chain = LLMMathChain(llm=llm)
tools = [
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
        return_direct=True
    )
]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

agent.run("what's 2x5")

# 运行错误