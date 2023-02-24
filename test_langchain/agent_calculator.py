# Import things that are needed generically
from langchain.agents import initialize_agent, Tool
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper

llm = OpenAI(temperature=0)

llm_math_chain = LLMMathChain(llm=llm)


class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "useful for when you need to answer questions about math"

    def _run(self, query: str) -> str:
        """Use the tool."""
        return llm_math_chain.run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")


tools = [CustomCalculatorTool()]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
agent.run("What is 2 + 3 ?")

# 运行错误
