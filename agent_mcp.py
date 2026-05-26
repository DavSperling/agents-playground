import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os


load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

async def main():
    # MultiServerMCPClient prend un dict : nom_serveur → config
    # config stdio a besoin de 3 clés : command, args, transport
    config = {
        "filesystem": {
            "command": "npx",
            "args": ["@modelcontextprotocol/server-filesystem", "/Users/david/Documents/Python/FORMATION_IA/github/agents-playground"],
            "transport": "stdio",
        }
    }

    client = MultiServerMCPClient(config)
    # Récupère la liste des tools exposés par le serveur
    tools = await client.get_tools()
    print([t.name for t in tools])  # juste pour vérifier

    # TODO : créer l'agent avec ChatOpenAI gpt-4o-mini + ces tools
    model = ChatOpenAI(model="gpt-4o-mini")

    agent =  create_react_agent(model, tools)

    # TODO : invoquer avec "List the files in the current directory"
    result =await  agent.ainvoke({"messages": [("human", "List the files in the current directory")]} )
    print(result)

asyncio.run(main())