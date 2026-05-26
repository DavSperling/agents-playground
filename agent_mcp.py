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
  
    "tavily-mcp": {
      "command": "npx",
      "transport": "stdio",
      "args": ["-y", "tavily-mcp@latest"],
      "env": {
        "TAVILY_API_KEY": os.environ["TAVILY_API_KEY"],
        "DEFAULT_PARAMETERS": "{\"include_images\": true, \"max_results\": 15, \"search_depth\": \"advanced\"}"
      }},
      "filesystem": {
            "command": "npx",
            "args": ["@modelcontextprotocol/server-filesystem", "/Users/david/Documents/Python/FORMATION_IA/github/agents-playground"],
            "transport": "stdio",
        }
  
}

    client = MultiServerMCPClient(config)
    # Récupère la liste des tools exposés par le serveur
    tools = await client.get_tools()
    #print([t.name for t in tools])  # juste pour vérifier

    # TODO : créer l'agent avec ChatOpenAI gpt-4o-mini + ces tools
    model = ChatOpenAI(model="gpt-4o-mini")

    agent =  create_react_agent(model, tools)

    # TODO : invoquer avec "List the files in the current directory"
    result =await  agent.ainvoke({"messages": [("human", "cherche les actus du jour et écris un résumé dans un fichier résumé.txt")]} )
    print(result)

asyncio.run(main())