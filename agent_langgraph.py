from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

import warnings
warnings.filterwarnings("ignore")


load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Get the weather in a city"""
    return f"Il fait 22°C et ensoleillé à {city}."
@tool
def get_time(city: str) -> str:
    """Get the time in a city"""
    return f"Il est 14h32 à {city}."

# TODO 2 : instancie le modèle

model = ChatOpenAI(model="gpt-4o-mini")

# TODO 3 : crée l'agent
# create_react_agent prend deux arguments : le modèle et la liste de tools
# Une ligne. C'est tout.
agent = create_react_agent(model, [get_weather, get_time] )

# TODO 4 : invoque l'agent
# La signature est différente de ce que tu connais :
# agent.invoke({"messages": [("user", "ta question")]})
# Stocke le résultat dans result


# result = agent.invoke({"messages": [("user", "Quel temps fait-il à Tel Aviv et quelle heure est-il à Paris ?")]})


# TODO 5 : print la réponse finale
# result est un dict avec une clé "messages" — liste de tous les messages
# Le dernier message est la réponse finale du modèle
# result["messages"][-1].content
# print(result["messages"][-1].content)


# --- Chunk 2 : mémoire ---

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


# TODO 1 : crée un checkpointer
with SqliteSaver.from_conn_string("memory.db") as memory:

    # TODO 2 : recrée l'agent avec le checkpointer
    # create_react_agent accepte un 3ème argument : checkpointer=
    agent = create_react_agent(model, [get_weather, get_time], checkpointer=memory)

    # TODO 3 : invoque avec un thread_id
    # Sans checkpointer, on faisait agent.invoke({"messages": [...]})
    # Avec checkpointer, il faut passer un config :
    config = {"configurable": {"thread_id": "thread-1"}}

    result1 = agent.invoke(
        {"messages": [("user", "Quel temps fait-il à Tel Aviv ?")]},
        config=config
    )
    print(result1["messages"][-1].content)

    # TODO 4 : second appel dans le MÊME thread
    # Pose une question de suivi qui nécessite le contexte du premier appel :
    # "Et à Paris ?"  — sans répéter "quel temps"
    # Si la mémoire fonctionne, l'agent comprend que tu parles encore de météo

    result2 = agent.invoke(
        {"messages": [("user", "Et à Paris ?")]},
        config=config
    )
    print(result2["messages"][-1].content)