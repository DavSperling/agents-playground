from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import numexpr as ne
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import ToolMessage
from langgraph.graph.message import add_messages
import os


load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"



@tool
def get_weather(city: str) -> str:
    """Get the weather in a city"""
    return f"Il fait 22°C et ensoleillé à {city}."

@tool
def calculatrice(expression: str) -> str:
    """
    Utile pour évaluer des expressions mathématiques et faire des calculs précis.
    L'entrée doit être une expression mathématique valide sous forme de chaîne de caractères.
    Exemples valides : '200 * 0.15', '(1500 / 3) + 42', '2**10'.
    """
    # Nettoyage basique au cas où le LLM ajoute des espaces ou des guillemets inutiles
    expression = expression.strip(" '\"`")
    
    try:
        # Évaluation sécurisée de l'expression mathématique
        result = ne.evaluate(expression)
        
        # On convertit le résultat (souvent un numpy array scalaire) en float puis en string
        return str(float(result))
    
    except Exception as e:
        return f"Erreur de syntaxe dans le calcul. Dis à l'utilisateur que l'expression '{expression}' n'a pas pu être calculée. Détail technique : {str(e)}"


# ===== Tes 2 tools (get_weather, calculatrice) déjà définis =====

tools = [get_weather, calculatrice]
tools_by_name = {t.name: t for t in tools}  # tu vas comprendre l'intérêt dans le node "tools"

# ===== Setup LLM =====
# instancier ChatOpenAI (gpt-4o-mini, temp=0) ET le bind aux tools
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

# ===== Définition du State =====
# TODO 3 : TypedDict avec un seul champ "messages" annoté avec add_messages
class State(TypedDict): 
    messages : Annotated[list, add_messages]
 

# ===== Node 1 : agent =====
# Il prend le state, appelle le LLM avec TOUT l'historique, renvoie la réponse.
# TODO 4 :
def agent_node(state: State) -> dict:
    result = llm_with_tools.invoke(state["messages"])
    return {"messages": [result]}


# ===== Node 2 : tools =====
# Il regarde le DERNIER message de l'historique (qui doit être un AIMessage avec tool_calls),
# exécute chaque tool_call, et renvoie autant de ToolMessage que de tool_calls.
# TODO 5 :
def tools_node(state: State) -> dict:
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        rep = tool.invoke(tool_call["args"])
        results.append(ToolMessage(content = rep, tool_call_id=tool_call["id"]))
    return {"messages": results}

# ===== Edge conditionnel : should_continue =====
# Regarde le dernier message du state. Renvoie une string :
# - "tools" si le dernier message a des tool_calls
# - END sinon
# TODO 6 :
def should_continue(state: State) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls == [] :
        return END
    return "tools"

# ===== Construction du graph =====
# TODO 7 :
graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)
graph.add_edge(START, "agent")              
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
app = graph.compile()

# ===== Test =====
# TODO 8 :
# Lance une question qui force l'utilisation des DEUX tools
# Ex : "Quelle est la météo à Tel Aviv et combien fait 18 * 47 ?"
result = app.invoke({"messages": [("user", "Quelle est la météo à Tel Aviv et combien fait 18 * 47 ?")]})
for m in result["messages"]: m.pretty_print()
