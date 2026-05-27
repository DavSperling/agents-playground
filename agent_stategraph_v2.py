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
    return f"je n'ai pas compris la question ."

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


tools = [get_weather, calculatrice]
tools_by_name = {t.name: t for t in tools} 

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict): 
    messages : Annotated[list, add_messages]
    iteration : int 

def agent_node(state: State) -> dict:
    last_iteration = state["iteration"]
    result = llm_with_tools.invoke(state["messages"])
    return {"messages": [result],
            "iteration" :last_iteration+1 }

def tools_node(state: State) -> dict:
    last_message = state["messages"][-1]
    last_iteration = state["iteration"]
    results = []
    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        rep = tool.invoke(tool_call["args"])
        results.append(ToolMessage(content = rep, tool_call_id=tool_call["id"]))
    return {"messages": results,
        "iteration" :last_iteration+1}
        
def should_continue(state: State) -> str:
    last_message = state["messages"][-1]
    #last_message.tool_calls == [] or
    if  state["iteration"] >= 5:
        return END
    return "tools"


graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)
graph.add_edge(START, "agent")              
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
app = graph.compile()


message = "météo à Tel Aviv et 18*47 ?"
result = app.invoke({"messages": [("user",message)],    "iteration": 0})
for m in result["messages"]: m.pretty_print()
print(result["iteration"])