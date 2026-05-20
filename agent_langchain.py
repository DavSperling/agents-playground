from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage


load_dotenv()

# TODO 1 : définir get_weather comme LangChain tool
# Le décorateur @tool fait le JSON Schema automatiquement —
# plus besoin de l'écrire à la main comme avant.
# La docstring devient la "description" du tool.
@tool
def get_weather(city: str) -> str:
    """Get the weather in a city"""
    return f"Il fait 22°C et ensoleillé à {city}."
@tool
def get_time(city: str) -> str:
    """Get the time in a city"""
    return f"Il est 14h32 à {city}."

# TODO 2 : instancier le modèle
# ChatOpenAI remplace ton client + model="gpt-4o-mini"
model = ChatOpenAI(model="gpt-4o-mini")

# TODO 3 : binder le tool au modèle
# model.bind_tools([...]) — équivalent de ton tools= dans l'appel API
model_with_tools = model.bind_tools([get_weather, get_time])

# TODO 4 : faire un appel
# model_with_tools.invoke([...])
# Le message se construit avec HumanMessage ou juste une string
# response = model_with_tools.invoke("Quel temps fait-il à Paris et quelle est l'heure à Londres ?")

    
# tool_result = response.tool_calls[0]["args"]

# tool_message = ToolMessage(get_weather.invoke(tool_result["city"]), tool_call_id=response.tool_calls[0]["id"])

# final_response = model_with_tools.invoke(["Quel temps fait-il à Paris et quelle est l'heure à Londres ?",response,tool_message])

message = ["Quel temps fait-il à Paris et quelle est l'heure à Londres ?"]

tool_functions={
    "get_weather": get_weather,
    "get_time": get_time
}

"""
De manière simple ici on boucle uniquement pour utiliser des tools, les tools sont utilisé 
pour fournir des information supplementaire a l'agent. 

Avant la boucle je crée un tool_functions pour associer les tools aux fonctions python.

Dans le while : 
- on appelle le modèle
- on ajoute la réponse à message
- on vérifie si le finish_reason est stop -> si oui on break
- si finish_reason est tool_calls -> on boucle sur les tool_calls
- on récupère le nom de la fonction et les arguments
- on appelle la fonction correspondante
- on ajoute la réponse à message

Jusqu'a qu'on est plus de tool a utiliser et du coup le finish_reason est stop.

"""
while True:
    response = model_with_tools.invoke(message)
    message.append(response)

    finish_reason = response.response_metadata['finish_reason']

    if finish_reason == "stop":
        break

    if finish_reason == "tool_calls":
        for tool_call in response.tool_calls:
            function_name = tool_call["name"]
            function_args = tool_call["args"]
            
            if function_name in tool_functions:
                function_response = tool_functions[function_name].invoke(function_args)
                message.append(ToolMessage(function_response, tool_call_id=tool_call["id"]))
            else:
                break

# TODO 5 : print response et response.tool_calls
print(response.content)
