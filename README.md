# Agents Playground

Bienvenue dans **Agents Playground** ! Ce projet est un espace d'apprentissage progressif pour comprendre comment construire et utiliser des agents d'IA avec des appels d'outils (Tool Calling).

L'évolution se fait en 3 étapes, allant d'une implémentation "from scratch" avec l'API d'OpenAI, jusqu'à l'utilisation de frameworks avancés comme LangChain et LangGraph.

## 🚀 Les 3 approches

### 1. `agent.py` : Vanilla OpenAI
Ce script montre comment construire un agent en utilisant uniquement le SDK officiel `openai`. 
- Définition manuelle du schéma JSON des outils.
- Boucle `while` manuelle pour gérer les appels d'outils (`tool_calls`).
- Exécution des fonctions Python et renvoi des résultats au modèle.

### 2. `agent_langchain.py` : L'approche LangChain
Ici, on introduit **LangChain** pour simplifier la définition et la liaison des outils.
- Utilisation du décorateur `@tool` pour générer automatiquement le schéma JSON depuis la signature et la docstring de la fonction.
- Utilisation de `ChatOpenAI` et `bind_tools()`.
- La boucle d'exécution reste manuelle pour bien comprendre la mécanique sous-jacente.

### 3. `agent_langgraph.py` : L'approche LangGraph (avec mémoire)
L'étape finale avec **LangGraph** pour un agent complètement géré et doté d'une mémoire.
- Utilisation de `create_react_agent` qui gère automatiquement la boucle d'exécution des outils.
- Ajout de `SqliteSaver` pour sauvegarder l'état de la conversation (mémoire persistante) via une base de données locale (`memory.db`), permettant à l'agent de se souvenir du contexte entre différentes questions (ex: "Quel temps fait-il à Tel Aviv ?" -> "Et à Paris ?").

## 🛠️ Installation et prérequis

1. Clonez ce dépôt.
2. Créez un environnement virtuel (optionnel mais recommandé) :
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
4. Créez un fichier `.env` à la racine du projet et ajoutez votre clé API OpenAI :
   ```env
   OPENAI_API_KEY=votre_cle_api_ici
   ```

## ▶️ Exécution

Vous pouvez lancer chaque script indépendamment pour observer son fonctionnement :

```bash
python agent.py
python agent_langchain.py
python agent_langgraph.py
```
