import os

# Silencia avisos no terminal
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ingest import CHROMA_PATH, get_embedding_model

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

PROMPT_TEMPLATE = """
Você é um assistente técnico especializado em analisar documentos acadêmicos e de engenharia.
Responda à pergunta do usuário baseado-se exclusivamente nos trechos de contexto fornecidos abaixo.
Se a informação não estiver no contexto, diga claramente que o documento não contém essa informação. Não tente adivinhar.

CONTEXTO EXTRAÍDO DO DOCUMENTO:
{contexto}

--- 
PERGUNTA DO USUÁRIO:
{pergunta}
"""

def get_vector_db() -> Chroma:
    """
    Conecta ao banco de dados que já foi salvo.
    """
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(
            f"Banco vetorial não encontrado em '{CHROMA_PATH}. Rode o script 'src/ingest.py' primeiro."
        )
        
    embedding_model = get_embedding_model()
    
    # Chroma direto para ler do disco local
    vector_db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
        client_settings=Settings(anonymized_telemetry=False)
    )
    return vector_db

def ask_question(pergunta: str, k: int = 5) -> dict:
    """
    Faz a busca semântica do banco e envia o contexto e a pergunta para o LLM via OpenRouter.
    Retorna um dicionário com a resposta da IA e a lista de páginas consultadas.
    """
    
    # Abre o banco e busca os k trechos mais relevantes
    db = get_vector_db()
    resultados = db.similarity_search_with_score(pergunta, k=k)
    
    # Junta os textos das chunks e guarda o número das págs
    trechos_formatados = []
    paginas_fontes = set() # set para nao repetir o numero de pags
    
    for doc, _ in resultados:
        num_pagina = doc.metadata.get("page", 0) + 1
        paginas_fontes.add(num_pagina)
        trechos_formatados.append(f"[Página {num_pagina}]: {doc.page_content}")
        
    # Junta todos os itens da lista separando por duas quebras de linha
    contexto_completo = "\n\n".join(trechos_formatados)
    
    # Preenche o template com o contexto e a pergunta
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt_final = prompt_template.format(
        contexto=contexto_completo,
        pergunta=pergunta
    )
    
    # Configuração da conexão com o OpenRouter
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Chave OPENROUTER_API_KEY não encontrada. Verifique seu arquivo '.env'.")
    
    llm = ChatOpenAI(
        model="openrouter/free", # Modelos gratuitos
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1 # Evitar criatividade e alucinações do modelo
    )
    
    # Envia o prompt para a IA e recebe a resposta
    resposta_ia = llm.invoke(prompt_final)
    
    # Retorna um dicionário organizado
    return{
        "resposta": resposta_ia.content,
        "fontes": sorted(list(paginas_fontes)),
        "chunks_usados": trechos_formatados
    }
    
# Teste
if __name__ == "__main__":
    
    my_question = "O STEC possui quantos tipos de atuadores e quais são eles?"
    print(f"Perguntado para a IA: '{my_question}'...\n")
    
    result = ask_question(my_question, k=6)
    
    print("=== Resposta da IA ===")
    print(result["resposta"])
    
    print(f"\n Página consultadas: {result['fontes']}")