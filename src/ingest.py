import os
import shutil

# Silenciar aviso de symlinks da HF no Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings

# Constantes com os caminhos padrão do projeto
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def load_and_split_pdf(pdf_path: str, chunk_size: int = 250, chunk_overlap: int = 80) -> list:
    """
    Lê um arquivo PDF e divide seu conteúdo em chuncks
    """
    
    # Verifica se o arquivo existe no caminho informado
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Arquivo não encontrado no caminho: {pdf_path}")
    
    print(f"Carregando o documento: {pdf_path}")
    
    # Instancia o leitor de PDF e extrai as páginas
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Total de páginas lidas: {len(pages)}")
    
    # Configura o splitter de texto
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, # Tamamho máximo de caracteres de cada chunk
        chunk_overlap=chunk_overlap, # Quantos caracteres se repetem entre um chunk e o próximo
        separators=["\n\n", "\n", ".", " ", ""] # Ordem de preferência para cortar o texto
    )
    
    chunks = text_splitter.split_documents(pages)
    print(f"Documento dividido em {len(chunks)} chunks com sucesso.\n")
    
    return chunks

def save_to_chroma(chunks: list, clear_existing: bool = True) -> Chroma:
    """
    Transforma os chunks em embeddings e salva no banco ChromaDB local.
    """
    
    # Se clear_existing for True e a pasta do banco já existir, apagamos para não duplicar os dados de teste.
    if clear_existing and os.path.exists(CHROMA_PATH):
        print(f"Limpando banco de dados antigo em '{CHROMA_PATH}'...")
        shutil.rmtree(CHROMA_PATH)
        
    print("Inicializando modelo de embeddings...")
    
    # Carrega o modelo da HuggingFace que roda localmente na sua CPU
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True}
    )
    
    print("Gerando vetores e salvando no ChromaDB...")
    
    # Cria o banco a partir dos documentos e já salva automaticamente na pasta 'CHROMA_PATH'        
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH,
        client_settings=Settings(anonymized_telemetry=False)
    )

    print(f"Deu bom! {len(chunks)} chunks foram vetorizados e salvos em '{CHROMA_PATH}'.\n")
    
    return vector_db



# Bloco de teste
if __name__ == "__main__":
    caminho_pdf = "data/documento.pdf"
    
    meus_chunks = load_and_split_pdf(caminho_pdf)
    
    db = save_to_chroma(meus_chunks)
    
    pergunta_teste = "O STEC possui quantos tipos de atuadores?"
    print(f"Testando busca no banco para a pergunta: '{pergunta_teste}'\n")
    
    resultados = db.similarity_search_with_score(pergunta_teste, k=5) # os k chunks mais parecidos com a pergunta
    
    for i, (doc, score) in enumerate(resultados, start=1):
        pagina = doc.metadata.get("page", 0) + 1
        print(f"--- RESULTADO {i} | Página {pagina} | Distância (Score): {score:.4f} ---")
        print(doc.page_content)
        print("-" * 50 + "\n")
    
    