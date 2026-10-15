import os
from functools import lru_cache

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

@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Carrega o modelo de embeddings apenas na primeira vez e mantém na memória RAM.
    """
    print("Inicializando modelo de embeddings na memória...")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True}
    )

def load_and_split_pdf(pdf_path: str, chunk_size: int = 250, chunk_overlap: int = 80) -> list:
    """
    Lê um arquivo PDF e divide seu conteúdo em chuncks
    """
    
    # Verifica se o arquivo existe no caminho informado
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Arquivo não encontrado no caminho: {pdf_path}")
    
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"O overlap deve ser menor que o tamanho do chunk."
        )
    
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
    
    if not chunks:
        raise ValueError("Nenhum texto extraído. Verifique se o PDF não é apenas imagem escaneada.")
    
    print(f"Documento dividido em {len(chunks)} chunks com sucesso.\n")
    
    return chunks

def save_to_chroma(chunks: list, clear_existing: bool = True) -> Chroma:
    """
    Transforma os chunks em embeddings e salva no banco ChromaDB local.
    """
    # Pega o modelo que já está na memória RAM
    embedding_model = get_embedding_model()
    
    # Conecta ao banco ChromaDB existente
    vector_db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
        client_settings=Settings(anonymized_telemetry=False)
    )
    
    # Se clear_existing for True, pede para o banco esvaziar os dados antigos
    if clear_existing:
        print("Limpando dados antigos da coleção no ChromaDB...")
        vector_db.reset_collection()
    
    # Insere os novos chunks
    print("Gerando vetores e salvando no ChromaDB...")
    vector_db.add_documents(documents=chunks)

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
    
    