import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdf(pdf_path: str) -> list:
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
        chunk_size = 600, # Tamamho máximo de caracteres de cada chunk
        chunk_overlap = 100, # Quantos caracteres se repetem entre um chunk e o próximo
        separators=["\n\n", "\n", ".", ",", ""] # Ordem de preferência para cortar o texto
    )
    
    chunks = text_splitter.split_documents(pages)
    print(f"Documento dividido em {len(chunks)} chunks com sucesso.\n")
    
    return chunks

# Bloco de execução principal para teste
if __name__ == "__main__":
    caminho_pdf = "data/documento.pdf"
    
    meus_chunks = load_and_split_pdf(caminho_pdf)
    
    primeiro_chunk = meus_chunks[0]
    
    print("=== CONTEÚDO DO CHUNK 0 ===")
    print(primeiro_chunk.page_content)
    
    print("\n=== METADADOS DO CHUNK 0 (Dicionário Python) ===")
    print(primeiro_chunk.metadata)