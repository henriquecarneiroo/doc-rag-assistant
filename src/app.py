import os
import streamlit as st

from ingest import load_and_split_pdf, save_to_chroma, CHROMA_PATH
from rag_engine import ask_question

# Configuração da aba do navegador
st.set_page_config(
    page_title="DocQA - Assistente RAG",
    page_icon=None,
    layout="wide"
)

st.title("Assistente inteligente de documentos")
st.markdown("Faça upload de um PDF na barra lateral e tire suas dúvidas com base exclusiva com base no conteúdo dele.")

# Barra lateral para upload de documentos e ajustes de parâmetros do RAG
with st.sidebar:
    st.header("Configurações e Uploads")
    
    arquivo_pdf = st.file_uploader("Selecione um arquivo PDF", type=["pdf"])
    
    # SLiders para ajustar o tamanho do chunk e k
    st.subheader("Parâmetros do RAG")
    chunk_size = st.slider(
        "Tamanho do chunk (caracteres)",
        min_value=50, max_value=2000, value=300, step=10,
        help="Chunks menores são mais precisos para frases curtas; maiores mantêm mais contexto."
    )
    
    chunk_overlap = st.slider(
        "Overlap entre chunks",
        min_value=0, max_value=300, value=60, step=10,
        help="Quantidade de caracteres repetidos entre um chunk e o seguinte."
    )
    
    k_chunks = st.slider(
        "Quantidade de trechos buscados (k)",
        min_value=1, max_value=20, value=5, step=1,
        help="Quantos chunks o ChromaDB vai enviar para a IA ler."
    )
    
    botao_processar = st.button("Processar documento", use_container_width=True)
    
    if botao_processar:
        if arquivo_pdf is not None:
            with st.spinner("Lendo PDF, fatiando em chunks e gerando embeddings..."):
                # Garante que a pasta existe e salva o documento la
                os.makedirs("data", exist_ok=True)
                caminho_temp = os.path.join("data", "documento_atual.pdf")
                
                with open(caminho_temp, "wb") as f:
                    f.write(arquivo_pdf.getbuffer())
                    
                chunks = load_and_split_pdf(
                    caminho_temp,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                save_to_chroma(chunks, clear_existing=True)
                
                st.success(f"Sucesso! Documento dividido em **{len(chunks)} chunks** e salvo no banco.")
        else:
            st.warning("Por favor, selecione um arquivo PDF primeiro.")
            
# Inicializa historico de conversa se ainda nao existir
if "historico_mensagens" not in st.session_state:
    st.session_state.historico_mensagens = []
    
# Mostra todas as mensagens que já estão salvas no histórico
for msg in st.session_state.historico_mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Se for uma resposta da IA e tiver fontes salvas, mostra a caixa
        if "fontes" in msg:
            with st.expander(f"Ver trechos consultados (Páginas: {msg['fontes']})"):
                for trecho in msg["chunks"]:
                    st.info(trecho)
                    
# Caixa de digitação do usuário no rodapé
pergunta_usuario = st.chat_input("Digite sua pergunta aqui...")

if pergunta_usuario:
    # Verifica se o usuário processou o documento antes de perguntar
    if not os.path.exists(CHROMA_PATH):
        st.error("O banco vetorial ainda está vazio. Faça um upload de um PDF na barra lateral e clique em 'Processar documento'.")
    else:
        # Mostra a pergunta do usuário na tela e salva no historico
        with st.chat_message("user"):
            st.markdown(pergunta_usuario)
            
        st.session_state.historico_mensagens.append({
            "role": "user",
            "content": pergunta_usuario
        })
        # Chama a engine e mostra a resposta da IA
        with st.chat_message("assistant"):
            with st.spinner("Consultando banco vetorial e gerando resposta..."):
                resultado = ask_question(pergunta_usuario, k=k_chunks)
                
                resposta_texto = resultado["resposta"]
                paginas_fontes = resultado["fontes"]
                chunks_usados = resultado["chunks_usados"]
                
                st.markdown(resposta_texto)
                
                # Aba recolhida para visualizar os chunks
                with st.expander(f"Ver trechos consultados (Páginas: {paginas_fontes})"):
                    for trecho in chunks_usados:
                        st.info(trecho)
                
        # Salva resposta da IA no historico da sessao
        st.session_state.historico_mensagens.append({
            "role": "assistant",
            "content": resposta_texto,
            "fontes": paginas_fontes,
            "chunks": chunks_usados
        })