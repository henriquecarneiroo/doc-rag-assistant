# DocQA - Assistente Inteligente de Documentação Técnica

Uma aplicação interativa de RAG (Retrieval-Augmented Generation) em Python para ajudar em análises de documentos técnicos e editais em PDF. Em vez de depender apenas do conhecimento geral do modelo ou sofrer com alucinações, o sistema faz uma busca semântica no documento fornecido e traz respostas acompanhadas das páginas de onde o conteúdo foi extraído.

Este repositório nasceu como um projeto prático de estudos para entender como a arquitetura de RAG funciona, explorando desde a matemática dos vetores até o gerenciamento de sessões na web.

---

<!-- ## Demonstração

<!-- Dica: Tire um print da sua tela do Streamlit funcionando (com a gaveta de trechos aberta) ou grave um GIF curto pelo ShareX/Loom, salve como demo.png na raiz e a imagem vai aparecer aqui
![Demonstração da aplicação](demo.png)

--->

## Como funciona

O fluxo segue a separação clássica de um sistema RAG em duas etapas independentes:

```text
[ Documento PDF ]
       │
       ▼ (PyPDFLoader + RecursiveSplitter)
[ Chunks de texto ] 
       │
       ▼ (Embeddings)
[ Vetores normalizados ] ──► Armazenamento no ChromaDB local
                                     │
[ Pergunta do usuário ]              │
       │                             ▼
       └──────────────► [ Busca semântica por similaridade ]
                                     │
                                     ▼
                            [ Contexto relevante ]
                                     │
                                     ▼ (OpenRouter API)
                            [ Resposta + Fontes ]

```

Na etapa de ingestão, o PDF enviado é dividido em partes dinamicamente com base em parágrafos e frases. Cada pedaço vira um vetor geométrico através de um modelo leve que roda localmente na CPU, armazenando os dados no ChromaDB sem necessidade de subir o arquivo para servidores externos.

A partir disso, é permitido fazer consultas e respostas: ao digitar uma dúvida, o banco vetorial localiza os trechos com maior proximidade semântica e esses trechos são grampeados em um prompt protegido contra respostas inventadas e enviados para a IA via OpenRouter, retornando a explicação e os números das páginas consultadas.

Durante a implementação, surgiram obstáculos práticos, como o dilema dos tamanhos dos chunks, bloqueio de arquivos no Windows e limites de modelos de LLM's gratuitos. Esses problemas foram resolvidos da seguinte forma:
- Sliders práticos foram adicionados para calibrar os tamnahos dos chunks e outros parâmetros relevantes de contexto;
- Troca de remoções manuais para método de limpeza interna da coleção do ChromaDB;
- Mecanismo de fallback com tentativas sequenciais e pequenas pausas programadas.

## Como rodar o projeto

### Pré-requisitos

- Python 3.10+;
- Chave de API gratuita do OpenRouter.

### Passo a passo

```
git clone [https://github.com/henriquecarneiroo/doc-rag-assistant.git](https://github.com/henriquecarneiroo/doc-rag-assistant.git)
cd doc-rag-assistant
```

Crie o ambiente virtual e instale as dependências:

```
python -m venv venv

# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

Crie um arquivo .env na raiz do projeto (use o .env.example como base) e informe sua chave:

```
OPENROUTER_API_KEY=sua_chave_aqui
```

Inicie o servidor do Streamlit:

```
streamlit run src/app.py
```

O navegador abrirá automaticamente no endereço local http://localhost:8501. Faça upload de um PDF, ajuste os parâmetros na barra lateral e clique em 'Processar documento' para começar a conversar.

## Tecnologias usadas

- **LangChain**: para gerenciamento do pipeline, cortes de texto e templates de prompt;
- **HuggingFace e PyTorch**: geração local de embeddings com ```sentence-transformers```;
- **ChromaDB**: banco de dados vetorial leve e persistente no disco;
- **OpenRouter**: acesso dinâmico a modelos de linguagem;
    Streamlit: frontend reativo e acessível com histórico de sessão e componentes de chat.
