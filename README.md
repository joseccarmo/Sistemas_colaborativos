# Trabalho final de Sistemas Colaborativos

Felipe Seiji Momma Valente	12543700

Jean Patrick Ngandu Mamani       14712678

José Pedro Cioni do Carmo          12623988

Thiago Shimada			12691032

## Introducao

O cenário escolhido foi um de produção textual, onde os humanos colaboram entre si e com LLM para desenvolver textos, artigos, apresentações, entre outros. 
Inicialmente, os usuários delimitam temas, escopos e apresentam pesquisas prévias que serviram como base de dados para o RAG. 
Através do arquivo inicial e da proposta de texto a ser criado, o LLM faz pesquisas nos dados e gera um rascunho, que pode ser vetado ou aprovado através de uma votação realizada pelos colaboradores.

## Instalacao
```
pip install -r requirements.txt

cp .env.example .env
#altere .env com sua chave de API

python -m streamlit run .\collab_streamlit_app.py
```

## Uso

1. adicione um PDF
2. clique em Build/Update Index
3. escolha um usuario
4. Utilize o char normalmente
5. Caso queira confirmacao para fazer alguma alteracao utilize a votacao de prompt
