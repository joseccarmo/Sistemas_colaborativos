import os
import uuid
import tempfile
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from fpdf import FPDF

from collab_agent_rag import (
    build_llm,
    build_embeddings,
    build_vectorstore_from_pages,
    build_retriever,
    build_agent,
    load_pdf_pages,
)

USERS = ["Jose", "Felipe", "Thiago", "Jean"]


def get_shared_vectorstore_dir() -> str:
    base = os.environ.get("RAG_VDB_DIR", "./vdb")
    if not os.path.exists(base):
        os.makedirs(base)
    return base


def ensure_session_state():
    """Initialize all session variables"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "selected_user" not in st.session_state:
        st.session_state.selected_user = USERS[0]

    if "llm" not in st.session_state:
        st.session_state.llm = None

    if "original_pdf_bytes" not in st.session_state:
        st.session_state.original_pdf_bytes = None

        # lista de pdfs editados pela ia
    if "edited_pdfs" not in st.session_state:
        st.session_state.edited_pdfs = []


    if "vote_active" not in st.session_state:
        st.session_state.vote_active = False
    if "vote_initiator" not in st.session_state:
        st.session_state.vote_initiator = None
    if "vote_topic" not in st.session_state:
        st.session_state.vote_topic = ""
    if "vote_votes" not in st.session_state:
        st.session_state.vote_votes = {}
    if "vote_prompt" not in st.session_state:
        st.session_state.vote_prompt = ""
    if "vote_creating" not in st.session_state:
        st.session_state.vote_creating = False


def reset_vote_session():
    st.session_state.vote_active = False
    st.session_state.vote_initiator = None
    st.session_state.vote_topic = ""
    st.session_state.vote_votes = {}
    st.session_state.vote_prompt = ""
    st.session_state.vote_creating = False

    if "new_vote_prompt" in st.session_state:
        del st.session_state["new_vote_prompt"]


def build_or_update_index(uploaded_pdf_bytes: bytes, filename: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
        tmp.write(uploaded_pdf_bytes)
        tmp_path = tmp.name

    embeddings = build_embeddings()
    pages = load_pdf_pages(tmp_path)
    vectorstore = build_vectorstore_from_pages(
        pages,
        embeddings,
        persist_directory=get_shared_vectorstore_dir(),
        collection_name="book",
    )
    retriever = build_retriever(vectorstore)
    os.unlink(tmp_path)
    return retriever

# converte o texto para ser compativel com o fpdf
def to_latin1_safe(text: str) -> str:

    text = text.replace("●", "-").replace("•", "-")
    # se ainda tiver algo fora de latin-1, substitui por '?'
    return text.encode("latin-1", "replace").decode("latin-1")


def edit_pdf_bytes_with_instruction(pdf_bytes: bytes, instruction: str, llm) -> bytes:

    reader = PdfReader(BytesIO(pdf_bytes))

    full_text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        full_text_parts.append(page_text)
    full_text = "\n\n".join(full_text_parts)

    # prompt para o modelo
    prompt = (
        "Você receberá o texto integral de um PDF.\n"
        "Aplique EXATAMENTE a seguinte instrução do usuário, "
        "editando o texto conforme pedido.\n\n"
        f"Instrução do usuário: {instruction}\n\n"
        "IMPORTANTE:\n"
        "- Devolva APENAS o novo texto completo resultante, sem explicações.\n"
        "- Mantenha uma formatação legível com quebras de linha e parágrafos.\n\n"
        "Texto do PDF (entre as tags <PDF> e </PDF>):\n"
        "<PDF>\n"
        f"{full_text}\n"
        "</PDF>"
    )

    resp = llm.invoke(prompt)
    if hasattr(resp, "content"):
        new_text = resp.content
    else:
        new_text = str(resp)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    pdf.add_page()
    for line in new_text.splitlines():
        safe_line = to_latin1_safe(line)
        pdf.multi_cell(0, 8, safe_line)

    pdf_str = pdf.output(dest="S")

    if isinstance(pdf_str, bytes):
        pdf_bytes_out = pdf_str
    else:
        pdf_bytes_out = pdf_str.encode("latin-1", "replace")

    return pdf_bytes_out


#votacao

def render_vote_sidebar():
    """Renderiza a seção de votação na barra lateral (usando session_state)."""
    st.header("Votação de prompt para IA")
    current_user = st.session_state.selected_user

    # nao ha votacao ativa
    if not st.session_state.vote_active:
        st.caption("Não há votação em andamento.")

        # usuario clica para criar
        if not st.session_state.vote_creating:
            if st.button("Criar votação para enviar prompt ao agente", key="start_vote_prompt"):
                st.session_state.vote_creating = True
                st.rerun()
            return

        # usuario que clicou define o prompt
        st.info(f"{current_user} está criando uma votação de prompt.")
        prompt_text = st.text_area(
            "Prompt a ser enviado ao agente ( /pdf ...):",
            key="new_vote_prompt",
            placeholder="Ex: /pdf traduza o texto para ingles OU uma pergunta normal para o agente..."
        )

        col_ok, col_cancel = st.columns(2)
        with col_ok:
            if st.button("Confirmar votação", key="confirm_vote_prompt"):
                if not prompt_text.strip():
                    st.warning("O prompt não pode ser vazio.")
                    return
                if st.session_state.agent is None:
                    st.warning("Não há agente criado. Crie o agente antes de iniciar a votação.")
                    return

                # Inicia a votação de fato
                st.session_state.vote_active = True
                st.session_state.vote_initiator = current_user
                st.session_state.vote_topic = "Votação para enviar um prompt ao agente"
                st.session_state.vote_prompt = prompt_text.strip()
                st.session_state.vote_votes = {}
                st.session_state.vote_creating = False
                st.success(f"Votação criada por {current_user}.")
                st.rerun()

        with col_cancel:
            if st.button("Cancelar", key="cancel_vote_prompt"):
                reset_vote_session()
                st.rerun()

        return

    # --------- EXISTE UMA VOTAÇÃO ATIVA ----------
    st.warning(
        f"Votação ativa: **{st.session_state.vote_topic}**\n\n"
        f"Iniciada por: **{st.session_state.vote_initiator}**"
    )

    # Mostrar o prompt proposto como justificativa
    st.markdown("**Prompt proposto para a IA:**")
    st.code(st.session_state.vote_prompt, language="markdown")

    votes = st.session_state.vote_votes
    user_vote = votes.get(current_user, None)
    if user_vote is True:
        st.info("Seu voto atual: **Aprovar (executar)**")
    elif user_vote is False:
        st.info("Seu voto atual: **Rejeitar (não executar)**")
    else:
        st.info("Você ainda não votou nesta votação.")

    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("Aprovar (executar)", key=f"vote_yes_{current_user}"):
            votes[current_user] = True
            st.session_state.vote_votes = votes
            st.rerun()
    with col_no:
        if st.button("Rejeitar (não executar)", key=f"vote_no_{current_user}"):
            votes[current_user] = False
            st.session_state.vote_votes = votes
            st.rerun()

    total_users = len(USERS)
    yes_votes = sum(1 for v in votes.values() if v is True)
    no_votes = sum(1 for v in votes.values() if v is False)

    st.write(f"✅ Aprovar (executar): **{yes_votes}** / {total_users}")
    st.write(f"❌ Rejeitar (não executar): **{no_votes}** / {total_users}")

    # maioria mais um
    majority = total_users // 2 + 1

    prompt = st.session_state.vote_prompt
    prompt_lower = prompt.lower()

    # resultado da votacao
    if yes_votes >= majority:
        st.success("Maioria aprovou: o comando será executado.")

        # caso /pdf
        if prompt_lower.startswith("/pdf"):
            instruction = prompt[len("/pdf"):].strip()

            if not instruction:
                st.warning("O comando /pdf na votação não tem instrucao")
            elif st.session_state.original_pdf_bytes is None:
                st.warning("Nenhum PDF foi enviado. Faca o upload de um PDF na barra lateral.")
            elif st.session_state.llm is None:
                st.warning("Crie o agente na barra lateral antes de usar o comando /pdf.")
            else:
                st.info(f"Enviando prompt: {instruction}")
                new_pdf_bytes = edit_pdf_bytes_with_instruction(
                    st.session_state.original_pdf_bytes,
                    instruction,
                    st.session_state.llm
                )

                # guarda permanentemente
                label = f"PDF editado (votação /pdf): {instruction[:40]}..."
                st.session_state.edited_pdfs.append(
                    {
                        "label": label,
                        "data": new_pdf_bytes,
                    }
                )

                st.download_button(
                    label="Baixar PDF editado (votação - último)",
                    data=new_pdf_bytes,
                    file_name="pdf_editado_votacao.pdf",
                    mime="application/pdf",
                    key="last_pdf_from_vote",
                )

                # registra no histórico
                st.session_state.messages.append(
                    {
                        "user": f"[Votação /pdf] {st.session_state.vote_initiator}",
                        "role": "user",
                        "content": f"/pdf {instruction}",
                    }
                )
        # caso nao seja pdf, prompt normal para o agente
        else:
            if st.session_state.agent is None:
                st.warning("Não há agente criado no momento. Crie o agente na barra lateral.")
            else:
                with st.spinner("Enviando prompt aprovado ao agente..."):
                    st.session_state.messages.append(
                        {
                            "user": f"[Votação] {st.session_state.vote_initiator}",
                            "role": "user",
                            "content": prompt,
                        }
                    )
                    result = st.session_state.agent.invoke({
                        "messages": [
                            {"type": "human", "content": prompt}
                        ]
                    })
                    content = result["messages"][-1].content
                    st.session_state.messages.append({"role": "assistant", "content": content})

        reset_vote_session()
        st.rerun()

    # se a maioria rejeitar, encerra a votacao sem executar
    elif no_votes >= majority:
        st.info("Maioria rejeitou: nada será executado. Votação encerrada.")
        reset_vote_session()
        st.rerun()



def main():
    load_dotenv()
    st.set_page_config(page_title="RAG Chat Drafter", page_icon="📄")
    ensure_session_state()

    st.title("📄 RAG Chat Drafter")
    st.caption("Faca um upload de um pdf, entao converse com o agente sobre ele ou use /pdf para fazer alteracoes e obter o arquivo")

    with st.sidebar:

        st.header("Usuario")
        st.session_state.selected_user = st.selectbox(
            "Usuario ativo",
            USERS,
            index=USERS.index(st.session_state.selected_user)
        )
        st.caption("Mensagens enviadas serao atribuidas a esse usuario")

        st.header("Data")
        # upload PDF
        uploaded = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded is not None:
            # guarda o PDF original em memória
            pdf_bytes = uploaded.read()
            st.session_state.original_pdf_bytes = pdf_bytes

            # if success, allow button to build/update vectorDB index
            if st.button("Build/Update Index", type="primary"):
                with st.spinner("Criando index... "):
                    st.session_state.retriever = build_or_update_index(pdf_bytes, uploaded.name)
                st.success("Index criado.")

        st.divider()
        st.header("Agente")
        # slider k
        k = st.slider("Retrieve k chunks", min_value=2, max_value=10, value=6, step=2)

        # model selector
        model = st.selectbox("LLM", ["gpt-4o-mini", "gpt-4o"], index=0)

        # slider temperature
        temp = st.slider("Model temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)

        # button for agent creation
        if st.button("(re)Criar Agente"):
            if st.session_state.retriever is None:
                st.warning("Porfavor faca o index primeiro")
            else:
                llm = build_llm(model=model, temperature=temp)
                retriever = st.session_state.retriever
                retriever.search_kwargs["k"] = k
                st.session_state.agent = build_agent(retriever, llm)
                st.session_state.llm = llm   # guarda o LLM para editar PDF
                st.success("Agente pronto.")

        st.divider()
        # seccao de votacao para enviar o prompt
        render_vote_sidebar()

    # Chat area
    st.subheader("Conversa (compartilhado com todos os usuarios)")
    for msg in st.session_state.messages:
        author = msg.get("user", "User") if msg["role"] == "user" else "assistant"
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(f"**{author}**: {msg['content']}")

    # seccao fixa de pdfs gerados
    if st.session_state.edited_pdfs:
        st.subheader("📄 PDFs gerados pelo agente")
        for i, pdf_info in enumerate(st.session_state.edited_pdfs):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"- {pdf_info['label']}")
            with col2:
                st.download_button(
                    label="Baixar",
                    data=pdf_info["data"],
                    file_name=f"pdf_editado_{i + 1}.pdf",
                    mime="application/pdf",
                    key=f"download_pdf_{i}",
                )

    # Chat normal + comando /pdf
    if prompt := st.chat_input(f"{st.session_state.selected_user} diz: "):

        #  /pdf <instrução>
        if prompt.lower().startswith("/pdf"):
            instruction = prompt[len("/pdf"):].strip()

            with st.chat_message("user"):
                st.markdown(f"**{st.session_state.selected_user}** (comando /pdf): {instruction}")

            if not instruction:
                with st.chat_message("assistant"):
                    st.warning("Você precisa escrever uma instrução depois de /pdf. Ex: /pdf exclua 5 parágrafos")
            elif st.session_state.original_pdf_bytes is None:
                with st.chat_message("assistant"):
                    st.warning("Nenhum PDF foi enviado. Faça o upload de um PDF na barra lateral.")
            elif st.session_state.llm is None:
                with st.chat_message("assistant"):
                    st.warning("Crie o agente na barra lateral antes de usar o comando /pdf.")
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Editando PDF conforme instrução..."):
                        new_pdf_bytes = edit_pdf_bytes_with_instruction(
                            st.session_state.original_pdf_bytes,
                            instruction,
                            st.session_state.llm
                        )

                        # guarda permanentemente na sessão
                        label = f"PDF editado (/pdf): {instruction[:40]}..."
                        st.session_state.edited_pdfs.append(
                            {
                                "label": label,
                                "data": new_pdf_bytes,
                            }
                        )

                        st.download_button(
                            label="Baixar PDF editado (último)",
                            data=new_pdf_bytes,
                            file_name="pdf_editado.pdf",
                            mime="application/pdf",
                            key="last_pdf_from_chat",
                        )

                # registra no histórico
                st.session_state.messages.append(
                    {
                        "user": st.session_state.selected_user,
                        "role": "user",
                        "content": f"[COMANDO /pdf] {instruction}",
                    }
                )

        # fluxo normal do chat
        else:
            st.session_state.messages.append(
                {"user": st.session_state.selected_user, "role": "user", "content": prompt}
            )
            with st.chat_message("user"):
                st.markdown(f"**{st.session_state.selected_user}**: {prompt}")

            if st.session_state.agent is None:
                with st.chat_message("assistant"):
                    st.warning("Crie o agente na barra lateral antes de perguntar.")
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Pensando..."):
                        # Use LC message dict format expected by the agent
                        result = st.session_state.agent.invoke({
                            "messages": [
                                {"type": "human", "content": prompt}
                            ]
                        })
                        content = result["messages"][-1].content
                        st.markdown(content)
                        st.session_state.messages.append({"role": "assistant", "content": content})


if __name__ == "__main__":
    main()
