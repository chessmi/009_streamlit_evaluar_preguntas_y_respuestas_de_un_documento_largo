import streamlit as st
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain
from langchain.prompts import PromptTemplate

def generar_respuesta(
    archivo_subido,
    api_key_openai,
    pregunta_texto,
    respuesta_texto
):
    # Formatear el archivo subido
    documentos = [archivo_subido.read().decode()]
    
    # Dividirlo en fragmentos pequeños
    divisor_texto = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    textos = divisor_texto.create_documents(documentos)
    incrustaciones = OpenAIEmbeddings(
        openai_api_key=api_key_openai
    )
    
    # Crear un almacén vectorial y almacenar los textos allí
    db = FAISS.from_documents(textos, incrustaciones)
    
    # Crear una interfaz de recuperación
    recuperador = db.as_retriever()
    
    # Crear un diccionario de preguntas y respuestas reales
    qa_real = [
        {
            "question": pregunta_texto,
            "answer": respuesta_texto
        }
    ]
    
    # FORZAR RESPUESTA EN ESPAÑOL CON UN PROMPT PERSONALIZADO
    prompt_espanol = PromptTemplate(
        input_variables=["context", "question"],
        template="Usa la siguiente información para responder la pregunta en español de forma clara y concisa:\n\n{context}\n\nPregunta: {question}\n\nRespuesta en español:"
    )
    
    # Cadena de preguntas y respuestas (QA)
    cadena_qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=api_key_openai),
        chain_type="stuff",
        retriever=recuperador,
        input_key="question",
        chain_type_kwargs={"prompt": prompt_espanol}  # Se fuerza el español
    )
    
    # Generar predicciones
    predicciones = cadena_qa.apply(qa_real)
    
    # Crear una cadena de evaluación
    cadena_evaluacion = QAEvalChain.from_llm(
        llm=OpenAI(openai_api_key=api_key_openai)
    )
    
    # Evaluar las respuestas generadas
    resultados_evaluados = cadena_evaluacion.evaluate(
        qa_real,
        predicciones,
        question_key="question",
        prediction_key="result",
        answer_key="answer"
    )
    
    respuesta = {
        "predicciones": predicciones,
        "resultados_evaluados": resultados_evaluados
    }
    
    return respuesta

st.set_page_config(
    page_title="Evaluar una Aplicación RAG"
)
st.title("Evaluar una Aplicación RAG")

with st.expander("Evalúa la calidad de una aplicación RAG"):
    st.write("""
        Para evaluar la calidad de una aplicación RAG,
        le haremos preguntas cuyas respuestas reales ya conocemos.
        
        De esta manera, podemos comprobar si la aplicación
        está proporcionando respuestas correctas o si está generando información incorrecta.
    """)

archivo_subido = st.file_uploader(
    "Sube un documento .txt",
    type="txt"
)

pregunta_texto = st.text_input(
    "Ingresa una pregunta cuya respuesta ya hayas verificado:",
    placeholder="Escribe tu pregunta aquí",
    disabled=not archivo_subido
)

respuesta_texto = st.text_input(
    "Ingresa la respuesta real a la pregunta:",
    placeholder="Escribe la respuesta confirmada aquí",
    disabled=not archivo_subido
)

resultado = []
with st.form(
    "mi_formulario",
    clear_on_submit=True
):
    api_key_openai = st.text_input(
        "Clave API de OpenAI:",
        type="password",
        disabled=not (archivo_subido and pregunta_texto)
    )
    enviado = st.form_submit_button(
        "Enviar",
        disabled=not (archivo_subido and pregunta_texto)
    )
    if enviado and api_key_openai.startswith("sk-"):
        with st.spinner(
            "Espera, por favor. Estoy trabajando en ello..."
            ):
            respuesta = generar_respuesta(
                archivo_subido,
                api_key_openai,
                pregunta_texto,
                respuesta_texto
            )
            resultado.append(respuesta)
            del api_key_openai
            
if len(resultado):
    st.write("Pregunta")
    st.info(respuesta["predicciones"][0]["question"])
    st.write("Respuesta real")
    st.info(respuesta["predicciones"][0]["answer"])
    st.write("Respuesta proporcionada por la aplicación de IA")
    st.info(respuesta["predicciones"][0]["result"])
    st.write("Por lo tanto, la respuesta de la aplicación de IA fue")
    st.info(respuesta["resultados_evaluados"][0]["results"])
