import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# --- إعدادات واجهة التطبيق ---
st.set_page_config(page_title="🤖 شات بوت ببياناتك", layout="centered")
st.title("🤖 شات بوت ببياناتك الخاصة")

# --- قراءة مفتاح الـ API من الأسرار (Secrets) ---
# لن نطلب من المستخدم إدخال المفتاح، سنقرأه من st.secrets
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("لم يتم العثور على Groq API Key في الأسرار (Secrets)!")
    st.info("الرجاء إضافة GROQ_API_KEY إلى أسرار التطبيق في Streamlit Cloud.")
    st.stop()

# --- دالة لمعالجة البيانات وإنشاء الـ Retriever ---
# (نفس الدالة، لا تغيير)
@st.cache_resource
def load_and_process_data(_file_content):
    try:
        file_content_as_string = _file_content.decode("utf-8")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs_splits = text_splitter.create_documents([file_content_as_string])
        
        # سيتم تحميل الموديل أول مرة فقط
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = Chroma.from_documents(docs_splits, embeddings)
        
        return vector_store.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة الملف: {e}")
        return None

# --- دالة لإنشاء سلسلة RAG ---
# (نفس الدالة، لا تغيير)
def get_rag_chain(retriever, llm):
    template = """
    أنت مساعد متخصص في الإجابة على الأسئلة.
    استخدم فقط المعلومات التالية (Context) للإجابة على سؤال المستخدم.
    إذا كان الجواب غير موجود في المعلومات، قل "أنا آسف، ليس لدي معلومات عن هذا".
    لا تحاول اختلاق إجابة.

    المعلومات (Context):
    {context}
    السؤال:
    {question}
    الإجابة المفيدة (باللغة العربية):
    """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- منطق التطبيق الأساسي (تم تعديله) ---

# 1. إنشاء الموديل (LLM)
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant", # أحدث موديل
    temperature=0.7
)

# 2. قراءة ملف البيانات "data.txt" الثابت
try:
    with open("data.txt", "rb") as f: # "rb" = read bytes
        file_bytes = f.read()
    
    # رسالة للـ "أول مرة" فقط عند تحميل الموديل
    with st.spinner("...جاري تحضير الذاكرة (أول مرة فقط)"):
        retriever = load_and_process_data(file_bytes)

except FileNotFoundError:
    st.error("الملف 'data.txt' غير موجود. الرجاء التأكد من رفع الملف مع المشروع.")
    st.stop()

if retriever:
    # 3. إنشاء سلسلة RAG
    rag_chain = get_rag_chain(retriever, llm)

    # 4. إعداد ذاكرة الشات
    if "messages" not in st.session_state:
        st.session_state.messages = [
            AIMessage(content="أهلاً بك! أنا جاهز للإجابة على أسئلتك من البيانات المتاحة.")
        ]

    # 5. عرض رسائل الشات السابقة
    for message in st.session_state.messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)

    # 6. استقبال إدخال المستخدم
    if prompt := st.chat_input("اسأل أي شيء عن ملفك..."):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("... أفكر"):
                response = rag_chain.invoke(prompt)
                st.write(response)
        
        st.session_state.messages.append(AIMessage(content=response))