import streamlit as st
import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# --- 1. 加载 .env (Day 2) ---
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    print("未找到 DeepSeek API Key!")
    # 在 Streamlit 中，我们用 st.error 来显示错误
    st.error("未找到 DeepSeek API Key! 请检查你的 .env 文件。")
    st.stop() # 停止执行


# --- 2. (核心) RAG 链条创建函数 (Day 5) ---
#
# 🌟🌟🌟 使用“魔法” @st.cache_resource 🌟🌟🌟
# 告诉 Streamlit：只运行这个函数一次，然后把结果“存”起来。
@st.cache_resource 
def get_rag_chain(file_path: str): # ⬅️ 🌟 升级：添加 file_path 参数
    print(f"--- 🧠 正在为 {file_path} 创建 RAG 引擎... (此过程只应运行一次!) ---")
    # 1-4. 加载、分割、嵌入、存储
    loader = PyPDFLoader(file_path) # ⬅️ 🌟 升级：使用传入的参数
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # 5. 检索
    retriever = vectorstore.as_retriever()

    # 6. RAG 链
    template = """
    你是一个问答助手。请根据以下提供的资料来回答问题。
    如果你不知道答案，请回答 "我不知道"。

    资料:
    {context}

    问题:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chat = ChatDeepSeek(model="deepseek-chat")
    output_parser = StrOutputParser()

    setup_and_retrieval = RunnableParallel(
        context=retriever,
        question=RunnablePassthrough()
    )

    rag_chain = setup_and_retrieval | prompt | chat | output_parser

    print("--- RAG 引擎创建完毕 ---")
    return rag_chain

# --- 3. (核心) Streamlit UI (Day 10 终极版) ---

st.title("🤖 Chat with your *own* PDF")

# 3a. (新) 在侧边栏创建“文件上传”UI
st.sidebar.title("📚 PDF 知识库")
uploaded_file = st.sidebar.file_uploader("请在此处上传你的 PDF:", type="pdf")

# 3b. (新) 只有当用户上传了文件后，才显示聊天界面
if uploaded_file is not None:
    
    # 1. (新) 将上传的文件“暂存”到磁盘
    # (这是最稳健的做法，能让 PyPDFLoader 正常读取)
    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True) # 确保文件夹存在
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer()) # getbuffer() 是获取“内存中文件”内容的方法
    
    st.sidebar.success(f"文件 '{uploaded_file.name}' 已成功上传并索引。")

    # 2. (新) “智能”获取 RAG 引擎
    #    @st.cache_resource 的“魔法”在这里：
    #    - 如果 `get_rag_chain(temp_file_path)` 之前运行过，它会“瞬间”返回缓存的引擎。
    #    - 如果这是一个“新”的 `temp_file_path`，它会“正常运行”函数（花1分钟），
    #      然后把“新引擎”缓存起来。
    try:
        rag_chain = get_rag_chain(temp_file_path)
    except Exception as e:
        st.error(f"创建 RAG 引擎失败： {e}")
        st.stop()

    # 3. (新) 为“每个文件”创建“专属”的聊天记录
    #    (我们把文件名作为“记忆芯片”的 Key)
    file_specific_history = f"messages_{uploaded_file.name}"
    if file_specific_history not in st.session_state:
        st.session_state[file_specific_history] = []

    # 4. (新) 显示“专属”聊天记录
    for message in st.session_state[file_specific_history]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 5. (新) 聊天输入框
    if prompt := st.chat_input(f"请提问关于 '{uploaded_file.name}' 的问题..."):
        
        # (存入并显示“专属”用户消息)
        st.session_state[file_specific_history].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # (调用 RAG 引擎)
        with st.chat_message("assistant"):
            try:
                with st.spinner("🧠 正在思考并检索你上传的 PDF..."):
                    response_content = rag_chain.invoke(prompt)
                st.markdown(response_content)
                
            except Exception as e:
                response_content = f"调用 RAG 引擎时出错： {e}"
                st.error(response_content)
        
        # (存入“专属”机器人回复)
        st.session_state[file_specific_history].append({"role": "assistant", "content": response_content})
        
else:
    # (如果还没上传文件)
    st.info("👋 请在左侧侧边栏上传一个 PDF 文件，开始聊天吧！")