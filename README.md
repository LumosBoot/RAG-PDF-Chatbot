# 🤖 Chat-with-PDF (一个基于 RAG 的 AI 知识库助手)

这是一个使用 Streamlit 和 LangChain 构建的 RAG (检索增强生成) 应用。
它允许用户上传自己的 PDF 文件，并基于该文件的内容进行智能问答。

## ✨ 核心功能 (Features)

* **📄 PDF 文件上传**: 用户可以上传任何 PDF 文档作为知识库。
* **🧠 RAG 问答**: 应用会“消化”PDF 内容，并基于文档回答用户的问题。
* **💬 聊天界面**: 使用 Streamlit `st.chat_message` 构建了直观的聊天 UI。
* **💡 智能缓存 (Multi-PDF Support)**: 
    * 使用 `@st.cache_resource` 为**每一个**上传的 PDF 动态创建并缓存一个专属的 RAG 引擎，实现多文件无缝切换和“秒级”加载。
    * 使用 `st.session_state` 为**每一个** PDF 维护一个专属的、独立的聊天记录。
* **🔒 约束回答**: 提示词 (Prompt) 经过精心设计，当 AI 在文档中找不到答案时，会回答“我不知道”，有效防止“胡说八道”(Hallucination)。


## 🛠️ 技术栈 (Technologies Used)

* **前端 (Frontend)**: `Streamlit`
* **后端 & AI 引擎 (Backend & AI)**: `LangChain`
* **LLM (大脑)**: `ChatDeepSeek` (API 调用)
* **嵌入模型 (Embeddings)**: `HuggingFaceEmbeddings` (本地模型 `all-MiniLM-L6-v2`)
* **向量数据库 (Vector DB)**: `ChromaDB` (本地)
* **PDF 解析**: `PyPDFLoader`
* **文本分割**: `RecursiveCharacterTextSplitter`

---

## 🏃‍♂️ 如何在本地运行 (How to Run)

1.  **克隆仓库 (Clone)**
    ```bash
    git clone [你的 GITHUB 仓库 URL]
    cd agent-sprint-project
    ```

2.  **创建并激活虚拟环境 (Setup Environment)**
    ```bash
    # (Mac/Linux)
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **安装依赖 (Install Dependencies)**
    ```bash
    pip install -r requirements.txt
    ```

4.  **设置 API 密钥 (Set API Key)**
    * 在项目根目录创建一个 `.env` 文件。
    * 在 `.env` 文件中添加你的 DeepSeek API Key：
    ```
    DEEPSEEK_API_KEY="sk-xxxxxxxxxx"
    ```

5.  **运行应用 (Run the App)**
    ```bash
    python3 -m streamlit run app.py
    ```

6.  在浏览器中打开 `http://localhost:8501`。