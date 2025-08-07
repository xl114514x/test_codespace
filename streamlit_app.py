import io, os, sys  # 导入标准库，用于字节流处理、操作系统接口和系统退出
from dotenv import load_dotenv  # 从 python-dotenv 加载 .env 环境变量

import streamlit as st  # 导入 Streamlit，用于搭建网页应用
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # 导入 LangChain 的 OpenAI LLM 接口和嵌入模型
from langchain_chroma import Chroma  # 改用新版 langchain-chroma
from langchain_core.output_parsers import StrOutputParser  # 导入输出解析器
from langchain_core.prompts import ChatPromptTemplate  # 导入对话提示模板
from langchain_core.runnables import RunnableBranch, RunnablePassthrough  # 导入可运行组件

import PyPDF2  # 导入 PyPDF2 用于解析 PDF 文本
import docx  # 导入 python-docx 用于解析 .docx 文档

# —— 环境 & API Key —— #
load_dotenv()  # 读取项目根目录下的 .env 文件
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # 从环境变量中获取 OpenAI API Key
if not OPENAI_API_KEY:
    st.error("请在 .env 文件中设置 OPENAI_API_KEY")  # 如果未设置，显示错误提示
    sys.exit(1)  # 退出程序

# —— 页面 & 样式 —— #
st.set_page_config(page_title="重庆科技大学问答系统", layout="wide")  # 配置 Streamlit 页面标题和布局
st.markdown(
    "<h1 style='text-align:center;color:#333;margin-bottom:16px;'>🎓 重庆科技大学问答系统</h1>",
    unsafe_allow_html=True,  # 允许 HTML 渲染，用于自定义标题
)
st.markdown("""
<style>
.upload-box {
  border:1px dashed #bbb;
  border-radius:6px;
  height:200px;
  text-align:center;
  padding-top:60px;
  color:#666;
  margin-bottom:16px;
}
header, footer { visibility: hidden; }  /* 隐藏默认的 header 和 footer */
</style>
""", unsafe_allow_html=True)  # 注入自定义 CSS 样式

# —— 向量库 & RAG —— #
PERSIST_DIR = "workspaces/test_codespace/chroma_db"  # 向量库持久化目录
EMBEDDER = OpenAIEmbeddings(api_key=OPENAI_API_KEY, base_url="https://xiaoai.plus/v1")
# 初始化 OpenAI 嵌入模型，指定 API Key 和接口地址

def split_text(text, chunk_size=500):
    """按固定大小拆分文本为多个片段"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def extract_text(f):
    """根据文件类型提取纯文本"""
    name = f.name.lower()
    data = f.read()
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    if name.endswith(".docx"):
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)
    return ""

def index_file(f):
    """将单个文件的文本拆分、嵌入并存入 Chroma 向量库"""
    txt = extract_text(f)
    if not txt:
        return 0
    chunks = split_text(txt)
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=EMBEDDER)
    db.add_texts(chunks)
    # 新版 Chroma 自动持久化，无需手动调用 db.persist()
    return len(chunks)

def get_retriever():
    """创建一个检索器，用于基于查询向量检索文档"""
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=EMBEDDER).as_retriever()

def combine_docs(docs):
    """将检索到的文档内容拼接为单个上下文字符串"""
    return "\n\n".join(d.page_content for d in docs["context"])

def build_chain():
    """构建一个 RAG 问答链，包括历史浓缩和最终问答"""
    retriever = get_retriever()
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0,
                     api_key=OPENAI_API_KEY, base_url="https://xiaoai.plus/v1")
    # 步骤一：如果有历史则浓缩，否则直接使用输入
    condense = ChatPromptTemplate([
        ("system","总结用户最近的问题；如无历史则返回原问题。"),
        ("placeholder","{chat_history}"),
        ("human","{input}"),
    ])
    branch = RunnableBranch(
        (lambda x: not x.get("chat_history"), (lambda x: x["input"]) | retriever),
        condense | llm | StrOutputParser() | retriever,
    )
    # 步骤二：基于检索上下文做最终问答
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是重庆科技大学问答系统，使用检索到的上下文回答；"
         "如无答案，请说“我不知道”。\n\n{context}"
        ),
        ("placeholder","{chat_history}"),
        ("human","{input}"),
    ])
    qa = (RunnablePassthrough().assign(context=combine_docs)
          | prompt | llm | StrOutputParser())
    return RunnablePassthrough().assign(context=branch).assign(answer=qa)

def get_answer(chain, q, hist):
    """调用 RAG 链并返回 AI 回答"""
    res = chain.invoke({"input": q, "chat_history": hist})
    return res["answer"]

# —— 主逻辑 —— #
def main():
    # 初始化会话状态
    if "msgs" not in st.session_state:
        st.session_state.msgs = []  # 存储人机对话的列表 [(role, msg), ...]
    if "chain" not in st.session_state:
        st.session_state.chain = build_chain()

    col1, col2 = st.columns([3, 1])  # 左右两列布局：3:1

    # 右侧：知识库上传（支持多文件）
    with col2:
        st.markdown("### 📂 上传文件到知识库", unsafe_allow_html=True)
        st.markdown(
            "<div class='upload-box'>将文件拖放或点击上传<br/>(txt/pdf/docx，支持多选)</div>",
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            label="上传文件",  # 非空 label
            type=["txt","pdf","docx"],
            accept_multiple_files=True,
            key="uploader",
            label_visibility="hidden"  # 隐藏 label
        )
        if uploaded_files:
            for f in uploaded_files:
                cnt = index_file(f)
                if cnt:
                    st.success(f"{f.name} 已索引 {cnt} 段文本")
                else:
                    st.error(f"{f.name} 解析/索引失败")

    # 左侧：消息列表和用户输入
    with col1:
        if prompt := st.chat_input("说点什么…", key="chat_input"):
            st.session_state.msgs.append(("human", prompt))
            with st.spinner("AI 正在思考…"):
                ans = get_answer(st.session_state.chain, prompt, st.session_state.msgs)
            st.session_state.msgs.append(("ai", ans))
            if len(st.session_state.msgs) > 20:
                st.session_state.msgs = st.session_state.msgs[-20:]

        for role, msg in st.session_state.msgs:
            with st.chat_message(role):
                if role == "ai":
                    st.code(msg)
                else:
                    st.write(msg)

        if st.session_state.msgs and st.session_state.msgs[-1][0] == "ai":
            if st.button("🔄 重新生成上次回答"):
                st.session_state.msgs.pop()
                last_user = next((m for r, m in reversed(st.session_state.msgs) if r == "human"), None)
                if last_user:
                    with st.spinner("AI 正在重新思考…"):
                        new_ans = get_answer(st.session_state.chain, last_user, st.session_state.msgs)
                    st.session_state.msgs.append(("ai", new_ans))

if __name__ == "__main__":
    main()  # 执行主逻辑

