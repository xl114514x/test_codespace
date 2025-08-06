import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
import sys
import openai  # 添加此导入以修复 NameError: openai 未定义

# SQLite3 版本修复
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # 如果 pysqlite3 不可用，继续使用系统 sqlite3

# ChromaDB 导入和错误处理
try:
    from langchain_community.vectorstores import Chroma
    CHROMA_AVAILABLE = True
except ImportError as e:
    st.error(f"ChromaDB导入失败: {e}")
    CHROMA_AVAILABLE = False
except RuntimeError as e:
    if "sqlite3" in str(e).lower():
        st.error("SQLite3 版本不兼容，正在使用备用方案...")
        CHROMA_AVAILABLE = False
    else:
        st.error(f"ChromaDB运行时错误: {e}")
        CHROMA_AVAILABLE = False

def get_retriever():
    if not CHROMA_AVAILABLE:
        st.warning("ChromaDB不可用，将使用简单的文本搜索作为备用方案")
        return None
    
    try:
        # 获取 API 密钥
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("未找到 OpenAI API 密钥，请在 Streamlit secrets 或环境变量中设置 OPENAI_API_KEY")
            return None
        
        # 定义 Embeddings
        embedding = OpenAIEmbeddings(
            openai_api_key=api_key
        )
        # 向量数据库持久化路径
        persist_directory = './data_base/vector_db/chroma'
        
        # 检查目录是否存在
        if not os.path.exists(persist_directory):
            st.warning(f"向量数据库目录不存在: {persist_directory}")
            return None
        
        # 加载数据库
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )
        return vectordb.as_retriever()
    
    except Exception as e:
        st.error(f"创建检索器时出错: {e}")
        return None

def simple_fallback_retriever(query, knowledge_base=None):
    """
    简单的备用检索器，当ChromaDB不可用时使用
    """
    if knowledge_base is None:
        # 这里可以添加一些预设的知识库内容
        knowledge_base = [
            "这是一个示例知识库内容。",
            "当ChromaDB不可用时，我们使用这个简单的文本匹配。",
            "您可以在这里添加您的知识库内容。"
        ]
    
    # 简单的关键词匹配
    relevant_docs = []
    for doc in knowledge_base:
        if any(keyword.lower() in doc.lower() for keyword in query.split()):
            relevant_docs.append(doc)
    
    return relevant_docs[:3]  # 返回最多3个相关文档

def combine_docs(docs):
    if isinstance(docs, dict) and "context" in docs:
        return "\n\n".join(doc.page_content for doc in docs["context"])
    elif isinstance(docs, list):
        return "\n\n".join(docs)
    return str(docs) if docs else ""

def get_qa_chain_without_retriever():
    """
    不使用检索器的简单问答链
    """
    try:
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("未找到 OpenAI API 密钥")
            return None
        
        llm = ChatOpenAI(
            model_name="gpt-4o", 
            temperature=0,
            openai_api_key=api_key
        )
        
        system_prompt = (
            "你是一个友好的助手。请根据用户的问题提供有帮助的回答。"
            "如果你不知道答案，请诚实地说不知道。"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])
        
        return qa_prompt | llm | StrOutputParser()
    
    except Exception as e:
        st.error(f"创建问答链时出错: {e}")
        return None

def get_qa_history_chain():
    retriever = get_retriever()
    
    try:
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("未找到 OpenAI API 密钥，请在 Streamlit secrets 中设置 OPENAI_API_KEY")
            return None
        
        llm = ChatOpenAI(
            model_name="gpt-4o", 
            temperature=0,
            openai_api_key=api_key
        )
        
        if retriever is None:
            # 使用简单的问答链作为备用方案
            st.info("使用简化模式，无检索功能")
            return get_qa_chain_without_retriever()
        
        condense_question_system_template = (
            "请根据聊天记录总结用户最近的问题，"
            "如果没有多余的聊天记录则返回用户的问题。"
        )
        
        condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])
        
        retrieve_docs = RunnableBranch(
            (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever,),
            condense_question_prompt | llm | StrOutputParser() | retriever,
        )
        
        system_prompt = (
            "你是一个问答任务的助手。 "
            "请使用检索到的上下文片段回答这个问题。 "
            "如果你不知道答案就说不知道。 "
            "请使用简洁的话语回答用户。"
            "\n\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])
        
        qa_chain = (
            RunnablePassthrough().assign(context=combine_docs)
            | qa_prompt
            | llm
            | StrOutputParser()
        )
        
        qa_history_chain = RunnablePassthrough().assign(
            context=retrieve_docs,
        ).assign(answer=qa_chain)
        
        return qa_history_chain
    
    except Exception as e:
        st.error(f"创建问答链时出错: {e}")
        return get_qa_chain_without_retriever()

def gen_response(chain, input_text, chat_history):
    if not chain:
        yield "抱歉，系统初始化失败，无法回答问题。"
        return
    
    try:
        response = chain.stream({
            "input": input_text,
            "chat_history": chat_history
        })
        for res in response:
            if isinstance(res, dict) and "answer" in res:
                yield res["answer"]
            elif isinstance(res, str):
                yield res
    except openai.AuthenticationError as e:
        yield f"认证错误: 提供的 OpenAI API 密钥无效。请访问 https://platform.openai.com/account/api-keys 获取新密钥，并更新 Streamlit secrets 或环境变量。"
    except Exception as e:
        yield f"生成回答时出错: {e}"

def main():
    st.set_page_config(
        page_title="问答助手",
        page_icon="🦜",
        layout="wide"
    )
    
    st.markdown('### 🦜🔗 动手学大模型应用开发')
    
    # 显示系统状态
    if not CHROMA_AVAILABLE:
        st.warning("⚠️ ChromaDB不可用，正在使用简化模式")
    else:
        st.success("✅ 系统正常运行")
    
    # 存储对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 存储检索问答链
    if "qa_history_chain" not in st.session_state:
        with st.spinner("正在初始化系统..."):
            st.session_state.qa_history_chain = get_qa_history_chain()
    
    # 建立容器
    messages = st.container(height=550)
    
    # 显示整个对话历史
    for message in st.session_state.messages:
        with messages.chat_message(message[0]):
            st.write(message[1])
    
    if prompt := st.chat_input("请输入您的问题"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append(("human", prompt))
        
        # 显示当前用户输入
        with messages.chat_message("human"):
            st.write(prompt)
        
        # 生成回复
        answer_generator = gen_response(
            chain=st.session_state.qa_history_chain,
            input_text=prompt,
            chat_history=st.session_state.messages
        )
        
        # 流式输出
        with messages.chat_message("ai"):
            output = st.write_stream(answer_generator)
        
        # 将输出存入对话历史
        st.session_state.messages.append(("ai", output))

if __name__ == "__main__":
    main()
