import os
import tempfile
from pathlib import Path

from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

"""
文档加载模块：将 Streamlit 上传的 UploadedFile 转为 LangChain Document 列表。
- pdf: PyPDFLoader
- txt/md/html/htm: TextLoader(utf-8)
- csv: CSVLoader(每行一条记录)
- json: 整份 JSON 按 utf-8 读入，作为一个 Document
"""


def load_file_to_document(uploaded_file):
    """
    根据上传后的文件(Streamlit 的 UploadedFile)加载为 Document 列表。
    """
    # 1. 根据文件名后缀判断文件类型（支持 jsonl）
    suffix = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if suffix == "pdf":
        file_type = "pdf"
    elif suffix in ["txt", "md", "html", "htm"]:
        file_type = suffix
    elif suffix == "csv":
        file_type = "csv"
    elif suffix == "json":
        file_type = "json"
    elif suffix == "jsonl":
        file_type = "jsonl"
    else:
        raise ValueError(f"暂不支持的文件类型: .{suffix}")

    # 2. 将上传内容写入临时文件，供各类 Loader 使用
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # 3. 按类型选择合适的 Loader
        if file_type == "pdf":
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        elif file_type == "csv":
            loader = CSVLoader(tmp_path, encoding="utf-8")
            docs = loader.load()
        elif file_type == "json":
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            docs = [Document(page_content=raw, metadata={"source": uploaded_file.name})]
        elif file_type == "jsonl":
            docs: list[Document] = []
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                for idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    docs.append(
                        Document(
                            page_content=line,
                            metadata={"source": uploaded_file.name, "line": idx},
                        )
                    )
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
            docs = loader.load()

        return docs
    finally:
        # 4. 删除临时文件
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    # 本文件主要供其他模块调用，单独运行时不做实际操作
    pass
