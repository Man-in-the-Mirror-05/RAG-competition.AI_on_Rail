import io
import os
from typing import Dict, List, Optional

import PyPDF2
import markdown
import json
from tqdm import tqdm
import tiktoken
from bs4 import BeautifulSoup
import re
import numpy as np
from PIL import Image

try:
    import fitz  # type: ignore
    try:
        fitz.TOOLS.mupdf_display_errors(False)  # type: ignore
    except Exception:
        pass
except ImportError:
    fitz = None  # type: ignore

try:
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
except ImportError:
    RapidOCR = None  # type: ignore

enc = tiktoken.get_encoding("cl100k_base")


class ReadFiles:
    """
    class to read files
    """

    DEFAULT_ENABLE_OCR = os.getenv("ENABLE_PDF_OCR", "0").lower() in ("1", "true", "yes")
    EMBEDDING_TOKEN_LIMIT_ENV = os.getenv("EMBEDDING_TOKEN_LIMIT")

    def __init__(self, path: str, enable_ocr: Optional[bool] = None,
                 embedding_token_limit: Optional[int] = None) -> None:
        self._path = path
        self.file_list = self.get_files()
        if enable_ocr is None:
            enable_ocr = self.DEFAULT_ENABLE_OCR
        self.enable_ocr = enable_ocr
        if embedding_token_limit is None:
            embedding_token_limit = self._parse_token_limit(self.EMBEDDING_TOKEN_LIMIT_ENV)
        self.embedding_token_limit = embedding_token_limit

    def get_files(self):
        # args：dir_path，目标文件夹路径
        file_list = []
        for filepath, _, filenames in os.walk(self._path):
            # os.walk 函数将递归遍历指定文件夹
            for filename in filenames:
                # 通过后缀名判断文件类型是否满足要求
                if filename.endswith(".md"):
                    # 如果满足要求，将其绝对路径加入到结果列表
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".txt"):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".pdf"):
                    file_list.append(os.path.join(filepath, filename))
        return file_list

    _ocr_engine = None

    def get_content(self, max_token_len: int = 600, cover_content: int = 150):
        docs = []
        limit = self.embedding_token_limit
        iterable = tqdm(self.file_list, desc="Reading documents", unit="file")
        # 读取文件内容
        for file in iterable:
            iterable.set_postfix_str(os.path.basename(file))
            content = self.read_file_content(
                file, enable_ocr=self.enable_ocr)
            chunk_content = self.get_chunk(
                content, max_token_len=max_token_len, cover_content=cover_content)
            source = os.path.basename(file)
            for chunk in chunk_content:
                chunk = chunk.strip()
                if not chunk:
                    continue
                prefix = f"【来源：{source}】\n"
                if limit:
                    prefix_tokens = len(enc.encode(prefix))
                    available_tokens = limit - prefix_tokens
                    if available_tokens <= 0:
                        continue
                    for part in self._split_text_by_token_limit(chunk, available_tokens):
                        part = part.strip()
                        if not part:
                            continue
                        docs.append(prefix + part)
                else:
                    docs.append(prefix + chunk)
        return docs

    @classmethod
    def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
        chunk_text: List[str] = []
        curr_len = 0
        curr_chunk = ''

        token_len = max_token_len - cover_content
        lines = text.splitlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            line_len = len(enc.encode(line))

            if line_len > max_token_len:
                if curr_chunk:
                    chunk_text.append(curr_chunk)
                    curr_chunk = ''
                    curr_len = 0

                line_tokens = enc.encode(line)
                num_chunks = (len(line_tokens) + token_len - 1) // token_len

                for i in range(num_chunks):
                    start_token = i * token_len
                    end_token = min(start_token + token_len, len(line_tokens))
                    chunk_tokens = line_tokens[start_token:end_token]
                    chunk_part = enc.decode(chunk_tokens)
                    if i > 0 and chunk_text:
                        prev_chunk = chunk_text[-1]
                        cover_part = prev_chunk[-cover_content:] if len(prev_chunk) > cover_content else prev_chunk
                        chunk_part = cover_part + chunk_part
                    chunk_text.append(chunk_part)

                curr_chunk = ''
                curr_len = 0
                continue

            if curr_len + line_len + 1 <= token_len:
                if curr_chunk:
                    curr_chunk += '\n'
                    curr_len += 1
                curr_chunk += line
                curr_len += line_len
            else:
                if curr_chunk:
                    chunk_text.append(curr_chunk)

                if chunk_text:
                    prev_chunk = chunk_text[-1]
                    cover_part = prev_chunk[-cover_content:] if len(prev_chunk) > cover_content else prev_chunk
                    curr_chunk = cover_part + '\n' + line
                    curr_len = len(enc.encode(cover_part)) + 1 + line_len
                else:
                    curr_chunk = line
                    curr_len = line_len

        if curr_chunk:
            chunk_text.append(curr_chunk)

        return chunk_text

    @classmethod
    def read_file_content(cls, file_path: str, enable_ocr: bool = False):
        # 根据文件扩展名选择读取方法
        if file_path.endswith('.pdf'):
            return cls.read_pdf(file_path, enable_ocr=enable_ocr)
        elif file_path.endswith('.md'):
            return cls.read_markdown(file_path)
        elif file_path.endswith('.txt'):
            return cls.read_text(file_path)
        else:
            raise ValueError("Unsupported file type")

    @classmethod
    def read_pdf(cls, file_path: str, enable_ocr: bool = False):
        # 先使用 PyMuPDF（速度快，且更稳定）
        if fitz is not None:
            try:
                text = cls._read_pdf_with_fitz(file_path)
                if text.strip():
                    return text
            except Exception as e:
                print(f"[警告] MuPDF 解析失败：{os.path.basename(file_path)} -> {e}")
        # 再尝试 PyPDF2，逐页捕获异常，避免卡死
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    try:
                        page_text = page.extract_text() or ''
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        page_text = ''
                    text += page_text
        except KeyboardInterrupt:
            raise
        except Exception:
            text = ''
        if text.strip():
            return text
        # fallback to OCR if the PDF only contains images
        if not enable_ocr:
            return ""
        ocr_text = cls._read_pdf_with_ocr(file_path)
        return ocr_text

    @classmethod
    def read_markdown(cls, file_path: str):
        # 读取Markdown文件
        with open(file_path, 'r', encoding='utf-8') as file:
            md_text = file.read()
            html_text = markdown.markdown(md_text)
            # 使用BeautifulSoup从HTML中提取纯文本
            soup = BeautifulSoup(html_text, 'html.parser')
            plain_text = soup.get_text()
            # 使用正则表达式移除网址链接
            text = re.sub(r'http\S+', '', plain_text) 
            return text

    @classmethod
    def read_text(cls, file_path: str):
        # 读取文本文件
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    @classmethod
    def _read_pdf_with_ocr(cls, file_path: str) -> str:
        if RapidOCR is None or fitz is None:
            return ""
        if cls._ocr_engine is None:
            cls._ocr_engine = RapidOCR()
        ocr_text = []
        doc = fitz.open(file_path)
        try:
            for page in doc:
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img_np = np.array(image)
                result, _ = cls._ocr_engine(img_np)
                if not result:
                    continue
                for _, text, score in result:
                    if text and score >= 0:
                        ocr_text.append(text)
        finally:
            doc.close()
        return "\n".join(ocr_text)

    @classmethod
    def _read_pdf_with_fitz(cls, file_path: str) -> str:
        if fitz is None:
            return ""
        doc = fitz.open(file_path)
        texts = []
        try:
            for page in doc:
                text = page.get_text("text") or ""
                if text:
                    texts.append(text)
        finally:
            doc.close()
        return "\n".join(texts)

    @staticmethod
    def _split_text_by_token_limit(text: str, token_limit: int):
        if token_limit <= 0:
            return []
        tokens = enc.encode(text)
        if len(tokens) <= token_limit:
            return [text]
        chunks = []
        for i in range(0, len(tokens), token_limit):
            sub_tokens = tokens[i:i+token_limit]
            chunks.append(enc.decode(sub_tokens))
        return chunks

    @staticmethod
    def _parse_token_limit(value: Optional[str]) -> Optional[int]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        return text.strip()

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        if not text:
            return []
        pattern = r'(?<=[。！？!?；;:\.])\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
