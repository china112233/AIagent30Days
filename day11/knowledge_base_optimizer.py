"""
Day 11: 知识库优化实现

实现文档清洗、自适应切分、质量评分等知识库优化策略。
"""

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class Document:
    """文档数据结构"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Document):
            return self.id == other.id
        return False


@dataclass
class Chunk:
    """文档块"""
    id: str
    document_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None  # 父块ID（用于父子索引）
    children: List[str] = field(default_factory=list)  # 子块ID列表


# ==================== 文档清洗器 ====================

class DocumentCleaner:
    """
    文档清洗器
    
    提供多种文档清洗功能
    """
    
    def __init__(self):
        """初始化文档清洗器"""
        # 常见噪声模式
        self.noise_patterns = [
            r'<[^>]+>',  # HTML 标签
            r'\[\d+\]',  # 引用标记 [1], [2]
            r'``````',  # Markdown 代码块标记
            r'\\n\\n',  # 多余换行
            r'\s{3,}',  # 多余空格
        ]
        
        # 页眉页脚关键词
        self.header_keywords = ["目录", "目录", "CONTENTS", "Table of Contents"]
        self.footer_keywords = ["第 \\d+ 页", "Page \\d+", "\\d+/\\d+"]
    
    def clean_html(self, content: str) -> str:
        """移除 HTML 标签"""
        return re.sub(r'<[^>]+>', '', content)
    
    def clean_whitespace(self, content: str) -> str:
        """清理空白字符"""
        # 统一换行符
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        # 移除多余空格
        content = re.sub(r'[ \t]+', ' ', content)
        # 移除多余空行
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content.strip()
    
    def clean_special_chars(self, content: str) -> str:
        """清理特殊字符"""
        # 移除不可见字符
        content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', content)
        # 统一引号
        content = content.replace('"', '"').replace('"', '"')
        content = content.replace(''', "'").replace(''', "'")
        return content
    
    def remove_references(self, content: str) -> str:
        """移除引用标记"""
        # 移除 [1], [2] 等引用
        content = re.sub(r'\[\d+\]', '', content)
        # 移除 (Smith et al., 2020) 等引用
        content = re.sub(r'\([A-Z][a-z]+ et al\.,?\s*\d{4}\)', '', content)
        return content
    
    def normalize_format(self, content: str) -> str:
        """标准化格式"""
        # 统一中英文标点
        punctuation_map = {
            '，': ', ',
            '。': '. ',
            '！': '! ',
            '？': '? ',
            '：': ': ',
            '；': '; ',
            '（': ' (',
            '）': ') ',
        }
        for cn, en in punctuation_map.items():
            content = content.replace(cn, en)
        return content
    
    def clean(
        self,
        content: str,
        remove_html: bool = True,
        remove_refs: bool = False,
        normalize: bool = True
    ) -> str:
        """
        综合清洗
        
        Args:
            content: 原始内容
            remove_html: 是否移除 HTML 标签
            remove_refs: 是否移除引用
            normalize: 是否标准化格式
            
        Returns:
            清洗后的内容
        """
        if remove_html:
            content = self.clean_html(content)
        
        if remove_refs:
            content = self.remove_references(content)
        
        content = self.clean_special_chars(content)
        content = self.clean_whitespace(content)
        
        if normalize:
            content = self.normalize_format(content)
        
        return content
    
    def clean_document(self, doc: Document, **kwargs) -> Document:
        """清洗文档"""
        cleaned_content = self.clean(doc.content, **kwargs)
        # 安全获取quality_score属性，如果不存在则使用默认值
        quality_score = getattr(doc, 'quality_score', None)
        # 创建新的Document对象
        new_doc = Document(
            id=doc.id,
            content=cleaned_content,
            metadata=doc.metadata.copy()
        )
        # 如果存在quality_score，则添加到新文档
        if quality_score is not None:
            new_doc.quality_score = quality_score
        return new_doc


# ==================== 文档去重器 ====================

class DocumentDeduplicator:
    """
    文档去重器
    
    支持精确去重和近似去重
    """
    
    def __init__(self, similarity_threshold: float = 0.9):
        """
        初始化文档去重器
        
        Args:
            similarity_threshold: 相似度阈值，超过此值视为重复
        """
        self.similarity_threshold = similarity_threshold
    
    def _get_hash(self, content: str) -> str:
        """计算内容哈希"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _get_minhash(self, content: str, num_hashes: int = 128) -> List[int]:
        """
        计算 MinHash（用于近似去重）
        
        Args:
            content: 文本内容
            num_hashes: 哈希函数数量
            
        Returns:
            MinHash 签名
        """
        # 分词
        words = content.lower().split()
        if not words:
            return [float('inf')] * num_hashes
        
        # 使用不同的哈希种子
        signature = []
        for i in range(num_hashes):
            min_hash = float('inf')
            for word in words:
                # 简单的哈希函数
                h = hash(f"{i}:{word}")
                min_hash = min(min_hash, h)
            signature.append(min_hash)
        
        return signature
    
    def _jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """计算 Jaccard 相似度（基于 MinHash）"""
        if len(sig1) != len(sig2):
            return 0.0
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)
    
    def exact_deduplicate(
        self,
        documents: List[Document]
    ) -> Tuple[List[Document], List[Document]]:
        """
        精确去重
        
        Args:
            documents: 文档列表
            
        Returns:
            (去重后的文档列表, 重复的文档列表)
        """
        seen_hashes: Set[str] = set()
        unique_docs = []
        duplicate_docs = []
        
        for doc in documents:
            content_hash = self._get_hash(doc.content)
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)
            else:
                duplicate_docs.append(doc)
        
        return unique_docs, duplicate_docs
    
    def approximate_deduplicate(
        self,
        documents: List[Document]
    ) -> Tuple[List[Document], List[Document]]:
        """
        近似去重（基于 MinHash）
        
        Args:
            documents: 文档列表
            
        Returns:
            (去重后的文档列表, 重复的文档列表)
        """
        unique_docs = []
        duplicate_docs = []
        signatures: List[Tuple[Document, List[int]]] = []
        
        for doc in documents:
            sig = self._get_minhash(doc.content)
            
            # 检查是否与已有文档相似
            is_duplicate = False
            for existing_doc, existing_sig in signatures:
                similarity = self._jaccard_similarity(sig, existing_sig)
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                duplicate_docs.append(doc)
            else:
                unique_docs.append(doc)
                signatures.append((doc, sig))
        
        return unique_docs, duplicate_docs


# ==================== 自适应切分器 ====================

class AdaptiveChunker:
    """
    自适应切分器
    
    根据内容类型自动选择最佳切分策略
    """
    
    def __init__(
        self,
        default_chunk_size: int = 500,
        default_overlap: int = 50,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000
    ):
        """
        初始化自适应切分器
        
        Args:
            default_chunk_size: 默认块大小
            default_overlap: 默认重叠大小
            min_chunk_size: 最小块大小
            max_chunk_size: 最大块大小
        """
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # 分隔符层级（优先级从高到低）
        self.separators = [
            "\n\n\n",  # 章节分隔
            "\n\n",    # 段落分隔
            "\n",      # 行分隔
            "。",      # 中文句号
            ".",       # 英文句号
            "！",      # 中文感叹号
            "!",       # 英文感叹号
            "？",      # 中文问号
            "?",       # 英文问号
            "；",      # 中文分号
            ";",       # 英文分号
            "，",      # 中文逗号
            ",",       # 英文逗号
            " ",       # 空格
            "",        # 字符级别
        ]
    
    def _detect_content_type(self, content: str) -> str:
        """
        检测内容类型
        
        Args:
            content: 文档内容
            
        Returns:
            内容类型
        """
        # 检测代码
        if 'def ' in content or 'class ' in content or 'function ' in content:
            if '{' in content and '}' in content:
                return "code"
        
        # 检测问答对
        if re.search(r'^(Q|问)[:：]', content, re.MULTILINE):
            if re.search(r'^(A|答)[:：]', content, re.MULTILINE):
                return "qa"
        
        # 检测 Markdown
        if re.search(r'^#{1,6}\s', content, re.MULTILINE):
            return "markdown"
        
        # 检测 JSON
        content_stripped = content.strip()
        if content_stripped.startswith('{') and content_stripped.endswith('}'):
            return "json"
        if content_stripped.startswith('[') and content_stripped.endswith(']'):
            return "json"
        
        # 检测表格
        if re.search(r'\|.+\|', content) and '---' in content:
            return "table"
        
        # 默认为普通文本
        return "text"
    
    def _split_by_separators(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """
        按分隔符层级递归切分
        
        Args:
            text: 待切分文本
            chunk_size: 目标块大小
            overlap: 重叠大小
            
        Returns:
            切分结果
        """
        if len(text) <= chunk_size:
            return [text] if text.strip() else []
        
        # 尝试不同的分隔符
        for separator in self.separators:
            if separator in text:
                parts = text.split(separator)
                chunks = []
                current_chunk = ""
                
                for part in parts:
                    if not part.strip():
                        continue
                    
                    # 如果当前块 + 新部分 <= chunk_size，添加到当前块
                    if len(current_chunk) + len(separator) + len(part) <= chunk_size:
                        if current_chunk:
                            current_chunk += separator + part
                        else:
                            current_chunk = part
                    else:
                        # 当前块已满，保存并开始新块
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        
                        # 处理重叠
                        if overlap > 0 and current_chunk:
                            overlap_text = current_chunk[-overlap:]
                            current_chunk = overlap_text + separator + part
                        else:
                            current_chunk = part
                
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # 检查是否有超大块
                final_chunks = []
                for chunk in chunks:
                    if len(chunk) > self.max_chunk_size:
                        # 递归切分
                        final_chunks.extend(
                            self._split_by_separators(chunk, chunk_size, overlap)
                        )
                    else:
                        final_chunks.append(chunk)
                
                return final_chunks
        
        # 无法按分隔符切分，按字符切分
        return [
            text[i:i+chunk_size]
            for i in range(0, len(text), chunk_size - overlap)
        ]
    
    def _split_qa_pairs(self, content: str) -> List[str]:
        """切分问答对"""
        # 匹配问答对
        qa_pattern = r'(?:^|\n)([Q问][:：].*?)(?=(?:\n[Q问答A][:：])|$)'
        matches = re.findall(qa_pattern, content, re.DOTALL)
        
        # 同时获取答案
        qa_pairs = []
        for match in matches:
            # 提取问题和答案
            parts = re.split(r'\n[A答][:：]', match, maxsplit=1)
            if len(parts) == 2:
                qa_pairs.append(match.strip())
        
        return qa_pairs if qa_pairs else [content]
    
    def _split_markdown(self, content: str, chunk_size: int) -> List[str]:
        """按 Markdown 标题切分"""
        # 匹配标题
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = content.split('\n')
        
        sections = []
        current_section = []
        current_header = ""
        
        for line in lines:
            match = re.match(header_pattern, line)
            if match:
                # 遇到新标题，保存当前节
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
                current_header = match.group(2)
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        # 检查大小，可能需要进一步切分
        final_chunks = []
        for section in sections:
            if len(section) <= chunk_size:
                final_chunks.append(section)
            else:
                final_chunks.extend(
                    self._split_by_separators(section, chunk_size, self.default_overlap)
                )
        
        return final_chunks
    
    def _split_code(self, content: str, chunk_size: int) -> List[str]:
        """按函数/类切分代码"""
        # 匹配函数和类定义
        pattern = r'((?:def |class |function |const |let |var ).*?(?=\n(?:def |class |function |const |let |var )|$))'
        matches = re.findall(pattern, content, re.DOTALL)
        
        if matches:
            return [m.strip() for m in matches if m.strip()]
        
        # 无法按函数切分，按行数切分
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            if current_size + len(line) > chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            current_chunk.append(line)
            current_size += len(line)
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def chunk(
        self,
        document: Document,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        content_type: Optional[str] = None
    ) -> List[Chunk]:
        """
        切分文档
        
        Args:
            document: 文档对象
            chunk_size: 块大小
            overlap: 重叠大小
            content_type: 内容类型（自动检测如果不提供）
            
        Returns:
            文档块列表
        """
        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.default_overlap
        content = document.content
        
        # 检测内容类型
        if content_type is None:
            content_type = self._detect_content_type(content)
        
        # 根据内容类型选择切分策略
        if content_type == "qa":
            chunks_text = self._split_qa_pairs(content)
        elif content_type == "markdown":
            chunks_text = self._split_markdown(content, chunk_size)
        elif content_type == "code":
            chunks_text = self._split_code(content, chunk_size)
        else:
            chunks_text = self._split_by_separators(content, chunk_size, overlap)
        
        # 构建 Chunk 对象
        chunks = []
        for i, chunk_content in enumerate(chunks_text):
            chunk_id = f"{document.id}_chunk_{i}"
            chunks.append(Chunk(
                id=chunk_id,
                document_id=document.id,
                content=chunk_content,
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                    "content_type": content_type
                }
            ))
        
        return chunks
    
    def chunk_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> List[Chunk]:
        """批量切分文档"""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk(doc, **kwargs)
            all_chunks.extend(chunks)
        return all_chunks


# ==================== 父子索引切分器 ====================

class ParentChildChunker:
    """
    父子索引切分器
    
    大块用于生成（完整上下文），小块用于检索（精确匹配）
    """
    
    def __init__(
        self,
        parent_size: int = 1000,
        child_size: int = 200,
        parent_overlap: int = 100,
        child_overlap: int = 20
    ):
        """
        初始化父子索引切分器
        
        Args:
            parent_size: 父块大小
            child_size: 子块大小
            parent_overlap: 父块重叠
            child_overlap: 子块重叠
        """
        self.parent_size = parent_size
        self.child_size = child_size
        self.parent_overlap = parent_overlap
        self.child_overlap = child_overlap
        
        self.base_chunker = AdaptiveChunker(
            default_chunk_size=parent_size,
            default_overlap=parent_overlap
        )
    
    def chunk_with_parents(
        self,
        document: Document
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """
        切分文档并建立父子关系
        
        Args:
            document: 文档对象
            
        Returns:
            (父块列表, 子块列表)
        """
        # 切分父块
        parent_chunks = self.base_chunker.chunk(
            document,
            chunk_size=self.parent_size,
            overlap=self.parent_overlap
        )
        
        # 对每个父块切分子块
        child_chunks = []
        for parent in parent_chunks:
            # 创建临时文档用于切分子块
            temp_doc = Document(
                id=parent.id,
                content=parent.content,
                metadata=parent.metadata
            )
            
            # 切分子块
            children = self.base_chunker.chunk(
                temp_doc,
                chunk_size=self.child_size,
                overlap=self.child_overlap
            )
            
            # 建立父子关系
            for child in children:
                child.parent_id = parent.id
                child.metadata["parent_content"] = parent.content
                child_chunks.append(child)
            
            # 更新父块的子块列表
            parent.children = [c.id for c in children]
        
        return parent_chunks, child_chunks


# ==================== 质量评分器 ====================

class QualityScorer:
    """
    文档质量评分器
    
    评估文档的完整性、准确性、时效性等维度
    """
    
    def __init__(self):
        """初始化质量评分器"""
        # 停用词（用于计算信息密度）
        self.stop_words = {
            "的", "是", "在", "了", "和", "与", "或", "等", "也", "都",
            "这", "那", "有", "为", "能", "会", "可以", "但", "而",
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall"
        }
    
    def score_completeness(self, content: str) -> float:
        """
        评估完整性
        
        检查内容是否完整（未截断、无缺失）
        """
        score = 1.0
        
        # 检查是否有截断迹象
        truncation_patterns = [
            r'\.\.\.$',  # 以...结尾
            r'\[未完\]',  # 未完标记
            r'\[待续\]',  # 待续标记
        ]
        for pattern in truncation_patterns:
            if re.search(pattern, content):
                score -= 0.3
        
        # 检查是否有配对符号不匹配
        pairs = [('{', '}'), ('[', ']'), ('(', ')'), ('"', '"'), ('"', '"')]
        for open_c, close_c in pairs:
            if content.count(open_c) != content.count(close_c):
                score -= 0.1
        
        return max(0.0, score)
    
    def score_information_density(self, content: str) -> float:
        """
        评估信息密度
        
        信息密度 = (非停用词数) / 总词数
        """
        words = content.lower().split()
        if not words:
            return 0.0
        
        meaningful_words = [w for w in words if w not in self.stop_words]
        return len(meaningful_words) / len(words)
    
    def score_readability(self, content: str) -> float:
        """
        评估可读性
        
        基于平均句子长度和段落结构
        """
        # 计算平均句子长度
        sentences = re.split(r'[。！？.!?]', content)
        sentences = [s for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
        
        # 理想句子长度在 15-25 个字符
        if 15 <= avg_sentence_length <= 25:
            readability = 1.0
        elif avg_sentence_length < 15:
            readability = avg_sentence_length / 15
        else:
            readability = 25 / avg_sentence_length
        
        return readability
    
    def score_timeliness(
        self,
        doc: Document,
        reference_date: Optional[datetime] = None
    ) -> float:
        """
        评估时效性
        
        基于文档日期和参考日期的差异
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        # 从元数据获取文档日期
        doc_date = doc.metadata.get("date") or doc.metadata.get("created_at")
        
        if doc_date is None:
            return 0.5  # 无日期信息，给中等分数
        
        if isinstance(doc_date, str):
            try:
                doc_date = datetime.fromisoformat(doc_date)
            except ValueError:
                return 0.5
        
        # 计算日期差异（天）
        days_diff = (reference_date - doc_date).days
        
        # 时效性衰减
        if days_diff <= 30:
            return 1.0
        elif days_diff <= 180:
            return 0.8
        elif days_diff <= 365:
            return 0.6
        elif days_diff <= 730:
            return 0.4
        else:
            return 0.2
    
    def score(
        self,
        document: Document,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        计算综合质量分数
        
        Args:
            document: 文档对象
            weights: 各维度权重
            
        Returns:
            综合质量分数 (0-1)
        """
        if weights is None:
            weights = {
                "completeness": 0.3,
                "information_density": 0.3,
                "readability": 0.2,
                "timeliness": 0.2
            }
        
        scores = {
            "completeness": self.score_completeness(document.content),
            "information_density": self.score_information_density(document.content),
            "readability": self.score_readability(document.content),
            "timeliness": self.score_timeliness(document)
        }
        
        total_score = sum(scores[k] * weights[k] for k in weights)
        return total_score
    
    def score_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> List[Document]:
        """
        批量评分并更新文档
        
        Args:
            documents: 文档列表
            
        Returns:
            带有质量分数的文档列表
        """
        scored_docs = []
        for doc in documents:
            score = self.score(doc, **kwargs)
            scored_doc = Document(
                id=doc.id,
                content=doc.content,
                metadata=doc.metadata.copy(),
                quality_score=score
            )
            scored_docs.append(scored_doc)
        return scored_docs


# ==================== 增量更新管理器 ====================

class IncrementalUpdater:
    """
    增量更新管理器
    
    管理知识库的增量更新
    """
    
    def __init__(self):
        """初始化增量更新管理器"""
        self.document_hashes: Dict[str, str] = {}
        self.document_versions: Dict[str, int] = {}
    
    def _compute_hash(self, content: str) -> str:
        """计算内容哈希"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def check_update(
        self,
        documents: List[Document]
    ) -> Tuple[List[Document], List[Document], List[str]]:
        """
        检查文档更新状态
        
        Args:
            documents: 新文档列表
            
        Returns:
            (新增文档, 更新文档, 删除文档ID列表)
        """
        new_docs = []
        updated_docs = []
        
        current_ids = set()
        
        for doc in documents:
            current_ids.add(doc.id)
            content_hash = self._compute_hash(doc.content)
            
            if doc.id not in self.document_hashes:
                # 新文档
                new_docs.append(doc)
                self.document_hashes[doc.id] = content_hash
                self.document_versions[doc.id] = 1
            elif self.document_hashes[doc.id] != content_hash:
                # 内容已更新
                updated_docs.append(doc)
                self.document_hashes[doc.id] = content_hash
                self.document_versions[doc.id] += 1
        
        # 检查删除的文档
        deleted_ids = []
        for doc_id in list(self.document_hashes.keys()):
            if doc_id not in current_ids:
                deleted_ids.append(doc_id)
                del self.document_hashes[doc_id]
                del self.document_versions[doc_id]
        
        return new_docs, updated_docs, deleted_ids
    
    def get_version(self, doc_id: str) -> int:
        """获取文档版本号"""
        return self.document_versions.get(doc_id, 0)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_documents": len(self.document_hashes),
            "total_versions": sum(self.document_versions.values())
        }


# ==================== 使用示例 ====================

def demo_knowledge_base_optimization():
    """演示知识库优化"""
    
    print("=" * 60)
    print("Day 11: 知识库优化演示")
    print("=" * 60)
    
    # 准备示例文档
    documents = [
        Document(
            id="doc1",
            content="""
# 向量数据库简介

向量数据库是一种专门用于存储和检索向量嵌入的数据库系统。

## 主要特点
- 支持高效的相似度搜索
- 可扩展性强
- 支持多种索引类型

## 应用场景
1. 推荐系统
2. 图像检索
3. 自然语言处理
""",
            metadata={"category": "database", "author": "张三"}
        ),
        Document(
            id="doc2",
            content="""
def vector_search(query_vector, k=10):
    \"\"\"执行向量搜索\"\"\"
    # 计算相似度
    similarities = cosine_similarity(query_vector, vectors)
    # 返回 top-k
    return np.argsort(similarities)[-k:]

class VectorStore:
    def __init__(self, dimension):
        self.vectors = []
        self.dimension = dimension
    
    def add(self, vector):
        self.vectors.append(vector)
""",
            metadata={"category": "code", "language": "python"}
        ),
        Document(
            id="doc3",
            content="""
Q: 什么是混合检索？
A: 混合检索是结合向量检索和关键词检索的技术，能够同时利用语义理解和精确匹配的优势，提高检索质量。

Q: 为什么需要重排序？
A: 初步检索结果可能存在顺序不准确的问题，重排序可以更精确地计算查询与文档的相关性，提高最相关结果的排名。
""",
            metadata={"category": "qa"}
        ),
        Document(
            id="doc4",
            content="这是一个测试文档，用于演示文档去重功能。",  # 简短内容
            metadata={"category": "test"}
        ),
        Document(
            id="doc5",
            content="这是一个测试文档，用于演示文档去重功能。",  # 重复内容
            metadata={"category": "test"}
        ),
    ]
    
    # 1. 文档清洗
    print("\n" + "=" * 60)
    print("1. 文档清洗")
    print("=" * 60)
    
    cleaner = DocumentCleaner()
    
    # 模拟带有噪声的文档
    noisy_content = """
<div class="content">
<p>这是一个<strong>测试</strong>文档。</p>
<p>内容包含[1]引用标记。</p>
</div>
"""
    
    cleaned = cleaner.clean(noisy_content)
    print(f"原始内容:\n{noisy_content[:100]}...")
    print(f"\n清洗后:\n{cleaned}")
    
    # 2. 文档去重
    print("\n" + "=" * 60)
    print("2. 文档去重")
    print("=" * 60)
    
    deduplicator = DocumentDeduplicator(similarity_threshold=0.9)
    unique_docs, duplicate_docs = deduplicator.exact_deduplicate(documents)
    
    print(f"原始文档数: {len(documents)}")
    print(f"去重后文档数: {len(unique_docs)}")
    print(f"重复文档数: {len(duplicate_docs)}")
    
    # 3. 自适应切分
    print("\n" + "=" * 60)
    print("3. 自适应切分")
    print("=" * 60)
    
    chunker = AdaptiveChunker(
        default_chunk_size=200,
        default_overlap=20
    )
    
    for doc in documents[:3]:
        print(f"\n文档: {doc.id}")
        content_type = chunker._detect_content_type(doc.content)
        print(f"检测类型: {content_type}")
        
        chunks = chunker.chunk(doc)
        print(f"切分结果: {len(chunks)} 个块")
        for i, chunk in enumerate(chunks):
            print(f"  块 {i+1}: {len(chunk.content)} 字符")
    
    # 4. 父子索引切分
    print("\n" + "=" * 60)
    print("4. 父子索引切分")
    print("=" * 60)
    
    pc_chunker = ParentChildChunker(
        parent_size=300,
        child_size=100
    )
    
    parent_chunks, child_chunks = pc_chunker.chunk_with_parents(documents[0])
    
    print(f"父块数量: {len(parent_chunks)}")
    print(f"子块数量: {len(child_chunks)}")
    
    if child_chunks:
        print(f"\n子块示例:")
        print(f"  ID: {child_chunks[0].id}")
        print(f"  父块ID: {child_chunks[0].parent_id}")
        print(f"  内容: {child_chunks[0].content[:50]}...")
    
    # 5. 质量评分
    print("\n" + "=" * 60)
    print("5. 质量评分")
    print("=" * 60)
    
    scorer = QualityScorer()
    scored_docs = scorer.score_documents(documents)
    
    for doc in scored_docs:
        print(f"\n文档: {doc.id}")
        print(f"  质量分数: {doc.quality_score:.3f}")
        print(f"  完整性: {scorer.score_completeness(doc.content):.3f}")
        print(f"  信息密度: {scorer.score_information_density(doc.content):.3f}")
        print(f"  可读性: {scorer.score_readability(doc.content):.3f}")
    
    # 6. 增量更新
    print("\n" + "=" * 60)
    print("6. 增量更新管理")
    print("=" * 60)
    
    updater = IncrementalUpdater()
    
    # 第一次检查
    new_docs, updated_docs, deleted_ids = updater.check_update(documents[:3])
    print(f"第一次检查:")
    print(f"  新增: {len(new_docs)}")
    print(f"  更新: {len(updated_docs)}")
    print(f"  删除: {len(deleted_ids)}")
    
    # 修改一个文档
    modified_doc = Document(
        id="doc1",
        content="修改后的内容...",
        metadata={"category": "database"}
    )
    
    new_docs, updated_docs, deleted_ids = updater.check_update(
        [modified_doc, documents[1], documents[3]]
    )
    print(f"\n第二次检查（修改 doc1，新增 doc4，删除 doc3）:")
    print(f"  新增: {len(new_docs)} - {[d.id for d in new_docs]}")
    print(f"  更新: {len(updated_docs)} - {[d.id for d in updated_docs]}")
    print(f"  删除: {len(deleted_ids)} - {deleted_ids}")
    
    print(f"\n统计信息:")
    stats = updater.get_stats()
    print(f"  总文档数: {stats['total_documents']}")
    print(f"  总版本数: {stats['total_versions']}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo_knowledge_base_optimization()