#!/usr/bin/env python3
"""
PDF Processing Module for GitHub to Qdrant Pipeline

Supports multiple PDF extraction methods:
1. PyMuPDF (fastest, local)
2. PyPDFLoader (LangChain native, local)
3. Mistral OCR API (cloud, highest quality)
"""

import base64
import logging
import os
from typing import List, Dict, Any, Optional
from langchain.schema import Document

# PyMuPDF (fitz) - optional but recommended
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# PyPDFLoader from LangChain
try:
    from langchain_community.document_loaders import PyPDFLoader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# Mistral AI for OCR
try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False


class PDFProcessor:
    """
    Processes PDF files using multiple extraction methods.
    Supports local (PyMuPDF, PyPDFLoader) and cloud (Mistral OCR) processing.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize PDF processor with configuration.
        
        Args:
            config: PDF processing configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.pdf_config = config.get('pdf_processing', {})
        self.mode = self.pdf_config.get('mode', 'local')
        
        # Initialize Mistral client if configured
        self.mistral_client = None
        if self.pdf_config.get('cloud', {}).get('enabled') and MISTRAL_AVAILABLE:
            mistral_config = config.get('mistral_ai', {})
            api_key = mistral_config.get('api_key')
            if api_key:
                self.mistral_client = Mistral(api_key=api_key)
                self.logger.info("Mistral OCR client initialized")
        
        self._log_available_methods()
    
    def _log_available_methods(self):
        """Log available PDF processing methods."""
        methods = []
        if PYMUPDF_AVAILABLE:
            methods.append("PyMuPDF")
        if PYPDF_AVAILABLE:
            methods.append("PyPDFLoader")
        if self.mistral_client:
            methods.append("Mistral OCR")
        
        self.logger.info(f"Available PDF methods: {', '.join(methods)}")
        self.logger.info(f"PDF processing mode: {self.mode}")
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """
        Process a PDF file using configured method(s).
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of LangChain Document objects
        """
        if not self.pdf_config.get('enabled', True):
            self.logger.debug(f"PDF processing disabled, skipping: {file_path}")
            return []
        
        file_name = os.path.basename(file_path)
        
        # Skip macOS metadata files
        if file_name.startswith('._'):
            self.logger.debug(f"Skipping macOS metadata file: {file_name}")
            return []
        
        self.logger.info(f"Processing PDF: {file_name} (mode: {self.mode})")
        
        # Check if file should use cloud processing
        if self._should_force_cloud(file_name):
            self.logger.info(f"Pattern match - forcing cloud processing for: {file_name}")
            return self._process_cloud(file_path)
        
        # Process based on mode
        if self.mode == 'local':
            return self._process_local(file_path)
        elif self.mode == 'cloud':
            return self._process_cloud(file_path)
        elif self.mode == 'hybrid':
            return self._process_hybrid(file_path)
        else:
            self.logger.warning(f"Unknown mode: {self.mode}, using local")
            return self._process_local(file_path)
    
    def _should_force_cloud(self, file_name: str) -> bool:
        """Check if file matches cloud-only patterns."""
        if self.mode != 'hybrid':
            return False
        
        patterns = self.pdf_config.get('hybrid', {}).get('force_cloud_patterns', [])
        for pattern in patterns:
            # Simple wildcard matching
            if '*' in pattern:
                pattern_regex = pattern.replace('*', '.*')
                import re
                if re.match(pattern_regex, file_name, re.IGNORECASE):
                    return True
            elif pattern.lower() in file_name.lower():
                return True
        return False
    
    def _process_local(self, file_path: str) -> List[Document]:
        """Process PDF using local methods."""
        local_config = self.pdf_config.get('local', {})
        primary = local_config.get('primary_method', 'pymupdf')
        fallback = local_config.get('fallback_method', 'pypdfloader')
        
        # Try primary method
        docs = []
        if primary == 'pymupdf' and PYMUPDF_AVAILABLE:
            docs = self._extract_with_pymupdf(file_path)
            if self._validate_extraction(docs, local_config):
                return docs
        elif primary == 'pypdfloader' and PYPDF_AVAILABLE:
            docs = self._extract_with_pypdf(file_path)
            if self._validate_extraction(docs, local_config):
                return docs
        
        # Try fallback method
        if fallback == 'pypdfloader' and PYPDF_AVAILABLE:
            self.logger.info("Trying fallback: PyPDFLoader")
            docs = self._extract_with_pypdf(file_path)
            if self._validate_extraction(docs, local_config):
                return docs
        elif fallback == 'pymupdf' and PYMUPDF_AVAILABLE:
            self.logger.info("Trying fallback: PyMuPDF")
            docs = self._extract_with_pymupdf(file_path)
            if self._validate_extraction(docs, local_config):
                return docs
        
        self.logger.warning(f"Local extraction failed or insufficient for: {file_path}")
        return docs if docs else []
    
    def _process_cloud(self, file_path: str) -> List[Document]:
        """Process PDF using cloud methods (Mistral OCR)."""
        if not self.mistral_client:
            self.logger.warning("Mistral OCR not configured, falling back to local")
            return self._process_local(file_path)
        
        return self._extract_with_mistral(file_path)
    
    def _process_hybrid(self, file_path: str) -> List[Document]:
        """Process PDF using hybrid approach (local first, then cloud if needed)."""
        hybrid_config = self.pdf_config.get('hybrid', {})
        
        # Try local first if preferred
        docs = []
        if hybrid_config.get('prefer_local', True):
            docs = self._process_local(file_path)
            
            # Check quality
            if self._assess_quality(docs) >= hybrid_config.get('quality_threshold', 0.7):
                self.logger.info("Local extraction quality sufficient")
                return docs
            
            self.logger.info("Local quality below threshold, trying cloud")
        
        # Try cloud processing
        if self.mistral_client:
            cloud_docs = self._extract_with_mistral(file_path)
            if cloud_docs:
                return cloud_docs
        
        # Return best available result
        return docs if docs else []
    
    def _extract_with_pymupdf(self, file_path: str) -> List[Document]:
        """Extract text using PyMuPDF."""
        if not PYMUPDF_AVAILABLE:
            return []
        
        try:
            docs = []
            pdf_document = fitz.open(file_path)
            
            local_config = self.pdf_config.get('local', {})
            preserve_layout = local_config.get('preserve_layout', True)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                # Extract text with layout preservation
                if preserve_layout:
                    text = page.get_text("text", sort=True)
                else:
                    text = page.get_text()
                
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "page": page_num + 1,  # 1-indexed page numbers
                            "total_pages": len(pdf_document),
                            "extraction_method": "pymupdf",
                            "file_type": "pdf"
                        }
                    )
                    docs.append(doc)
            
            pdf_document.close()
            self.logger.info(f"PyMuPDF extracted {len(docs)} pages from {os.path.basename(file_path)}")
            return docs
            
        except Exception as e:
            self.logger.error(f"PyMuPDF extraction failed: {e}")
            return []
    
    def _extract_with_pypdf(self, file_path: str) -> List[Document]:
        """Extract text using PyPDFLoader."""
        if not PYPDF_AVAILABLE:
            return []
        
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Add extraction method to metadata
            for doc in docs:
                doc.metadata["extraction_method"] = "pypdfloader"
                doc.metadata["file_type"] = "pdf"
            
            self.logger.info(f"PyPDFLoader extracted {len(docs)} pages from {os.path.basename(file_path)}")
            return docs
            
        except Exception as e:
            self.logger.error(f"PyPDFLoader extraction failed: {e}")
            return []
    
    def _extract_with_mistral(self, file_path: str) -> List[Document]:
        """Extract text using Mistral OCR API."""
        if not self.mistral_client:
            return []
        
        try:
            # Encode PDF to base64
            with open(file_path, "rb") as pdf_file:
                pdf_base64 = base64.b64encode(pdf_file.read()).decode('utf-8')
            
            # Process with Mistral OCR
            self.logger.info(f"Sending PDF to Mistral OCR: {os.path.basename(file_path)}")
            
            response = self.mistral_client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{pdf_base64}"
                },
                include_image_base64=False  # Don't include images to save tokens
            )
            
            # Parse response - Mistral returns markdown
            # The response structure may vary, handle different formats
            if hasattr(response, 'content'):
                content = response.content
            elif hasattr(response, 'text'):
                content = response.text
            elif isinstance(response, dict):
                content = response.get('content', response.get('text', str(response)))
            else:
                content = str(response)
            
            # Create document(s) from response
            # For now, treat as single document - could be enhanced to split by pages
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "extraction_method": "mistral_ocr",
                    "file_type": "pdf",
                    "total_pages": "unknown"  # Mistral doesn't provide page count
                }
            )
            
            self.logger.info(f"Mistral OCR extracted {len(content)} chars from {os.path.basename(file_path)}")
            return [doc]
            
        except Exception as e:
            self.logger.error(f"Mistral OCR extraction failed: {e}")
            return []
    
    def _validate_extraction(self, docs: List[Document], config: Dict) -> bool:
        """Validate extraction quality."""
        if not docs:
            return False
        
        min_text = config.get('min_text_per_page', 50)
        total_text = sum(len(doc.page_content) for doc in docs)
        avg_text = total_text / len(docs) if docs else 0
        
        return avg_text >= min_text
    
    def _assess_quality(self, docs: List[Document]) -> float:
        """
        Assess extraction quality (0-1 score).
        
        Simple heuristic based on text density.
        Could be enhanced with more sophisticated metrics.
        """
        if not docs:
            return 0.0
        
        total_chars = sum(len(doc.page_content) for doc in docs)
        avg_chars = total_chars / len(docs)
        
        # Normalize to 0-1 (assuming good pages have 500+ chars)
        quality = min(avg_chars / 500, 1.0)
        
        return quality