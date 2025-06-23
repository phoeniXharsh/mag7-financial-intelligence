"""
Document processing and chunking for SEC filings
Intelligent text extraction and preparation for vector storage
"""

from typing import List, Dict, Optional, Tuple
import logging
import re
import uuid
from pathlib import Path
from dataclasses import dataclass, asdict
import json

try:
    from bs4 import BeautifulSoup
    import tiktoken
except ImportError as e:
    logging.warning(f"Document processor dependencies not installed: {e}")
    BeautifulSoup = None
    tiktoken = None

@dataclass
class DocumentChunk:
    """Represents a processed document chunk"""
    chunk_id: str
    text: str
    metadata: Dict
    source_file: str
    token_count: int
    section: str
    chunk_index: int

class DocumentProcessor:
    """Process and chunk SEC filings for vector storage"""

    def __init__(self,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 1000):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.logger = logging.getLogger(__name__)

        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        except:
            self.tokenizer = None
            self.logger.warning("tiktoken not available, using approximate token counting")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Approximate: 1 token ≈ 4 characters
            return len(text) // 4

    def process_filing(self, file_path: str) -> List[DocumentChunk]:
        """
        Process a single SEC filing into chunks

        Args:
            file_path: Path to the SEC filing file

        Returns:
            List of processed document chunks
        """
        self.logger.info(f"Processing filing: {file_path}")

        try:
            # Extract text from filing
            raw_text = self.extract_text(file_path)
            if not raw_text:
                self.logger.warning(f"No text extracted from {file_path}")
                return []

            # Extract base metadata from file path and content
            base_metadata = self.extract_metadata(file_path, raw_text)

            # Identify and extract sections
            sections = self.identify_sections(raw_text)

            all_chunks = []

            # Process each section separately
            for section_name, section_text in sections.items():
                if len(section_text.strip()) < self.min_chunk_size:
                    continue

                # Create chunks for this section
                section_chunks = self.create_chunks(
                    text=section_text,
                    metadata={**base_metadata, "section": section_name},
                    source_file=file_path,
                    section=section_name
                )

                all_chunks.extend(section_chunks)

            self.logger.info(f"Created {len(all_chunks)} chunks from {file_path}")
            return all_chunks

        except Exception as e:
            self.logger.error(f"Error processing filing {file_path}: {e}")
            return []

    def extract_text(self, file_path: str) -> str:
        """Extract raw text from filing"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return ""

            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Check if content is HTML/XML
            if '<html>' in content.lower() or '<?xml' in content.lower():
                return self._extract_from_html(content)
            else:
                return self._extract_from_text(content)

        except Exception as e:
            self.logger.error(f"Error extracting text from {file_path}: {e}")
            return ""

    def _extract_from_html(self, content: str) -> str:
        """Extract text from HTML content"""
        if not BeautifulSoup:
            # Fallback: simple HTML tag removal
            return re.sub(r'<[^>]+>', ' ', content)

        try:
            soup = BeautifulSoup(content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text and clean it up
            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            return text

        except Exception as e:
            self.logger.error(f"Error parsing HTML: {e}")
            # Fallback to regex
            return re.sub(r'<[^>]+>', ' ', content)

    def _extract_from_text(self, content: str) -> str:
        """Extract and clean text from plain text content"""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)

        # Remove control characters
        content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', content)

        return content.strip()

    def identify_sections(self, text: str) -> Dict[str, str]:
        """Identify and extract key sections from SEC filing"""
        sections = {}

        # Define section patterns for SEC filings
        section_patterns = {
            "Business Overview": [
                r"(?i)(?:item\s+1\.|part\s+i.*item\s+1)[\s\S]*?business",
                r"(?i)business\s+overview",
                r"(?i)our\s+business"
            ],
            "Risk Factors": [
                r"(?i)(?:item\s+1a\.|part\s+i.*item\s+1a)[\s\S]*?risk\s+factors",
                r"(?i)risk\s+factors"
            ],
            "Financial Performance": [
                r"(?i)(?:item\s+7\.|part\s+ii.*item\s+7)[\s\S]*?(?:management.*discussion|md&a)",
                r"(?i)management.*discussion.*analysis",
                r"(?i)financial\s+performance",
                r"(?i)results\s+of\s+operations",
                r"(?i)fiscal\s+year\s+\d{4}.*overview",
                r"(?i)total\s+revenue.*fiscal\s+\d{4}",
                r"(?i)revenue.*\$[\d,]+.*billion.*fiscal",
                r"(?i)net\s+income.*fiscal\s+\d{4}"
            ],
            "Financial Statements": [
                r"(?i)(?:item\s+8\.|part\s+ii.*item\s+8)[\s\S]*?financial\s+statements",
                r"(?i)consolidated\s+statements",
                r"(?i)balance\s+sheet",
                r"(?i)income\s+statement",
                r"(?i)cash\s+flow"
            ],
            "Revenue": [
                r"(?i)revenue(?:s)?[\s\S]{0,500}",
                r"(?i)net\s+sales[\s\S]{0,500}",
                r"(?i)total\s+revenue[\s\S]{0,500}",
                r"(?i)total\s+revenue.*\$[\d,]+.*billion",
                r"(?i)fiscal\s+\d{4}.*revenue.*\$[\d,]+",
                r"(?i)revenue.*\$[\d,]+.*billion.*compared",
                r"(?i)revenue.*increase.*billion"
            ]
        }

        # Extract sections using patterns
        for section_name, patterns in section_patterns.items():
            section_text = self._extract_section_by_patterns(text, patterns)
            if section_text and len(section_text.strip()) > self.min_chunk_size:
                sections[section_name] = section_text

        # If no specific sections found, create general sections
        if not sections:
            sections = self._create_general_sections(text)

        return sections

    def _extract_section_by_patterns(self, text: str, patterns: List[str]) -> str:
        """Extract section text using regex patterns"""
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Get text around the match
                start = max(0, match.start() - 500)
                end = min(len(text), match.end() + 2000)
                section_text = text[start:end]

                if len(section_text.strip()) > self.min_chunk_size:
                    return section_text.strip()

        return ""

    def _create_general_sections(self, text: str) -> Dict[str, str]:
        """Create general sections when specific sections can't be identified"""
        sections = {}

        # Split text into roughly equal parts
        text_length = len(text)
        section_size = max(5000, text_length // 5)  # Aim for 5 sections

        section_count = 0
        for i in range(0, text_length, section_size):
            section_text = text[i:i + section_size]
            if len(section_text.strip()) > self.min_chunk_size:
                sections[f"Section {section_count + 1}"] = section_text.strip()
                section_count += 1

                if section_count >= 10:  # Limit to 10 sections
                    break

        return sections

    def create_chunks(self,
                     text: str,
                     metadata: Dict,
                     source_file: str,
                     section: str) -> List[DocumentChunk]:
        """Create chunks from text with intelligent splitting"""
        chunks = []

        # Split text into sentences for better chunk boundaries
        sentences = self._split_into_sentences(text)

        current_chunk = ""
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # Check if adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk from current content
                chunk = self._create_chunk(
                    text=current_chunk.strip(),
                    metadata=metadata,
                    source_file=source_file,
                    section=section,
                    chunk_index=chunk_index
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + " " + sentence
                current_tokens = self.count_tokens(current_chunk)
                chunk_index += 1
            else:
                # Add sentence to current chunk
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens

        # Add final chunk if it has content
        if current_chunk.strip() and self.count_tokens(current_chunk) >= self.min_chunk_size:
            chunk = self._create_chunk(
                text=current_chunk.strip(),
                metadata=metadata,
                source_file=source_file,
                section=section,
                chunk_index=chunk_index
            )
            chunks.append(chunk)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better chunk boundaries"""
        # Simple sentence splitting - can be improved with NLTK
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Filter out very short sentences and clean up
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Minimum sentence length
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get overlap text from the end of current chunk"""
        if overlap_tokens <= 0:
            return ""

        words = text.split()
        if len(words) <= overlap_tokens:
            return text

        # Take last N words as overlap
        overlap_words = words[-overlap_tokens:]
        return " ".join(overlap_words)

    def _create_chunk(self,
                     text: str,
                     metadata: Dict,
                     source_file: str,
                     section: str,
                     chunk_index: int) -> DocumentChunk:
        """Create a DocumentChunk object"""
        chunk_id = str(uuid.uuid4())
        token_count = self.count_tokens(text)

        return DocumentChunk(
            chunk_id=chunk_id,
            text=text,
            metadata=metadata.copy(),
            source_file=source_file,
            token_count=token_count,
            section=section,
            chunk_index=chunk_index
        )

    def extract_metadata(self, file_path: str, text: str) -> Dict:
        """Extract metadata from filing path and content"""
        metadata = {
            "company": "",
            "ticker": "",
            "filing_type": "",
            "period": "",
            "filing_date": "",
            "source_file": str(file_path)
        }

        try:
            # Extract info from file path
            path = Path(file_path)
            path_parts = path.parts

            # Find company and filing type from path structure
            # Handle both SEC EDGAR structure and our generated structure
            for i, part in enumerate(path_parts):
                if part == "sec-edgar-filings" and i + 2 < len(path_parts):
                    metadata["ticker"] = path_parts[i + 1]
                    metadata["filing_type"] = path_parts[i + 2]
                    break
                elif part == "raw_filings" and i + 1 < len(path_parts):
                    # Our generated structure: data/raw_filings/AAPL/AAPL_10-K_2019.txt
                    metadata["ticker"] = path_parts[i + 1]
                    # Extract filing type and year from filename
                    filename = path.stem  # AAPL_10-K_2019
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        metadata["filing_type"] = parts[1]  # 10-K
                        metadata["year"] = parts[2]  # 2019
                    break

            # Extract metadata from text content
            metadata.update(self._extract_text_metadata(text))

            # Set company name based on ticker if not found in text
            if not metadata["company"] and metadata["ticker"]:
                company_names = {
                    "AAPL": "Apple Inc.",
                    "MSFT": "Microsoft Corporation",
                    "AMZN": "Amazon.com Inc.",
                    "GOOGL": "Alphabet Inc.",
                    "META": "Meta Platforms Inc.",
                    "NVDA": "NVIDIA Corporation",
                    "TSLA": "Tesla Inc."
                }
                metadata["company"] = company_names.get(metadata["ticker"], metadata["ticker"])

        except Exception as e:
            self.logger.error(f"Error extracting metadata: {e}")

        return metadata

    def _extract_text_metadata(self, text: str) -> Dict:
        """Extract metadata from text content"""
        metadata = {}

        # Extract company name
        company_patterns = [
            r'COMPANY CONFORMED NAME:\s*([^\n]+)',
            r'<COMPANY-NAME>([^<]+)</COMPANY-NAME>',
            r'(?:Company|Corporation|Inc\.|LLC):\s*([^\n]+)'
        ]

        for pattern in company_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata["company"] = match.group(1).strip()
                break

        # Extract filing date
        date_patterns = [
            r'FILED AS OF DATE:\s*(\d{8})',
            r'<FILING-DATE>(\d{4}-\d{2}-\d{2})',
            r'Filing Date:\s*(\d{4}-\d{2}-\d{2})'
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                # Convert YYYYMMDD to YYYY-MM-DD if needed
                if len(date_str) == 8 and date_str.isdigit():
                    metadata["filing_date"] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                else:
                    metadata["filing_date"] = date_str
                break

        # Extract period
        period_patterns = [
            r'PERIOD OF REPORT:\s*(\d{8})',
            r'<PERIOD>(\d{4}-\d{2}-\d{2})',
            r'For the.*?ended\s+(\w+\s+\d{1,2},\s+\d{4})'
        ]

        for pattern in period_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                period_str = match.group(1)
                # Convert YYYYMMDD to YYYY-MM-DD if needed
                if len(period_str) == 8 and period_str.isdigit():
                    metadata["period"] = f"{period_str[:4]}-{period_str[4:6]}-{period_str[6:8]}"
                else:
                    metadata["period"] = period_str
                break

        return metadata

    def process_multiple_filings(self, file_paths: List[str]) -> List[DocumentChunk]:
        """Process multiple SEC filings and return all chunks"""
        all_chunks = []

        for file_path in file_paths:
            self.logger.info(f"Processing {file_path}")
            chunks = self.process_filing(file_path)
            all_chunks.extend(chunks)

        self.logger.info(f"Processed {len(file_paths)} filings, created {len(all_chunks)} total chunks")
        return all_chunks

    def save_chunks_to_json(self, chunks: List[DocumentChunk], output_file: str):
        """Save processed chunks to JSON file"""
        try:
            chunks_data = [asdict(chunk) for chunk in chunks]

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved {len(chunks)} chunks to {output_file}")

        except Exception as e:
            self.logger.error(f"Error saving chunks to JSON: {e}")

    def load_chunks_from_json(self, input_file: str) -> List[DocumentChunk]:
        """Load processed chunks from JSON file"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)

            chunks = []
            for chunk_data in chunks_data:
                chunk = DocumentChunk(**chunk_data)
                chunks.append(chunk)

            self.logger.info(f"Loaded {len(chunks)} chunks from {input_file}")
            return chunks

        except Exception as e:
            self.logger.error(f"Error loading chunks from JSON: {e}")
            return []

    def get_processing_stats(self, chunks: List[DocumentChunk]) -> Dict:
        """Get statistics about processed chunks"""
        if not chunks:
            return {}

        stats = {
            "total_chunks": len(chunks),
            "total_tokens": sum(chunk.token_count for chunk in chunks),
            "avg_tokens_per_chunk": sum(chunk.token_count for chunk in chunks) / len(chunks),
            "companies": list(set(chunk.metadata.get("company", "Unknown") for chunk in chunks)),
            "filing_types": list(set(chunk.metadata.get("filing_type", "Unknown") for chunk in chunks)),
            "sections": list(set(chunk.section for chunk in chunks)),
            "min_tokens": min(chunk.token_count for chunk in chunks),
            "max_tokens": max(chunk.token_count for chunk in chunks)
        }

        return stats
