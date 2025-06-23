"""
SEC Filing Scraper for MAG7 companies
Automated download and processing of 10-K and 10-Q filings
"""

from typing import List, Dict, Optional, Tuple
import os
import re
import json
from datetime import datetime, date
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import time

try:
    from sec_edgar_downloader import Downloader
    import requests
    from bs4 import BeautifulSoup
except ImportError as e:
    logging.warning(f"SEC scraper dependencies not installed: {e}")
    Downloader = None

@dataclass
class FilingMetadata:
    """Metadata for a SEC filing"""
    company: str
    ticker: str
    filing_type: str
    filing_date: str
    period_end: str
    accession_number: str
    file_path: str
    url: str
    file_size: int
    sections: List[str]

class SECScraper:
    """Automated SEC filing scraper for MAG7 companies"""

    def __init__(self, data_dir: str = "data/sec_filings", company_name: str = "MAG7_Scraper", email: str = "user@example.com"):
        self.data_dir = Path(data_dir)
        self.company_name = company_name
        self.email = email
        self.logger = logging.getLogger(__name__)

        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.data_dir / "filings_metadata.json"

        # Initialize downloader
        # Note: sec-edgar-downloader creates files in current directory, not the specified path
        if Downloader:
            self.downloader = Downloader(self.company_name, self.email)
            # The actual download directory will be "./sec-edgar-filings"
            self.download_dir = Path("sec-edgar-filings")
        else:
            self.downloader = None
            self.download_dir = None
            self.logger.error("sec-edgar-downloader not available. Please install: pip install sec-edgar-downloader")

    def download_filings(self,
                        companies: List[str],
                        filing_types: List[str] = ["10-K", "10-Q"],
                        start_year: int = 2015,
                        end_year: int = 2025,
                        limit_per_company: Optional[int] = None) -> Dict[str, List[str]]:
        """
        Download SEC filings for specified companies

        Args:
            companies: List of company tickers
            filing_types: Types of filings to download
            start_year: Start year for filings
            end_year: End year for filings
            limit_per_company: Limit number of filings per company (for testing)

        Returns:
            Dictionary mapping company to list of downloaded file paths
        """
        if not self.downloader:
            self.logger.error("SEC downloader not initialized")
            return {}

        self.logger.info(f"Starting download for companies: {companies}")
        self.logger.info(f"Filing types: {filing_types}")
        self.logger.info(f"Date range: {start_year}-{end_year}")

        downloaded_files = {}

        for company in companies:
            self.logger.info(f"Processing {company}...")
            company_files = []

            for filing_type in filing_types:
                try:
                    self.logger.info(f"Downloading {filing_type} filings for {company}")

                    # Download filings
                    num_downloaded = self.downloader.get(
                        filing_type,
                        company,
                        limit=limit_per_company,
                        after=f"{start_year}-01-01",
                        before=f"{end_year}-12-31"
                    )

                    self.logger.info(f"Downloaded {num_downloaded} {filing_type} filings for {company}")

                    # Find downloaded files in the actual download directory
                    company_dir = self.download_dir / company / filing_type
                    if company_dir.exists():
                        for filing_dir in company_dir.iterdir():
                            if filing_dir.is_dir():
                                # Look for the main filing file (various extensions)
                                filing_files = (list(filing_dir.glob("*.txt")) +
                                              list(filing_dir.glob("*.htm")) +
                                              list(filing_dir.glob("*.html")))
                                if filing_files:
                                    main_file = filing_files[0]  # Take the first file
                                    company_files.append(str(main_file))

                    # Add delay to be respectful to SEC servers
                    time.sleep(0.1)

                except Exception as e:
                    self.logger.error(f"Error downloading {filing_type} for {company}: {e}")
                    continue

            downloaded_files[company] = company_files
            self.logger.info(f"Total files downloaded for {company}: {len(company_files)}")

        # Extract and save metadata
        self._extract_and_save_metadata(downloaded_files)

        return downloaded_files

    def _extract_and_save_metadata(self, downloaded_files: Dict[str, List[str]]):
        """Extract metadata from downloaded files and save to JSON"""
        all_metadata = []

        for company, files in downloaded_files.items():
            for file_path in files:
                try:
                    metadata = self.get_filing_metadata(file_path)
                    if metadata:
                        all_metadata.append(asdict(metadata))
                except Exception as e:
                    self.logger.error(f"Error extracting metadata from {file_path}: {e}")

        # Save metadata to JSON
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(all_metadata, f, indent=2)
            self.logger.info(f"Saved metadata for {len(all_metadata)} filings to {self.metadata_file}")
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")

    def get_filing_metadata(self, file_path: str) -> Optional[FilingMetadata]:
        """Extract metadata from SEC filing"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return None

            # Extract info from file path
            # Path structure: .../sec-edgar-filings/COMPANY/FILING_TYPE/ACCESSION/file.txt
            path_parts = file_path.parts

            # Find company, filing type, and accession number from path
            company_idx = None
            for i, part in enumerate(path_parts):
                if part == "sec-edgar-filings" and i + 1 < len(path_parts):
                    company_idx = i + 1
                    break

            if company_idx is None or company_idx + 2 >= len(path_parts):
                self.logger.warning(f"Could not parse path structure: {file_path}")
                return None

            ticker = path_parts[company_idx]
            filing_type = path_parts[company_idx + 1]
            accession_number = path_parts[company_idx + 2]

            # Read file content to extract metadata
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Extract metadata from content
            filing_date = self._extract_filing_date(content)
            period_end = self._extract_period_end(content)
            company_name = self._extract_company_name(content)
            sections = self._extract_sections(content)

            # Construct SEC URL
            url = f"https://www.sec.gov/Archives/edgar/data/{accession_number.replace('-', '')}"

            return FilingMetadata(
                company=company_name or ticker,
                ticker=ticker,
                filing_type=filing_type,
                filing_date=filing_date,
                period_end=period_end,
                accession_number=accession_number,
                file_path=str(file_path),
                url=url,
                file_size=file_path.stat().st_size,
                sections=sections
            )

        except Exception as e:
            self.logger.error(f"Error extracting metadata from {file_path}: {e}")
            return None

    def _extract_filing_date(self, content: str) -> str:
        """Extract filing date from SEC filing content"""
        patterns = [
            r'FILED AS OF DATE:\s*(\d{8})',
            r'FILING DATE:\s*(\d{4}-\d{2}-\d{2})',
            r'<FILING-DATE>(\d{4}-\d{2}-\d{2})',
            r'Filing Date:\s*(\d{4}-\d{2}-\d{2})'
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                # Convert YYYYMMDD to YYYY-MM-DD if needed
                if len(date_str) == 8 and date_str.isdigit():
                    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                return date_str

        return ""

    def _extract_period_end(self, content: str) -> str:
        """Extract period end date from SEC filing content"""
        patterns = [
            r'PERIOD OF REPORT:\s*(\d{8})',
            r'<PERIOD>(\d{4}-\d{2}-\d{2})',
            r'Period Ended:\s*(\d{4}-\d{2}-\d{2})',
            r'For the.*?ended\s+(\w+\s+\d{1,2},\s+\d{4})'
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                # Convert YYYYMMDD to YYYY-MM-DD if needed
                if len(date_str) == 8 and date_str.isdigit():
                    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                return date_str

        return ""

    def _extract_company_name(self, content: str) -> str:
        """Extract company name from SEC filing content"""
        patterns = [
            r'<COMPANY-NAME>(.*?)</COMPANY-NAME>',
            r'COMPANY CONFORMED NAME:\s*(.*?)(?:\n|$)',
            r'Company Name:\s*(.*?)(?:\n|$)'
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return ""

    def _extract_sections(self, content: str) -> List[str]:
        """Extract section names from SEC filing content"""
        sections = []

        # Common SEC filing sections
        section_patterns = [
            r'(?:PART|ITEM)\s+(\d+[A-Z]?)\.\s*([^\n]+)',
            r'(?:ITEM|Part)\s+(\d+)\s*[-–]\s*([^\n]+)',
            r'<b>(?:ITEM|PART)\s+(\d+[A-Z]?)\.\s*([^<]+)</b>'
        ]

        for pattern in section_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                section_num = match.group(1)
                section_title = match.group(2).strip()
                if section_title and len(section_title) < 200:  # Reasonable title length
                    sections.append(f"Item {section_num}: {section_title}")

        # Remove duplicates while preserving order
        seen = set()
        unique_sections = []
        for section in sections:
            if section not in seen:
                seen.add(section)
                unique_sections.append(section)

        return unique_sections[:20]  # Limit to first 20 sections

    def list_downloaded_filings(self) -> List[str]:
        """List all downloaded filing files"""
        filings = []

        # Look for files in the actual download directory
        if self.download_dir and self.download_dir.exists():
            for company_dir in self.download_dir.iterdir():
                if company_dir.is_dir():
                    for filing_type_dir in company_dir.iterdir():
                        if filing_type_dir.is_dir():
                            for accession_dir in filing_type_dir.iterdir():
                                if accession_dir.is_dir():
                                    # Find filing files (multiple extensions)
                                    filing_files = (list(accession_dir.glob("*.txt")) +
                                                  list(accession_dir.glob("*.htm")) +
                                                  list(accession_dir.glob("*.html")))
                                    filings.extend([str(f) for f in filing_files])

        return filings

    def get_download_stats(self) -> Dict:
        """Get statistics about downloaded filings"""
        filings = self.list_downloaded_filings()

        stats = {
            "total_filings": len(filings),
            "companies": set(),
            "filing_types": set(),
            "total_size_mb": 0
        }

        for filing_path in filings:
            path = Path(filing_path)
            if path.exists():
                # Extract company and filing type from path
                parts = path.parts
                for i, part in enumerate(parts):
                    if part == "sec-edgar-filings" and i + 2 < len(parts):
                        stats["companies"].add(parts[i + 1])
                        stats["filing_types"].add(parts[i + 2])
                        break

                # Add file size
                stats["total_size_mb"] += path.stat().st_size / (1024 * 1024)

        # Convert sets to lists for JSON serialization
        stats["companies"] = list(stats["companies"])
        stats["filing_types"] = list(stats["filing_types"])
        stats["total_size_mb"] = round(stats["total_size_mb"], 2)

        return stats

    def load_metadata(self) -> List[Dict]:
        """Load filing metadata from JSON file"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading metadata: {e}")

        return []

    def get_filings_by_company(self, company: str) -> List[Dict]:
        """Get all filings for a specific company"""
        metadata = self.load_metadata()
        return [filing for filing in metadata if filing.get("ticker", "").upper() == company.upper()]

    def get_filings_by_type(self, filing_type: str) -> List[Dict]:
        """Get all filings of a specific type"""
        metadata = self.load_metadata()
        return [filing for filing in metadata if filing.get("filing_type", "").upper() == filing_type.upper()]

    def cleanup_downloads(self):
        """Clean up downloaded files (for testing/development)"""
        import shutil

        # Clean up the actual download directory
        if self.download_dir and self.download_dir.exists():
            shutil.rmtree(self.download_dir)
            self.logger.info("Cleaned up downloaded SEC filings")

        if self.metadata_file.exists():
            self.metadata_file.unlink()
            self.logger.info("Cleaned up metadata file")
