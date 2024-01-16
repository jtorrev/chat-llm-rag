import json
import os
import re
import requests
import zipfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from typing import IO

import chardet
from pypdf import PdfReader
from pypdf.errors import PdfStreamError
from langchain_community.document_loaders.unstructured import UnstructuredAPIFileIOLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

from danswer.utils.logger import setup_logger


logger = setup_logger()


def extract_metadata(line: str) -> dict | None:
    html_comment_pattern = r"<!--\s*DANSWER_METADATA=\{(.*?)\}\s*-->"
    hashtag_pattern = r"#DANSWER_METADATA=\{(.*?)\}"

    html_comment_match = re.search(html_comment_pattern, line)
    hashtag_match = re.search(hashtag_pattern, line)

    if html_comment_match:
        json_str = html_comment_match.group(1)
    elif hashtag_match:
        json_str = hashtag_match.group(1)
    else:
        return None

    try:
        return json.loads("{" + json_str + "}")
    except json.JSONDecodeError:
        return None


def read_pdf_file(file: IO[Any], file_name: str, pdf_pass: str | None = None) -> str:
    try:
        pdf_reader = PdfReader(file)

        # If marked as encrypted and a password is provided, try to decrypt
        if pdf_reader.is_encrypted and pdf_pass is not None:
            decrypt_success = False
            if pdf_pass is not None:
                try:
                    decrypt_success = pdf_reader.decrypt(pdf_pass) != 0
                except Exception:
                    logger.error(f"Unable to decrypt pdf {file_name}")
            else:
                logger.info(f"No Password available to to decrypt pdf {file_name}")

            if not decrypt_success:
                # By user request, keep files that are unreadable just so they
                # can be discoverable by title.
                return ""

        return "\n".join(page.extract_text() for page in pdf_reader.pages)
    except PdfStreamError:
        logger.exception(f"PDF file {file_name} is not a valid PDF")
    except Exception:
        logger.exception(f"Failed to read PDF {file_name}")

    # File is still discoverable by title
    # but the contents are not included as they cannot be parsed
    return ""


def is_macos_resource_fork_file(file_name: str) -> bool:
    return os.path.basename(file_name).startswith("._") and file_name.startswith(
        "__MACOSX"
    )


def load_files_from_zip(
    zip_location: str | Path,
    ignore_macos_resource_fork_files: bool = True,
    ignore_dirs: bool = True,
) -> Generator[tuple[zipfile.ZipInfo, IO[Any]], None, None]:
    with zipfile.ZipFile(zip_location, "r") as zip_file:
        for file_info in zip_file.infolist():
            with zip_file.open(file_info.filename, "r") as file:
                if ignore_dirs and file_info.is_dir():
                    continue

                if ignore_macos_resource_fork_files and is_macos_resource_fork_file(
                    file_info.filename
                ):
                    continue
                yield file_info, file


def detect_encoding(file_path: str | Path) -> str:
    with open(file_path, "rb") as file:
        raw_data = file.read(50000)  # Read a portion of the file to guess encoding
    return chardet.detect(raw_data)["encoding"] or "utf-8"


def read_file(
    file_reader: IO[Any], encoding: str = "utf-8", errors: str = "replace"
) -> tuple[str, dict]:
    metadata = {}
    file_content_raw = ""
    for ind, line in enumerate(file_reader):
        try:
            line = line.decode(encoding) if isinstance(line, bytes) else line
        except UnicodeDecodeError:
            line = (
                line.decode(encoding, errors=errors)
                if isinstance(line, bytes)
                else line
            )

        if ind == 0:
            metadata_or_none = extract_metadata(line)
            if metadata_or_none is not None:
                metadata = metadata_or_none
            else:
                file_content_raw += line
        else:
            file_content_raw += line

    return file_content_raw, metadata


def read_file_from_unstructured(
    file_reader: IO[Any],
    file_name:str,
    extension:str
) -> str:
    if extension in [".xlsx",".xls"]:
        loader = UnstructuredExcelLoader(file_path=file_name, mode="single")
        docs = loader.load()
        file_content_raw = docs[0].page_content
    elif extension in [".doc", ".docx"]:
        loader = UnstructuredWordDocumentLoader(file_path=file_name, mode="single")
        docs = loader.load()
        file_content_raw = docs[0].page_content
    elif extension in [".ppt", ".pptx"]:
        loader = UnstructuredPowerPointLoader(file_path=file_name, mode="single")
        docs = loader.load()
        file_content_raw = docs[0].page_content   
    else:
        file_content_raw = ""
        url = "http://127.0.0.1:8005/general/v0/general"
        payload = {'strategy': 'hi_res'}
        files = {'files': file_reader}
        r = requests.post(url, files=files, json=payload)
        results = r.json()
        result = [x["text"] for x in results]
        file_content_raw = "\n".join(result)
    return file_content_raw
    # loader = UnstructuredAPIFileIOLoader(file=file_reader,file_path=file_name,mode="single",strategy="hi_res")
    # docs = loader.load()
    # return docs[0].page_content
