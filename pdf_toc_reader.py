import fitz  # PyMuPDF
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json
import re


def read_pdf_toc(pdf_path: str | Path) -> List[Dict]:
    """
    Read and return the table of contents from a PDF file using PyMuPDF with page ranges.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dictionaries containing TOC entries with their titles, page ranges, and children
    """
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()

    if not toc:
        doc.close()
        raise ValueError("No table of contents found in the PDF")

    def calculate_page_range(entries: List, start_idx: int) -> Tuple[Dict, int]:
        """
        Calculate the page range for a section and its children.
        Returns the processed entry and the index where processing stopped.
        """
        current_entry = entries[start_idx]
        level, title, start_page = current_entry[:3]

        # Initialize with the current entry's page as both start and end
        result = {
            "title": title,
            "start_page": start_page,
            "end_page": start_page,
            "children": [],
        }

        current_idx = start_idx + 1

        # Process children
        while current_idx < len(entries):
            child_level, _, child_page = entries[current_idx][:3]

            # If we encounter a higher level, we've moved to a new section
            if child_level <= level:
                break

            # If it's an immediate child (level + 1)
            if child_level == level + 1:
                child_entry, new_idx = calculate_page_range(entries, current_idx)
                result["children"].append(child_entry)
                current_idx = new_idx
                # Update parent's end page if child ends later
                result["end_page"] = max(result["end_page"], child_entry["end_page"])
            else:
                current_idx += 1

        # If this is a leaf node or the last entry, use the next section's start page
        # or the current page as the end page (as the last page and next section's start page may be the same)
        if not result["children"] and current_idx < len(entries):
            result["end_page"] = entries[current_idx][2]

        return result, current_idx

    try:
        # Process the root level entries
        root_entries = []
        idx = 0
        while idx < len(toc):
            if toc[idx][0] == 1:  # Root level
                entry, idx = calculate_page_range(toc, idx)
                root_entries.append(entry)
            else:
                idx += 1
        return root_entries
    finally:
        doc.close()


def print_toc_with_ranges(toc: List[Dict], level: int = 0) -> None:
    """
    Print the table of contents with page ranges.

    Args:
        toc: List of TOC entries with page ranges
        level: Current indentation level
    """
    for item in toc:
        indent = "  " * level
        range_info = f"(pages {item['start_page']}-{item['end_page']})"
        print(f"{indent}{item['title']} {range_info}")

        if item["children"]:
            print_toc_with_ranges(item["children"], level + 1)


def sanitize_filename(name: str, max_length: int = 30) -> str:
    """
    Convert a string to a valid filename across operating systems.

    Args:
        name: The string to convert
        max_length: Maximum length of the resulting filename

    Returns:
        A sanitized filename string
    """
    # Replace invalid characters with underscores
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    # Remove leading/trailing spaces and dots
    name = name.strip(". ")
    # Limit length
    if len(name) > max_length:
        name = name[:max_length].rstrip(". ")
    # Ensure we have a valid name
    if not name:
        name = "unnamed_section"
    return name


def create_directory_structure(
    pdf_path: str | Path,
    toc: List[Dict],
    output_path: str | Path = ".",
    doc: Optional[fitz.Document] = None,
    skip_keywords: set[str] = set(),
) -> None:
    """
    Create a directory structure matching the TOC and save page screenshots.

    Args:
        pdf_path: Path to the PDF file
        toc: Table of contents data
        output_path: Where to create the directory structure
        doc: Optional open PDF document (to avoid reopening)
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)

    # Use PDF filename as top directory name
    top_dir = output_path / sanitize_filename(pdf_path.stem)
    top_dir.mkdir(parents=True, exist_ok=True)

    # Initialize mapping dictionary
    path_mapping = {}

    # Open PDF if not provided
    should_close_doc = False
    if doc is None:
        doc = fitz.open(pdf_path)
        should_close_doc = True

    try:

        def get_trimmed_pixmap(page: fitz.Page) -> fitz.Pixmap:
            """Get a trimmed pixmap of the page content."""
            # Get the content bounds by joining all content bboxes
            content_bbox = fitz.Rect()  # start with empty rectangle
            for item in page.get_bboxlog():
                # item[1] contains the bbox coordinates
                content_bbox |= item[1]  # join this bbox into result

            # If no content was found, use the full page
            if content_bbox.is_empty:
                content_bbox = page.bound()

            # Add small margin (5 points) around content
            margin = 5
            content_bbox += (-margin, -margin, margin, margin)

            # Ensure we stay within page bounds
            page_bbox = page.bound()
            content_bbox &= page_bbox  # intersect with page bounds

            # Calculate matrix for 300 DPI
            zoom = 300 / 72
            matrix = fitz.Matrix(zoom, zoom)

            # Create pixmap of just the content area
            return page.get_pixmap(
                matrix=matrix,
                clip=content_bbox,
            )

        def process_toc_item(
            item: Dict[str, Any],
            current_path: Path,
            parent_path: str = "",
            full_toc_path: str = "",
        ) -> None:
            """Process a TOC item and create corresponding directory structure"""
            # Sanitize the current directory name
            if any(keyword in item["title"].lower() for keyword in skip_keywords):
                return
            dir_name = sanitize_filename(item["title"])
            dir_path = current_path / dir_name

            # Calculate the relative path using sanitized names
            if parent_path:
                # If there's a parent, join with sanitized parent path
                rel_path = f"{parent_path}/{dir_name}"
            else:
                # If no parent (root level), just use sanitized name
                rel_path = dir_name

            # Calculate the full unsanitized path
            if full_toc_path:
                full_unsanitized_path = f"{full_toc_path}/{item['title']}"
            else:
                full_unsanitized_path = item["title"]

            # Create directory
            dir_path.mkdir(exist_ok=True)

            # Add to mapping
            path_mapping[rel_path] = {
                "title": item["title"],
                "page_range": {"start": item["start_page"], "end": item["end_page"]},
                "parent": parent_path,
                "full_toc_path": full_unsanitized_path,
            }

            if item["children"]:
                # Process children
                for child in item["children"]:
                    process_toc_item(child, dir_path, rel_path, full_unsanitized_path)
            else:
                # Leaf node: save page screenshots
                for page_num in range(item["start_page"] - 1, item["end_page"]):
                    try:
                        page = doc[page_num]
                        pix = get_trimmed_pixmap(page)
                        image_path = dir_path / f"page_{page_num + 1}.jpg"
                        pix.save(str(image_path))
                    except Exception as e:
                        print(f"Error processing page {page_num + 1}: {e}")

        # Process all root entries
        for item in toc:
            process_toc_item(item, top_dir)

        # Save mapping file
        mapping_path = top_dir / "directory_mapping.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(path_mapping, f, indent=2, ensure_ascii=False)

    finally:
        if should_close_doc:
            doc.close()


def main():
    pdf_path = "fluent_python.pdf"
    try:
        # Open the document once and reuse it
        doc = fitz.open(pdf_path)
        toc = read_pdf_toc(pdf_path)

        print("Table of Contents with Page Ranges:")
        print("-" * 40)
        print_toc_with_ranges(toc)

        print("\nCreating directory structure...")
        create_directory_structure(
            pdf_path,
            toc,
            doc=doc,
            skip_keywords={
                "copyright",
                "about the author",
                "about the translator",
                "about the editor",
                "further reading",
                "acknowledgements",
                "about the cover",
                "about the ebook",
                "chapter summary",
                "new in this chapter",
                "table of contents",
            },
        )
        print("Done!")

        doc.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
