import base64
from dataclasses import dataclass
from pathlib import Path
import re
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
import json
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)
from typing import List, Dict, Optional, Set, TypedDict, Union
import logging
from datetime import datetime
import sys
from tqdm import tqdm
import csv
import markdown
import shutil
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name, PythonLexer


# Type definitions
class Flashcard(TypedDict, total=False):
    Front: str
    FrontCode: str
    Back: str
    BackCode: str
    Comments: str


@dataclass
class AnkiDeckConfig:
    """Configuration for Anki deck creation."""

    fields: List[str]
    strip_markdown_code_fields: Set[str]
    render_markdown: bool = True
    syntax_highlight_html: bool = False
    separator: str = ":"
    mapping_file: str = "directory_mapping.json"


def encode_image(image_path: str | Path) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
def create_flashcards_with_backoff(
    client: OpenAI, messages: List[Dict], max_tokens: int = 2000
):
    """Make API call with exponential backoff retry logic."""
    try:
        return client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
    except RateLimitError:
        print("Rate limit hit, backing off...")
        raise
    except APIConnectionError as e:
        print(f"Connection error: {e.__cause__}")
        raise
    except APIError as e:
        print(f"API error: {e}")
        raise


def setup_logging(log_file: Optional[str] = None) -> None:
    """Set up logging configuration."""
    # Create formatter for file logging
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # File handler only
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def is_leaf_section(rel_path: str, section_info: Dict, mappings: Dict) -> bool:
    """
    Determine if a section is a leaf section in the TOC hierarchy.

    Args:
        rel_path: The relative path key of the section
        section_info: The section's information dictionary
        mappings: The complete mappings dictionary

    Returns:
        bool: True if this is a leaf section, False otherwise
    """
    # Not a leaf if it's a parent of any other section
    if any(other_info["parent"] == rel_path for other_info in mappings.values()):
        return False

    # Not a leaf if it's a root section
    if section_info["parent"] == "":
        return False

    # Must be in the actual hierarchy
    return any(
        other_info["full_toc_path"].startswith(section_info["full_toc_path"])
        for other_info in mappings.values()
    )


def create_flashcards_for_book(
    book_title: str,
    mapping_file: str | Path,
    resume_from: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Create flashcards for all sections in a book using the mapping file.
    """
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_file is None:
        log_file = f"flashcard_creation_{timestamp}.log"
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    logger.info(f"Starting flashcard creation for book: {book_title}")
    logger.info(f"Using mapping file: {mapping_file}")
    if resume_from:
        logger.info(f"Resuming from section: {resume_from}")

    mapping_file = Path(mapping_file)
    if not mapping_file.exists():
        logger.error(f"Mapping file not found: {mapping_file}")
        raise ValueError(f"Mapping file not found: {mapping_file}")

    # Get the root directory (parent of mapping file)
    root_dir = mapping_file.parent

    try:
        with open(mapping_file) as f:
            mappings = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in mapping file: {e}")
        raise ValueError(f"Invalid JSON in mapping file: {mapping_file}")
    except Exception as e:
        logger.error(f"Error reading mapping file: {e}")
        raise

    # Track progress
    total_sections = sum(
        1
        for rel_path, info in mappings.items()
        if is_leaf_section(rel_path, info, mappings)
    )
    processed_sections = set()
    resume_found = resume_from is None

    # Create progress bar
    pbar = tqdm(total=total_sections, desc="Processing sections")
    current_section_display = ""

    try:
        # Process each section in the mapping file
        for rel_path, section_info in mappings.items():
            # Skip if not a leaf section
            if not is_leaf_section(rel_path, section_info, mappings):
                continue

            # Skip if we've already processed this section
            if rel_path in processed_sections:
                continue

            # Handle resume logic
            if not resume_found:
                if rel_path == resume_from:
                    resume_found = True
                    logger.info(
                        f"Resuming from section: {section_info['full_toc_path']}"
                    )
                else:
                    logger.debug(
                        f"Skipping section (before resume point): {section_info['full_toc_path']}"
                    )
                    continue

            # Update progress display
            current_section_display = f"Current: {section_info['full_toc_path']}"
            pbar.set_postfix_str(current_section_display)
            logger.info(f"Processing section: {section_info['full_toc_path']}")

            try:
                section_dir = root_dir / rel_path
                if not section_dir.exists():
                    logger.warning(f"Directory not found, skipping: {section_dir}")
                    continue

                # Check if flashcards already exist
                flashcards_file = section_dir / "flashcards.json"
                if flashcards_file.exists():
                    logger.info(
                        f"Flashcards already exist for section, skipping: {section_info['full_toc_path']}"
                    )
                    processed_sections.add(rel_path)
                    pbar.update(1)
                    continue

                create_flashcards_from_directory(
                    directory_path=section_dir,
                    book_title=book_title,
                    chapter_title=section_info["full_toc_path"],
                )
                processed_sections.add(rel_path)
                logger.info(
                    f"Successfully created flashcards for: {section_info['full_toc_path']}"
                )
                pbar.update(1)

                # Add a small delay between sections to help avoid rate limits
                time.sleep(1)

            except Exception as e:
                logger.error(
                    f"Error processing section {section_info['full_toc_path']}: {e}",
                    exc_info=True,
                )
                # Save progress information
                progress_file = root_dir / f"flashcard_progress_{timestamp}.json"
                progress_info = {
                    "last_processed": rel_path,
                    "processed_sections": list(processed_sections),
                    "total_sections": total_sections,
                    "completed_sections": len(processed_sections),
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
                try:
                    with open(progress_file, "w") as f:
                        json.dump(progress_info, f, indent=2)
                    logger.info(f"Progress saved to: {progress_file}")
                except Exception as save_error:
                    logger.error(f"Error saving progress: {save_error}")

                # Close progress bar before raising error
                pbar.close()
                raise

    finally:
        pbar.close()

    logger.info(f"Completed flashcard creation for book: {book_title}")
    logger.info(f"Processed {len(processed_sections)}/{total_sections} sections")


def validate_flashcard_json(content: str) -> bool:
    """
    Validate that the content is valid JSON and has the expected structure.

    Args:
        content: JSON string to validate

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        data = json.loads(content)
        # Validate it's a list/object containing cards
        if not isinstance(data, (list, dict)):
            return False
        return True
    except json.JSONDecodeError:
        return False


def create_flashcards_from_directory(
    directory_path: str | Path,
    book_title: str,
    chapter_title: str,
    max_retries: int = 3,
) -> None:
    """
    Create Anki flashcards from images in a directory using GPT-4V.

    Args:
        directory_path: Path to directory containing images
        book_title: Title of the book
        chapter_title: Full path of the chapter/section in the book
        max_retries: Maximum number of retries for invalid JSON responses
    """
    directory_path = Path(directory_path)
    if not directory_path.exists():
        raise ValueError(f"Directory not found: {directory_path}")

    # Get all jpg files and sort them by page number
    image_files = sorted(
        directory_path.glob("*.jpg"),
        key=lambda x: int(re.search(r"page_(\d+)", x.name).group(1)),
    )

    if not image_files:
        raise ValueError(f"No images found in {directory_path}")

    # Create OpenAI client
    client = OpenAI()

    # Prepare the message with context and all images
    message_content = [
        {
            "type": "text",
            "text": f"""You are an expert in computer science education and Python, helping students master `{book_title}` using spaced repetition. You are presented with screenshots of the chapter titled "`{chapter_title}`". Create flashcards that cover key concepts and insights from the chapter, up to 3 cards per page.

**Guidelines:**
- **Variety:** Include questions ranging from basic to complex, presenting facts from different angles.
- **Self-Containment:** Ensure each question is self-contained, similar to exam questions.
  - If a question references specific code snippets or classes from the chapter (e.g., `ClubMember`), include the necessary code in the **FrontCode** field.
  - **No Answer Leaks:** Do not include code or comments in **FrontCode** that reveal the answer to the question. Such code should instead be placed in **BackCode**.
- **Techniques:** Use examples and mnemonic devices where possible.
- **Clarity:** Craft concise, engaging questions and answers, simplifying complex ideas into clear maxims.

**Output Format:** Provide a JSON list of cards, each with the following fields using Markdown syntax where applicable:
- **Front:** A concise, self-contained question that does not require additional context.
- **FrontCode:** All necessary code snippets from the chapter needed to understand and answer the question. Ensure that **FrontCode** does not contain code or comments that leak the answer.
- **Back:** A brief answer or explanation.
- **BackCode:** Optional concise, properly formatted code example from the chapter that illustrates the answer. Leave empty if it is included in **FrontCode**.
- **Comments:** Optional mnemonics or study notes to aid learning.

**Note:** Base each card solely on the content of "`{chapter_title}`", ignoring any overlapping content from adjacent chapters in the screenshots.
""",
        }
    ]

    # Add each image to the message
    for image_path in image_files:
        base64_image = encode_image(image_path)
        message_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        )

    # Make the API call with retry logic
    attempts = 0
    last_error = None

    while attempts < max_retries:
        try:
            response = create_flashcards_with_backoff(
                client=client,
                messages=[{"role": "user", "content": message_content}],
            )

            content = response.choices[0].message.content

            # Validate JSON response
            if validate_flashcard_json(content):
                # Save the flashcards
                flashcards_file = directory_path / "flashcards.json"
                with open(flashcards_file, "w", encoding="utf-8") as f:
                    f.write(content)
                return

            # If JSON is invalid, try again
            print(
                f"Attempt {attempts + 1}/{max_retries}: Invalid JSON response, retrying..."
            )
            attempts += 1
            time.sleep(1)  # Small delay before retry

        except (RateLimitError, APIConnectionError) as e:
            print(f"Failed due to rate limit or connection error: {e}")
            raise
        except APIError as e:
            print(f"API error: {e}")
            raise
        except Exception as e:
            last_error = e
            print(f"Unexpected error on attempt {attempts + 1}: {e}")
            attempts += 1
            if attempts < max_retries:
                time.sleep(1)
                continue
            break

    # If we get here, all attempts failed
    error_msg = f"Failed to generate valid flashcards after {max_retries} attempts"
    if last_error:
        error_msg += f": {str(last_error)}"
    raise ValueError(error_msg)


def strip_markdown_code_block(text: str) -> str:
    """
    Strip markdown code block syntax if present.

    Args:
        text: Text that might contain a markdown code block

    Returns:
        Text with code block syntax removed if it was a code block,
        otherwise the original text
    """
    if not text:
        return text

    # Check if it's a code block (starts with ```language_name and ends with ```)
    lines = text.strip().split("\n")
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        # Remove first and last lines (the ``` markers)
        return "\n".join(lines[1:-1])
    return text


def render_markdown_text(text: str) -> str:
    """
    Render markdown text to HTML.

    Args:
        text: Markdown text to render

    Returns:
        HTML-rendered text
    """
    if not text:
        return text

    # Use markdown package with common extensions
    return markdown.markdown(
        text,
        extensions=[
            "markdown.extensions.fenced_code",  # for code blocks
            "markdown.extensions.tables",  # for tables
            "markdown.extensions.nl2br",  # newlines to <br>
        ],
    )


def syntax_highlight_code(code: str, language: str = "python") -> str:
    """
    Highlight code using Pygments with inline styles.

    Args:
        code: The code to highlight
        language: Programming language for syntax highlighting

    Returns:
        HTML string with syntax highlighted code
    """
    try:
        lexer = get_lexer_by_name(language)
    except ValueError:
        # Fallback to Python if language not found
        lexer = PythonLexer()

    # Use inline styles for Anki compatibility
    formatter = HtmlFormatter(
        style="default",
        cssclass="highlight",
        noclasses=True,  # Important: embed styles inline
        nowrap=True,  # Don't wrap in a div
    )

    return highlight(code, lexer, formatter)


def process_section_images(
    section_dir: Path,
    img_assets_dir: Path,
    root_dir_name: str,
) -> str:
    """
    Process images for a section, copying them to img_assets with Anki-compatible names.

    Args:
        section_dir: Directory containing the section's images
        img_assets_dir: Target directory for processed images
        root_dir_name: Name of the root directory (used in image naming)

    Returns:
        HTML string containing image tags for all processed images
    """
    image_files = sorted(
        section_dir.glob("*.jpg"),
        key=lambda x: int(re.search(r"page_(\d+)", x.name).group(1)),
    )

    image_tags = []
    for img_path in image_files:
        # Extract page number
        page_num = re.search(r"page_(\d+)", img_path.name).group(1)

        # Create filename for Anki
        new_filename = f"{root_dir_name}_{page_num}.jpg"
        new_img_path = img_assets_dir / new_filename

        # Copy image to img_assets directory
        shutil.copy2(img_path, new_img_path)

        # Use just the filename in the img tag
        image_tags.append(f'<img src="{new_filename}">')

    return "\n".join(image_tags)


def process_card_field(
    value: str,
    field: str,
    strip_markdown_code_fields: Set[str],
    render_markdown: bool,
    syntax_highlight_html: bool,
) -> str:
    """
    Process a single field value according to formatting rules.

    Args:
        value: The field value to process
        field: The name of the field
        strip_markdown_code_fields: Set of fields to strip markdown code from
        render_markdown: Whether to render markdown as HTML
        syntax_highlight_html: Whether to apply syntax highlighting

    Returns:
        Processed field value
    """
    if not value:
        return value

    if field in strip_markdown_code_fields:
        value = strip_markdown_code_block(value)
        if value:
            if syntax_highlight_html:
                value = syntax_highlight_code(value)
                value = f"<pre>{value}</pre>"
            else:
                value = value.replace("\n", "<br>")
                value = value.replace(" ", "&nbsp;")
                value = f"<pre><code>{value}</code></pre>"
    elif render_markdown:
        value = render_markdown_text(value)

    return value


def create_anki_deck(
    directory: str | Path,
    config: AnkiDeckConfig,
) -> List[str]:
    """
    Create an Anki-importable deck from flashcards.json files, including media assets.

    This function:
    1. Creates a CSV file with flashcard content
    2. Copies and renames images to a flat img_assets directory for Anki import
    3. Processes markdown and code formatting
    4. Maintains source tracking and image references

    Args:
        directory: Root directory containing the mapping file and section directories
        config: Configuration for deck creation

    Returns:
        List of directories that don't have flashcards.json

    Creates:
        - deck.csv: The Anki-importable deck file
        - img_assets/: Directory containing renamed images for Anki import
    """
    logger = logging.getLogger(__name__)
    directory = Path(directory)
    mapping_path = directory / config.mapping_file

    # Create img_assets directory
    img_assets_dir = directory / "img_assets"
    img_assets_dir.mkdir(exist_ok=True)

    if not mapping_path.exists():
        raise ValueError(f"Mapping file not found: {mapping_path}")

    try:
        with open(mapping_path) as f:
            mappings = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in mapping file: {mapping_path}")

    # Prepare CSV
    csv_path = directory / "deck.csv"
    missing_flashcards = []

    # Add source and source_images to fields
    all_fields = config.fields + ["source", "source_images"]

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=all_fields,
            delimiter=config.separator,
            quoting=csv.QUOTE_ALL,
            escapechar="\\",
            doublequote=True,
        )
        writer.writeheader()

        # Process each leaf section
        for rel_path, section_info in mappings.items():
            # Skip if not a leaf section
            if not is_leaf_section(rel_path, section_info, mappings):
                continue

            section_dir = directory / rel_path
            flashcards_path = section_dir / "flashcards.json"

            # Check if flashcards exist
            if not flashcards_path.exists():
                missing_flashcards.append(str(section_dir))
                continue

            try:
                # Process section
                logger.info(f"Processing section: {section_info['full_toc_path']}")

                # Load and validate flashcards
                with open(flashcards_path, encoding="utf-8") as f:
                    flashcards_data = json.load(f)

                cards = extract_flashcards(flashcards_data)
                if not cards:
                    logger.warning(f"No valid cards found in {flashcards_path}")
                    missing_flashcards.append(str(section_dir))
                    continue

                # Process images
                image_html = process_section_images(
                    section_dir, img_assets_dir, directory.name
                )

                # Process each card
                for card in cards:
                    row = {}
                    # Process fields
                    for field in config.fields:
                        value = card.get(field, "")
                        row[field] = process_card_field(
                            value,
                            field,
                            config.strip_markdown_code_fields,
                            config.render_markdown,
                            config.syntax_highlight_html,
                        )

                    # Add source information
                    row["source"] = section_info["full_toc_path"]
                    row["source_images"] = image_html

                    writer.writerow(row)

            except Exception as e:
                logger.error(f"Error processing {section_dir}: {e}", exc_info=True)
                missing_flashcards.append(str(section_dir))
                continue

    return missing_flashcards


def extract_flashcards(data: Union[Dict, List]) -> List[Flashcard]:
    """
    Extract flashcards from JSON data, handling both dict and list formats.

    Args:
        data: JSON data that might contain flashcards

    Returns:
        List of validated flashcards
    """
    if isinstance(data, dict):
        # If it's a dict, find the first list value
        cards = next((v for v in data.values() if isinstance(v, list)), [])
    elif isinstance(data, list):
        cards = data
    else:
        cards = []

    # Validate cards
    return [
        card
        for card in cards
        if isinstance(card, dict) and "Front" in card and "Back" in card
    ]


def main():
    # Example usage
    book_title = "Fluent Python: Clear, Concise, and Effective Programming"
    directory = "fluent_python"

    # Create configuration
    config = AnkiDeckConfig(
        fields=["Front", "FrontCode", "Back", "BackCode", "Comments"],
        strip_markdown_code_fields={"FrontCode", "BackCode"},
        render_markdown=True,
        syntax_highlight_html=True,
        separator=",",
    )

    try:
        # Create flashcards first
        create_flashcards_for_book(
            book_title=book_title,
            mapping_file=f"{directory}/directory_mapping.json",
            log_file=f"{directory}/flashcard_creation.log",
        )

        # Then create the Anki deck
        missing = create_anki_deck(
            directory=directory,
            config=config,
        )

        if missing:
            print("\nDirectories missing flashcards:")
            for dir_path in missing:
                print(f"  - {dir_path}")

    except Exception as e:
        logging.error("Job failed", exc_info=True)
        print("Job failed", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
