# Book-To-Anki

Motivation:
* There are books about things I understand to some extent (["Do not learn if you do not understand"](https://www.supermemo.com/en/blog/twenty-rules-of-formulating-knowledge)!) and want to read, but are hundreds of pages and I do not have enough activation energy.
* Incremental reading & making flashcards take a lot of time
Solution:
* We can generate cards with LLMs cheaply. Just suspend or delete ones that don't work.
* Let's read the book while studying cards incrementally by putting book sections on cards.

What this tool does:
- Takes a pdf file w/ a valid Table of Contents. Finds all the leaf sections and their page ranges
- Generates a directory tree matching ToS structure; saves cropped bitmap images to directories of leaf sections
- For each leaf section, uses the OpenAI multimodal API to generate flashcards; saves to `flashcards.json` of the section's directory
- Tangle CSV that can be imported to Anki
  - Code fields can be rendered to HTML using Pygments to support syntax highlighting

Each note is also associated with
- `source`: ToS path to section
- `source_imgs`: image tags pointing to each page in the section

page images are renamed & copied to `img_assets`; you can copy them to your Anki media folder to be able to view them while studying / reviewing.

## Example Usage
```python
from flashcard_creator import (
    create_flashcards_for_book,
    create_anki_deck,
    AnkiDeckConfig,
    FlashcardPromptConfig,
)

from pdf_toc_reader import process_pdf_and_create_structure


book_title = "Fluent Python: Clear, Concise, and Effective Programming"
book_path = "fluent_python.pdf"

custom_prompt = FlashcardPromptConfig(
    cards_per_page=3,
    model="gpt-4o",
    temperature=0.2,
    template="""Your custom template here, or use default""",
)

# Create directory tree matching ToS
directory = process_pdf_and_create_structure(
    book_path,
    skip_keywords={
        "copyright",
        "about the author",
        "further reading",
    },
)

# Generate flashcards for each leaf ToS section
create_flashcards_for_book(
    book_title=book_title,
    mapping_file=f"{directory}/directory_mapping.json",
    log_file=f"{directory}/flashcard_creation.log",
    prompt_config=custom_prompt,
)

# Create the Anki deck
result = create_anki_deck(
    directory=directory,
    config=AnkiDeckConfig(
        fields=["Front", "FrontCode", "Back", "BackCode", "Comments"],
        code_fields={"FrontCode", "BackCode"},
        render_markdown=True,
        syntax_highlight_html=True,
        separator=",",
    ),
)

print(f"\nCreated Anki csv at: {result.csv_path}")
print(f"Media assets directory: {result.img_assets_dir}")

if result.missing_sections:
    print("\nSections missing flashcards:")
    for section in result.missing_sections:
        print(f"  - {section}")

if result.error_sections:
    print("\nSections with errors during processing:")
    for section in result.error_sections:
        print(f"  - {section}")

```

### Notes
- Mapping and logging files are created in the ToS tree directory.
- Currently uses no batching or concurrent requests. It is not that slow for a whole book and I get rate-limited.
