import logging
import sys

from flashcard_creator import (
    create_flashcards_for_book,
    create_anki_deck,
    AnkiDeckConfig,
    FlashcardPromptConfig,
)

from pdf_toc_reader import process_pdf_and_create_structure


def main():
    book_title = "Fluent Python: Clear, Concise, and Effective Programming"
    book_path = "fluent_python.pdf"

    custom_prompt = FlashcardPromptConfig(
        cards_per_page=3,
        model="gpt-4o",
        template="""You are an expert in computer science education and Python, helping students master content from the book *{book_title}* using spaced repetition. You are presented with screenshots of the chapter '{chapter_path}'. Create flashcards that cover key concepts and insights from the chapter, on average {cards_per_page} cards per page.

**Guidelines:**
- **Variety:** Include questions ranging from basic to complex, presenting facts from different angles.
- **Self-Containment:** Ensure each question is self-contained, similar to exam questions.
  - If a question uses code that is specific to the book but not the general topic, include the necessary code in the **FrontCode** field.
  - **No Answer Leaks:** Do not include code or comments in **FrontCode** that reveal the answer to the question. Such code should instead be placed in **BackCode**.
- **Techniques:** Use examples and mnemonic devices where possible.
- **Clarity:** Craft concise, engaging questions and answers, simplifying complex ideas into clear maxims.

**Output Format:** Provide a JSON list of cards, each with the following fields using Markdown syntax where applicable:
- **Front:** A concise, self-contained question that does not require additional context.
- **FrontCode:** All necessary code snippets from the chapter needed to understand and answer the question. Ensure that **FrontCode** does not contain code or comments that leak the answer.
- **Back:** A brief answer or explanation.
- **BackCode:** Optional concise, properly formatted code example from the chapter that illustrates the answer. Leave empty if it is included in **FrontCode**.
- **Comments:** Optional mnemonics or study notes to aid learning.

**Note:** Base each card solely on the content of "{chapter_title}", ignoring any overlapping content from adjacent chapters in the screenshots.""",
    )

    try:
        # Create directory tree matching ToS
        directory = process_pdf_and_create_structure(
            book_path,
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

    except Exception as e:
        logging.error("Job failed", exc_info=True)
        print("Job failed", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
