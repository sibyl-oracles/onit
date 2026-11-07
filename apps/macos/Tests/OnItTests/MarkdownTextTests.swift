import XCTest
@testable import OnIt

/// The Markdown splitter. The contract it must hold: a fenced command is
/// never reflowed into prose, an unterminated fence still streams, and the
/// light block structure (headings, lists) survives.
///
/// MainActor like the view it serves — the splitter runs during rendering.
@MainActor
final class MarkdownTextTests: XCTestCase {
    func testProseOnly() {
        XCTAssertEqual(
            MarkdownText.blocks(of: "Just some text."),
            [.prose("Just some text.\n")]
        )
    }

    func testEmptyTextYieldsNoBlocks() {
        XCTAssertTrue(MarkdownText.blocks(of: "").isEmpty)
        XCTAssertTrue(MarkdownText.blocks(of: "   \n  \n").isEmpty)
    }

    func testFencedCodeWithLanguage() {
        let blocks = MarkdownText.blocks(of: "Before.\n```bash\nls -la\n```\nAfter.")
        XCTAssertEqual(
            blocks,
            [.prose("Before.\n"), .code("bash", "ls -la\n"), .prose("After.\n")]
        )
    }

    func testUnterminatedFenceStillStreams() {
        // A long code block is exactly what a person is waiting to read; the
        // open fence must render as code, not as mangled prose.
        let blocks = MarkdownText.blocks(of: "Run this:\n```python\ndf.head()")
        XCTAssertEqual(blocks, [.prose("Run this:\n"), .code("python", "df.head()\n")])
    }

    func testFenceWithoutLanguage() {
        let blocks = MarkdownText.blocks(of: "```\nplain\n```")
        XCTAssertEqual(blocks, [.code(nil, "plain\n")])
    }

    func testHeadingLevels() {
        XCTAssertEqual(MarkdownText.heading(of: "# Title")?.level, 1)
        XCTAssertEqual(MarkdownText.heading(of: "### Sub")?.level, 3)
        XCTAssertEqual(MarkdownText.heading(of: "###### Deep")?.level, 6)
        XCTAssertNil(MarkdownText.heading(of: "####### Too many"), "7 hashes is not a heading")
        XCTAssertNil(MarkdownText.heading(of: "#hashtag"), "no space, not a heading")
        XCTAssertNil(MarkdownText.heading(of: "# "), "empty heading")
        XCTAssertNil(MarkdownText.heading(of: "plain"))
        XCTAssertEqual(MarkdownText.heading(of: "## Setup")?.content, "Setup")
    }

    func testHeadingBlock() {
        let blocks = MarkdownText.blocks(of: "Intro\n## Setup\nDo it.")
        XCTAssertEqual(
            blocks,
            [.prose("Intro\n"), .heading(.init(content: "Setup", level: 2)), .prose("Do it.\n")]
        )
    }

    func testBulletsGroupIntoOneList() {
        let blocks = MarkdownText.blocks(of: "- one\n- two\n- three")
        XCTAssertEqual(blocks, [.list(.init(items: ["one", "two", "three"], ordered: false))])
    }

    func testAsteriskBullets() {
        let blocks = MarkdownText.blocks(of: "* one\n* two")
        XCTAssertEqual(blocks, [.list(.init(items: ["one", "two"], ordered: false))])
    }

    func testOrderedList() {
        let blocks = MarkdownText.blocks(of: "1. first\n2. second\n10. tenth")
        XCTAssertEqual(
            blocks,
            [.list(.init(items: ["first", "second", "tenth"], ordered: true))]
        )
    }

    func testListInterruptedByProseIsTwoLists() {
        let blocks = MarkdownText.blocks(of: "- a\n\nmid\n\n- b")
        XCTAssertEqual(
            blocks,
            [.list(.init(items: ["a"], ordered: false)), .prose("\nmid\n\n"), .list(.init(items: ["b"], ordered: false))]
        )
    }

    func testOrderedAndUnorderedDoNotMerge() {
        let blocks = MarkdownText.blocks(of: "1. a\n- b")
        XCTAssertEqual(
            blocks,
            [.list(.init(items: ["a"], ordered: true)), .list(.init(items: ["b"], ordered: false))]
        )
    }

    func testHorizontalRuleIsNotAList() {
        // "---" has no space after the dash; it must stay prose, not become
        // an item with empty text.
        XCTAssertNil(MarkdownText.listItem(of: "---"))
        XCTAssertNil(MarkdownText.listItem(of: "-item"))
        XCTAssertNil(MarkdownText.listItem(of: "1.item"))
        XCTAssertNil(MarkdownText.listItem(of: "plain"))
        XCTAssertEqual(MarkdownText.listItem(of: "- item")?.text, "item")
        XCTAssertEqual(MarkdownText.listItem(of: "- item")?.ordered, false)
        XCTAssertEqual(MarkdownText.listItem(of: "3. go")?.text, "go")
        XCTAssertEqual(MarkdownText.listItem(of: "3. go")?.ordered, true)
        XCTAssertNil(MarkdownText.listItem(of: "12345. too long"))
        XCTAssertNil(MarkdownText.listItem(of: "1. "))
    }

    func testListInsideFenceIsCode() {
        // The fence wins: a shell script with comment dashes is code.
        let blocks = MarkdownText.blocks(of: "```\n# not a heading\n- not a list\n```")
        XCTAssertEqual(blocks, [.code(nil, "# not a heading\n- not a list\n")])
    }

    func testMixedDocument() {
        let text = """
        ## Plan
        - fetch
        - run

        ```bash
        onit serve web
        ```

        Done. See `1.` above.
        """
        let blocks = MarkdownText.blocks(of: text)
        XCTAssertEqual(blocks.count, 4)
        XCTAssertEqual(blocks[0], .heading(.init(content: "Plan", level: 2)))
        XCTAssertEqual(blocks[1], .list(.init(items: ["fetch", "run"], ordered: false)))
        guard case let .code(lang, source) = blocks[2] else {
            return XCTFail("expected code, got \(blocks[2])")
        }
        XCTAssertEqual(lang, "bash")
        XCTAssertTrue(source.contains("onit serve web"))
        XCTAssertEqual(blocks[3], .prose("\nDone. See `1.` above.\n"))
    }

    func testStreamingPrefixesAreStable() {
        // The same text, arrived in two streaming halves, must split the way
        // the whole does — the transcript re-splits on every token.
        let whole = MarkdownText.blocks(of: "- a\n- b\ntail")
        let half = MarkdownText.blocks(of: "- a\n- b\n")
        XCTAssertEqual(half, [.list(.init(items: ["a", "b"], ordered: false))])
        XCTAssertEqual(whole.dropLast(), half)
    }
}
