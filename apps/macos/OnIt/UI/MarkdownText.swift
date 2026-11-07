import SwiftUI

/// A small Markdown renderer for assistant answers.
///
/// `AttributedString(markdown:)` handles inline emphasis, links and code spans
/// but not block structure, so fenced code blocks are split out first and
/// rendered as their own views. That is the part that matters: an answer with
/// a shell command in it must not be reflowed into prose.
struct MarkdownText: View {
    let text: String

    init(_ text: String) {
        self.text = text
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(Self.blocks(of: text).enumerated()), id: \.offset) { _, block in
                switch block {
                case let .prose(content):
                    if let attributed = try? AttributedString(
                        markdown: content,
                        options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)
                    ) {
                        Text(attributed)
                            .fixedSize(horizontal: false, vertical: true)
                    } else {
                        Text(content)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                case let .heading(heading):
                    Text(heading.content)
                        .font(heading.level <= 2 ? .title3.weight(.semibold) : .headline)
                        .padding(.top, heading.level <= 2 ? 6 : 2)
                case let .list(list):
                    VStack(alignment: .leading, spacing: 3) {
                        ForEach(Array(list.items.enumerated()), id: \.offset) { index, item in
                            HStack(alignment: .top, spacing: 7) {
                                Text(list.ordered ? "\(index + 1)." : "•")
                                    .font(Typography.body.monospacedDigit())
                                    .foregroundStyle(Palette.secondaryText)
                                if let attributed = try? AttributedString(
                                    markdown: item,
                                    options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)
                                ) {
                                    Text(attributed)
                                        .fixedSize(horizontal: false, vertical: true)
                                } else {
                                    Text(item)
                                }
                            }
                        }
                    }
                case let .code(language, source):
                    CodeBlock(language: language, source: source)
                }
            }
        }
    }

    enum Block: Equatable {
        case prose(String)
        case heading(Heading)
        case list(List)
        case code(String?, String)

        /// A `#`-heading. A struct rather than two associated values: enums
        /// with tuple-shaped payloads cannot synthesize `Equatable`, and the
        /// tests compare blocks directly.
        struct Heading: Equatable {
            let content: String
            let level: Int
        }

        /// One run of consecutive list lines.
        struct List: Equatable {
            let items: [String]
            let ordered: Bool
        }
    }

    /// Split on ``` fences, then read the light block structure of what is
    /// left: `#` headings and `-`/`*`/`1.` list lines. `AttributedString` is
    /// inline-only, so without this an agent's step list renders as a run-on
    /// paragraph with bare hyphens in it.
    ///
    /// Deliberately not a Markdown parser: no nesting, no blockquotes, no
    /// tables. A list interrupted by prose is two lists, and numbering
    /// restarts within each — close enough for agent output, honest about
    /// being a splitter.
    static func blocks(of text: String) -> [Block] {
        var result: [Block] = []
        var prose = ""
        var code: String?
        var language: String?
        var listItems: [String] = []
        var listOrdered = false

        func flushProse() {
            if !prose.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                result.append(.prose(prose))
            }
            prose = ""
        }

        func flushList() {
            if !listItems.isEmpty {
                result.append(.list(Block.List(items: listItems, ordered: listOrdered)))
                listItems = []
            }
        }

        func flushText() {
            flushProse()
            flushList()
        }

        for line in text.components(separatedBy: "\n") {
            if line.hasPrefix("```") {
                flushText()
                if code == nil {
                    let tag = line.dropFirst(3).trimmingCharacters(in: .whitespaces)
                    language = tag.isEmpty ? nil : tag
                    code = ""
                } else {
                    result.append(.code(language, code ?? ""))
                    code = nil
                    language = nil
                }
            } else if code != nil {
                code?.append(line)
                code?.append("\n")
            } else if let heading = Self.heading(of: line) {
                flushText()
                result.append(.heading(Block.Heading(content: heading.content, level: heading.level)))
            } else if let item = Self.listItem(of: line) {
                // A list interrupted by prose is a new list; keep the marker
                // style from the line that opened it.
                let ordered = item.ordered
                if !listItems.isEmpty && ordered != listOrdered {
                    flushList()
                }
                if listItems.isEmpty { listOrdered = ordered }
                listItems.append(item.text)
            } else if line.trimmingCharacters(in: .whitespaces).isEmpty {
                // A blank line ends a list but not the prose paragraph.
                flushList()
                prose.append("\n")
            } else {
                flushList()
                prose.append(line)
                prose.append("\n")
            }
        }

        // An unterminated fence still streams: show what arrived rather than
        // dropping it, because a long code block is exactly what a person is
        // waiting to read.
        if let open = code {
            flushText()
            result.append(.code(language, open))
        } else {
            flushText()
        }
        return result
    }

    /// `#` through `######`, one space after. Returns nil for anything else,
    /// including `#hashtag` — no space, not a heading.
    static func heading(of line: String) -> (content: String, level: Int)? {
        guard line.hasPrefix("#") else { return nil }
        let level = line.prefix(while: { $0 == "#" }).count
        guard (1...6).contains(level) else { return nil }
        let rest = line.dropFirst(level)
        guard rest.hasPrefix(" ") else { return nil }
        let content = rest.trimmingCharacters(in: .whitespaces)
        guard !content.isEmpty else { return nil }
        return (content, level)
    }

    /// A list line: `- `, `* `, or `1. ` (a short run of digits, a dot, a space).
    static func listItem(of line: String) -> (text: String, ordered: Bool)? {
        if line.hasPrefix("- ") || line.hasPrefix("* ") {
            let text = String(line.dropFirst(2))
            return text.isEmpty ? nil : (text, false)
        }
        let digits = line.prefix(while: { $0.isNumber })
        guard !digits.isEmpty, digits.count <= 3 else { return nil }
        let afterDigits = line.index(line.startIndex, offsetBy: digits.count)
        guard afterDigits < line.endIndex, line[afterDigits] == "." else { return nil }
        let afterDot = line.index(afterDigits, offsetBy: 1)
        guard afterDot < line.endIndex, line[afterDot] == " " else { return nil }
        let text = String(line[line.index(afterDot, offsetBy: 1)...])
        return text.isEmpty ? nil : (text, true)
    }
}

private struct CodeBlock: View {
    let language: String?
    let source: String
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(language ?? "code")
                    .font(Typography.caption)
                    .foregroundStyle(Palette.tertiaryText)
                Spacer()
                Button(copied ? "Copied" : "Copy") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(source, forType: .string)
                    copied = true
                    Task {
                        try? await Task.sleep(for: .seconds(1.2))
                        copied = false
                    }
                }
                .buttonStyle(.plain)
                .font(Typography.caption)
                .foregroundStyle(Palette.secondaryText)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)

            Divider().overlay(Palette.hairline)

            ScrollView(.horizontal, showsIndicators: false) {
                Text(source)
                    .font(Typography.mono)
                    .textSelection(.enabled)
                    .padding(10)
            }
        }
        .background(Palette.raised, in: RoundedRectangle(cornerRadius: Metrics.controlRadius))
        .overlay {
            RoundedRectangle(cornerRadius: Metrics.controlRadius)
                .stroke(Palette.hairline, lineWidth: 1)
        }
    }
}
