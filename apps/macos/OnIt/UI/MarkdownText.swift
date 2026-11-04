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
                case let .code(language, source):
                    CodeBlock(language: language, source: source)
                }
            }
        }
    }

    enum Block {
        case prose(String)
        case code(String?, String)
    }

    /// Split on ``` fences. Everything outside a fence is prose.
    static func blocks(of text: String) -> [Block] {
        var result: [Block] = []
        var prose = ""
        var code: String?
        var language: String?

        for line in text.components(separatedBy: "\n") {
            if line.hasPrefix("```") {
                if code == nil {
                    if !prose.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        result.append(.prose(prose))
                    }
                    prose = ""
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
            } else {
                prose.append(line)
                prose.append("\n")
            }
        }

        // An unterminated fence still streams: show what arrived rather than
        // dropping it, because a long code block is exactly what a person is
        // waiting to read.
        if let open = code {
            result.append(.code(language, open))
        } else if !prose.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            result.append(.prose(prose))
        }
        return result
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
