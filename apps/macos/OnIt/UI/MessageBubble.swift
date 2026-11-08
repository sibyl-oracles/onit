import SwiftUI

/// One message in the transcript.
///
/// The assistant's reasoning is folded into a disclosure above the answer and
/// closes itself when the answer arrives — the same behaviour the browser has,
/// which is why the backend keeps `think` and `token` on separate channels.
///
/// The view is `Equatable` and applied with `.equatable()`: a token that
/// appends to the streaming bubble must not re-render the other bubbles, and
/// each non-streaming bubble's body is skipped entirely unless its message
/// changed. Without this, every token invalidates the whole transcript body
/// and every bubble re-runs its Markdown parse.
struct MessageBubble: View, Equatable {
    let message: ChatMessage
    @State private var showReasoning = false

    static func == (lhs: MessageBubble, rhs: MessageBubble) -> Bool {
        lhs.message == rhs.message
    }

    var body: some View {
        switch message.role {
        case .user:
            userBubble
        case .assistant:
            assistantBubble
        case .system:
            Text(message.content)
                .font(Typography.caption)
                .foregroundStyle(Palette.tertiaryText)
        }
    }

    private var userBubble: some View {
        HStack {
            Spacer(minLength: 40)
            Text(message.content)
                .font(Typography.body)
                .lineSpacing(Typography.lineSpacing)
                .textSelection(.enabled)
                .padding(.horizontal, 12)
                .padding(.vertical, 9)
                .background(Palette.userBubble, in: RoundedRectangle(cornerRadius: Metrics.cardRadius))
        }
    }

    private var assistantBubble: some View {
        VStack(alignment: .leading, spacing: 10) {
            if !message.reasoning.isEmpty {
                reasoningDisclosure
            }

            if !message.steps.isEmpty {
                ToolLedger(steps: message.steps)
            }

            if !message.content.isEmpty {
                // While the answer is streaming it renders as plain text:
                // a token arriving every few milliseconds must not re-run a
                // full Markdown block parse on a growing string. The parse
                // happens once, when the turn completes and isStreaming
                // flips false.
                if message.isStreaming {
                    Text(message.content)
                        .font(Typography.body)
                        .lineSpacing(Typography.lineSpacing)
                        .textSelection(.enabled)
                } else {
                    MarkdownText(message.content)
                        .font(Typography.body)
                        .lineSpacing(Typography.lineSpacing)
                        .textSelection(.enabled)
                }
            }

            if message.isStreaming && message.content.isEmpty {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.mini)
                    Text(message.status ?? "Working…")
                        .font(Typography.caption)
                        .foregroundStyle(Palette.secondaryText)
                }
            }

            if !message.files.isEmpty {
                FileChips(files: message.files)
            }

            if let correction = message.correction {
                CorrectionNote(correction: correction)
            }

            if let error = message.error {
                Text(error)
                    .font(Typography.caption)
                    .foregroundStyle(Palette.danger)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Palette.danger.opacity(0.08), in: RoundedRectangle(cornerRadius: Metrics.controlRadius))
            }

            footer
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var reasoningDisclosure: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button {
                withAnimation(.onit) { showReasoning.toggle() }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: showReasoning ? "chevron.down" : "chevron.right")
                        .font(.system(size: 9, weight: .semibold))
                    Text(message.isStreaming ? "Thinking…" : "Thought for a moment")
                        .font(Typography.caption)
                }
                .foregroundStyle(Palette.tertiaryText)
            }
            .buttonStyle(.plain)

            if showReasoning {
                Text(message.reasoning)
                    .font(Typography.caption)
                    .foregroundStyle(Palette.secondaryText)
                    .lineSpacing(3)
                    .padding(.leading, 13)
                    .textSelection(.enabled)
            }
        }
    }

    @ViewBuilder
    private var footer: some View {
        if let elapsed = message.elapsed, !message.isStreaming {
            HStack(spacing: 10) {
                Text(String(format: "%.1fs", elapsed))
                    .font(Typography.caption)
                    .foregroundStyle(Palette.tertiaryText)

                if message.ratingEnabled, let turn = message.turn {
                    RatingButtons(rating: message.rating, turn: turn)
                }
            }
        }
    }
}

/// Thumbs on an answer.
///
/// The one signal no heuristic can supply: whether the person who asked got
/// what they wanted. Only drawn when the server says recording is on — a
/// button that accepts a click and drops it is worse than no button.
private struct RatingButtons: View {
    let rating: Int?
    let turn: Int
    @Environment(AppModel.self) private var model

    var body: some View {
        HStack(spacing: 4) {
            button(systemName: "hand.thumbsup", value: 1)
            button(systemName: "hand.thumbsdown", value: -1)
        }
    }

    private func button(systemName: String, value: Int) -> some View {
        Button {
            guard let session = model.selectedSession else { return }
            session.rate(rating == value ? nil : value)
        } label: {
            Image(systemName: rating == value ? systemName + ".fill" : systemName)
                .font(.system(size: 10))
                .foregroundStyle(rating == value ? Color.accentColor : Palette.tertiaryText)
        }
        .buttonStyle(.plain)
    }
}

private struct CorrectionNote: View {
    let correction: Correction

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label("Correction", systemImage: "checkmark.seal")
                .font(Typography.caption)
                .foregroundStyle(Color.orange)
            if !correction.note.isEmpty {
                Text(correction.note)
                    .font(Typography.caption)
                    .foregroundStyle(Palette.secondaryText)
            }
            if !correction.content.isEmpty {
                MarkdownText(correction.content)
                    .font(Typography.caption)
                    .textSelection(.enabled)
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: Metrics.controlRadius))
    }
}

private struct FileChips: View {
    let files: [FileInfo]
    @Environment(AppModel.self) private var model
    @Environment(AudioCenter.self) private var audio

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(files) { file in
                if file.isAudio {
                    AudioRow(file: file)
                } else {
                    fileRow(file)
                }
            }
        }
    }

    /// The row every non-audio attachment gets: icon, name, size, Open.
    private func fileRow(_ file: FileInfo) -> some View {
        HStack(spacing: 6) {
            Image(systemName: file.isImage ? "photo" : (file.isCode ? "chevron.left.forwardslash.chevron.right" : "doc"))
                .font(.system(size: 10))
            Text(file.name)
                .font(Typography.caption)
                .lineLimit(1)
            Text(byteString(file.size))
                .font(Typography.caption)
                .foregroundStyle(Palette.tertiaryText)
            Spacer()
            Button("Open") { open(file) }
                .buttonStyle(.plain)
                .font(Typography.caption)
                .foregroundStyle(Color.accentColor)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(Palette.raised, in: RoundedRectangle(cornerRadius: Metrics.controlRadius))
    }

    private func open(_ file: FileInfo) {
        guard let base = model.selectedConnection?.baseURL,
              let url = URL(string: file.url, relativeTo: base) else { return }
        NSWorkspace.shared.open(url)
    }

    private func byteString(_ bytes: Int) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }
}

/// One playable attachment: a Play/Stop button, the elapsed side of a
/// progress bar, and the name.
///
/// The row reads the center's state for its URL, not its own @State, so a
/// second Play anywhere in the app stops this one and the bar follows.
/// Playback itself stays in AudioCenter: a row does not own a player that
/// outlives its message.
private struct AudioRow: View {
    let file: FileInfo
    @Environment(AppModel.self) private var model
    @Environment(AudioCenter.self) private var audio

    var body: some View {
        let state = audio.state(for: file.url)
        HStack(spacing: 8) {
            Button {
                Task { await toggle(state.isPlaying) }
            } label: {
                Image(systemName: state.isPlaying ? "stop.circle.fill" : "play.circle.fill")
                    .font(.system(size: 18))
                    .foregroundStyle(Color.accentColor)
            }
            .buttonStyle(.plain)
            .help(state.isPlaying ? "Stop" : "Play \(file.name)")
            .accessibilityLabel(state.isPlaying ? "Stop \(file.name)" : "Play \(file.name)")

            VStack(alignment: .leading, spacing: 3) {
                Text(file.name)
                    .font(Typography.caption)
                    .lineLimit(1)
                if state.isPlaying, state.duration > 0 {
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Capsule().fill(Palette.tertiaryText.opacity(0.3))
                            Capsule()
                                .fill(Color.accentColor)
                                .frame(width: geo.size.width * progress(state))
                        }
                    }
                    .frame(height: 3)
                }
            }

            Spacer()

            if state.isPlaying {
                Text(timeString(state.elapsed))
                    .font(Typography.caption.monospacedDigit())
                    .foregroundStyle(Palette.tertiaryText)
            } else if state.loadFailed {
                Text("cannot play")
                    .font(Typography.caption)
                    .foregroundStyle(Palette.tertiaryText)
            } else {
                Text(byteString(file.size))
                    .font(Typography.caption)
                    .foregroundStyle(Palette.tertiaryText)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(Palette.raised, in: RoundedRectangle(cornerRadius: Metrics.controlRadius))
    }

    private func toggle(_ wasPlaying: Bool) async {
        if wasPlaying {
            audio.stopAll()
            return
        }
        guard let data = await model.attachmentData(file.url) else { return }
        audio.toggle(url: file.url, data: data)
    }

    private func progress(_ state: AudioCenter.RowState) -> CGFloat {
        guard state.duration > 0 else { return 0 }
        return CGFloat(min(1, state.elapsed / state.duration))
    }

    private func timeString(_ t: TimeInterval) -> String {
        String(format: "%d:%02d", Int(t) / 60, Int(t) % 60)
    }

    private func byteString(_ bytes: Int) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }
}
