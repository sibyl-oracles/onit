import SwiftUI

/// The transcript and the composer for one session.
struct ChatView: View {
    @Bindable var session: Session
    @Environment(DictationCenter.self) private var dictation
    @State private var draft = ""
    @State private var attachments: [URL] = []
    @State private var isUploading = false
    @State private var uploadError: String?
    @State private var isDropTargeted = false
    /// True while the reader is at (or near) the bottom of the transcript.
    /// Only then does a new token auto-scroll; scrolled-up reading wins.
    @State private var isNearBottom = true
    @FocusState private var composerFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            transcript
            Divider().overlay(Palette.hairline)
            composer
        }
        .background(Palette.surface)
        .navigationTitle(session.title)
        .task { await session.load() }
        .onAppear { composerFocused = true }
    }

    // MARK: - Transcript

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 18) {
                    if session.messages.isEmpty {
                        EmptyTranscript()
                            .padding(.top, 80)
                    }
                    ForEach(session.messages) { message in
                        // .equatable(): a token appending to the streaming
                        // bubble re-renders only that bubble; every other
                        // bubble's body is skipped unless its message changed.
                        MessageBubble(message: message)
                            .equatable()
                            .id(message.id)
                    }
                    // The approval card sits in the flow, next to the turn it
                    // gates, so the command it names is still on screen. The
                    // card gets the absolute deadline and ticks itself.
                    if let approval = session.pendingApproval {
                        ApprovalCard(
                            approval: approval,
                            deadline: session.approvalDeadline
                        ) { decision in
                            session.answer(approval, decision: decision)
                        }
                        .id("approval")
                    }
                    Color.clear.frame(height: 1).id("bottom")
                }
                .padding(.horizontal, Metrics.gutter)
                .padding(.vertical, 20)
                .frame(maxWidth: Metrics.transcriptMaxWidth, alignment: .leading)
                .frame(maxWidth: .infinity)
            }
            .modifier(NearBottomTracker(isNearBottom: $isNearBottom))
            .onChange(of: session.messages.count) {
                withAnimation(.onit) { proxy.scrollTo("bottom", anchor: .bottom) }
            }
            .onChange(of: session.messages.last?.activityFootprint) { _, _ in
                // Auto-scroll only while the reader is already at the bottom.
                // A token arriving every few milliseconds must not yank the
                // scroll position out from under someone reading earlier in
                // the transcript — and each forced scroll is a full
                // scroll-view layout pass, which is half the streaming CPU.
                if isNearBottom {
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }
            // The answer replacing the working state can shrink the transcript
            // or land a large final bubble in one pass; a lazy stack has not
            // laid out its full height yet, so the first scrollTo can undershoot.
            // One deferred re-scroll lands the reader at the true bottom.
            .onChange(of: session.messages.last?.isStreaming) { _, streaming in
                guard streaming == false, isNearBottom else { return }
                withAnimation(.onit) { proxy.scrollTo("bottom", anchor: .bottom) }
                DispatchQueue.main.async {
                    withAnimation(.onit) { proxy.scrollTo("bottom", anchor: .bottom) }
                }
            }
            .onChange(of: session.pendingApproval?.id) { _, newValue in
                if newValue != nil {
                    withAnimation(.onit) { proxy.scrollTo("approval", anchor: .center) }
                    // A gated command must never expire because the window was
                    // behind another one.
                    NSApp.activate(ignoringOtherApps: true)
                }
            }
        }
    }

    // MARK: - Composer

    private var composer: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !attachments.isEmpty {
                HStack(spacing: 6) {
                    ForEach(attachments, id: \.self) { url in
                        HStack(spacing: 4) {
                            Image(systemName: "paperclip")
                                .font(.system(size: 10))
                            Text(url.lastPathComponent)
                                .font(Typography.caption)
                                .lineLimit(1)
                            Button {
                                attachments.removeAll { $0 == url }
                            } label: {
                                Image(systemName: "xmark")
                                    .font(.system(size: 8))
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel("Remove \(url.lastPathComponent)")
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Palette.raised, in: Capsule())
                    }
                }
            }

            if isUploading {
                HStack(spacing: 6) {
                    ProgressView()
                        .controlSize(.mini)
                    Text("Uploading…")
                        .font(Typography.caption)
                        .foregroundStyle(Palette.secondaryText)
                }
            }

            if let uploadError {
                Text(uploadError)
                    .font(Typography.caption)
                    .foregroundStyle(Palette.danger)
                    .textSelection(.enabled)
            }

            DictationStatus()

            HStack(alignment: .bottom, spacing: 8) {
                Button(action: attach) {
                    Image(systemName: "paperclip")
                        .font(.system(size: 13))
                        .foregroundStyle(Palette.secondaryText)
                }
                .buttonStyle(.plain)
                .disabled(session.isProcessing || isUploading)
                .help("Attach a file")
                .accessibilityLabel("Attach a file")

                DictationButton(draft: $draft, session: session, onSend: send)
                TextField(
                    "Ask OnIt…",
                    text: $draft,
                    axis: .vertical
                )
                .textFieldStyle(.plain)
                .font(Typography.body)
                .lineLimit(1...8)
                .focused($composerFocused)
                .onSubmit(send)
                // An upload in flight means the send this field could trigger
                // would race the attachments it is about to carry.
                .disabled(session.isProcessing || isUploading)

                if session.isProcessing {
                    Button(action: session.stop) {
                        Image(systemName: "stop.circle.fill")
                            .font(.system(size: 18))
                            .foregroundStyle(Palette.secondaryText)
                    }
                    .buttonStyle(.plain)
                    .help("Stop (⌘.)")
                    .accessibilityLabel("Stop generating")
                } else {
                    Button(action: send) {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 18))
                            .foregroundStyle(
                                canSend ? Color.accentColor : Palette.tertiaryText
                            )
                    }
                    .buttonStyle(.plain)
                    .disabled(!canSend)
                    .keyboardShortcut(.return, modifiers: .command)
                    .accessibilityLabel("Send message")
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(Palette.raised, in: RoundedRectangle(cornerRadius: Metrics.cardRadius))
            .overlay {
                RoundedRectangle(cornerRadius: Metrics.cardRadius)
                    .stroke(Palette.hairline, lineWidth: 1)
            }
            .overlay {
                // Where a dropped file would land. Only shown while a drag is
                // over the composer, so it costs nothing at rest.
                if isDropTargeted {
                    RoundedRectangle(cornerRadius: Metrics.cardRadius)
                        .stroke(Color.accentColor, lineWidth: 2)
                        .background(Color.accentColor.opacity(0.08), in: RoundedRectangle(cornerRadius: Metrics.cardRadius))
                }
            }
            .dropDestination(for: URL.self) { urls, _ in
                let files = urls.filter { $0.isFileURL }
                guard !files.isEmpty else { return false }
                attachments.append(contentsOf: files)
                composerFocused = true
                return true
            } isTargeted: { hovering in
                isDropTargeted = hovering
            }
        }
        .padding(.horizontal, Metrics.gutter)
        .padding(.vertical, 12)
        .frame(maxWidth: Metrics.transcriptMaxWidth)
        .frame(maxWidth: .infinity)
    }

    private var canSend: Bool {
        !session.isProcessing
            && !isUploading
            && !(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && attachments.isEmpty)
    }

    private func send() {
        guard canSend else { return }
        let text = draft
        let files = attachments
        draft = ""
        attachments = []
        uploadError = nil
        // The dictation accumulator still holds the words just sent; without
        // this, the next dictation session would resurrect them.
        dictation.reset()
        Task {
            var paths: [String] = []
            var failures: [String] = []
            if !files.isEmpty {
                isUploading = true
                for url in files {
                    do {
                        paths.append(try await session.client.upload(session: session.id, fileURL: url))
                    } catch {
                        failures.append("\(url.lastPathComponent) — \(error.localizedDescription)")
                    }
                }
                isUploading = false
            }
            // A failed upload must not vanish: the user believes the model got
            // the file. Restore the draft and the chips, name what failed, and
            // send nothing — a partial attachment set is a silent wrong answer.
            if !failures.isEmpty {
                draft = text
                attachments = files
                uploadError = "Not uploaded:\n" + failures.joined(separator: "\n")
                return
            }
            session.send(text, files: paths)
        }
    }

    private func attach() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.message = "Choose files to send with your message"
        if panel.runModal() == .OK {
            attachments.append(contentsOf: panel.urls)
        }
    }
}

/// Tracks whether the enclosing ScrollView is near its bottom.
///
/// macOS 15's `onScrollGeometryChange` is the clean way to do this; on 14 the
/// sentinel's frame in the scroll coordinate space does the same job. Either
/// way the answer feeds the auto-scroll gate: a token arriving every few
/// milliseconds must not yank the scroll position out from under someone
/// reading earlier in the transcript.
private struct NearBottomTracker: ViewModifier {
    @Binding var isNearBottom: Bool

    func body(content: Content) -> some View {
        if #available(macOS 15.0, *) {
            content.onScrollGeometryChange(for: Bool.self) { geometry in
                // "Near the bottom": the bottom edge is within half a
                // viewport of the visible area. A strict ==0 test would flip
                // false from a half-line of scroll and strand the reader.
                let visible =
                    geometry.bounds.maxY - geometry.contentOffset.y - geometry.containerSize.height
                return visible < geometry.containerSize.height * 0.5
            } action: { _, near in
                isNearBottom = near
            }
            .onScrollGeometryChange(for: CGFloat?.self) { geometry in
                geometry.contentOffset.y
            } action: { old, new in
                // Disarm on a real upward scroll only. The scroll offset is
                // the one axis the user alone controls: transcript growth
                // extends the content below the viewport and leaves the
                // offset untouched, a programmatic auto-scroll only increases
                // it, and only the reader's wheel or trackpad moves it
                // backwards. Treating growth as "the reader scrolled up"
                // would disarm auto-scroll for the rest of the turn.
                if let old, let new, new < old - 1 {
                    isNearBottom = false
                }
            }
        } else {
            // Pre-macOS 15 there is no scroll geometry to read: always follow,
            // which matches the app's behavior before the gate existed.
            content
        }
    }
}

private struct EmptyTranscript: View {
    var body: some View {
        VStack(spacing: 10) {
            Text("What should OnIt do?")
                .font(.system(size: 17, weight: .medium))
            Text("It can read files, run commands and browse — and it will ask before anything destructive.")
                .font(Typography.body)
                .foregroundStyle(Palette.secondaryText)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 380)
        }
        .frame(maxWidth: .infinity)
    }
}

/// The microphone button in the composer: an alternative to typing, backed by
/// Apple's Speech framework (the engine macOS Dictation uses).
///
/// While listening, the button pulses and the recognized text streams into
/// the draft field live — the user watches their words appear as they speak,
/// and can correct by keyboard mid-stream. The draft keeps its own text:
/// dictation appends to whatever is already there, and a second press stops
/// listening while keeping the words.
struct DictationButton: View {
    @Binding var draft: String
    @Environment(DictationCenter.self) private var dictation
    /// The session this composer sends into — spoken "onit, stop" stops it.
    let session: Session
    /// Sends the composer, with all the gating ChatView applies (canSend,
    /// uploads, draft clearing). Spoken "run it" goes through the same door
    /// as ⌘Return — never a side door.
    let onSend: () -> Void
    /// The draft as it was when listening began. Words recognized during
    /// this session append after it; the field itself is never rewritten
    /// wholesale, so keyboard edits mid-dictation survive.
    @State private var appendBase = ""
    /// The normalized tail that last fired a command. The transcript keeps
    /// its words after a command fires, so without this guard every later
    /// partial update would re-fire it — worst case re-killing a *new* task
    /// started after a barge-in. A fresh command changes the tail, so it
    /// still fires.
    @State private var firedTail: String?

    var body: some View {
        let snap = dictation.snapshot
        Button {
            if snap.isListening {
                dictation.stop()
            } else {
                appendBase = draft
                dictation.toggle()
            }
        } label: {
            Image(systemName: snap.isListening ? "mic.fill" : "mic")
                .font(.system(size: 13))
                .foregroundStyle(snap.isListening ? Color.accentColor : Palette.secondaryText)
                .symbolEffect(.pulse, options: .repeating, isActive: snap.isListening)
        }
        .buttonStyle(.plain)
        .help(snap.isListening ? "Stop dictation" : "Dictate (click to speak)")
        .accessibilityLabel(snap.isListening ? "Stop dictation" : "Dictate a message")
        // The stream lands in the field as it is recognized. Base + dictated,
        // never dictated alone, so the user's mid-stream edits survive.
        .onChange(of: snap.currentText) {
            guard snap.isListening else { return }
            draft = appendBase + separator + snap.currentText
        }
        // Spoken commands ride the same stream: "run it" sends the composer,
        // "onit, stop" barges in on the running task. Both fire through the
        // center's callback so the matcher stays testable without a mic.
        .onChange(of: dictation.snapshot) { _, snap in
            guard snap.isListening,
                  let match = VoiceCommand.consume(from: snap.currentText)
            else { return }
            let tail = match.remaining + "|" + String(describing: match.command)
            guard tail != firedTail else { return }
            firedTail = tail
            switch match.command {
            case .run:
                // The command words must not ride into the sent message.
                draft = appendBase + separator + match.remaining
                onSend()
            case .stop:
                session.stop()
            }
            // The buffer that produced the command words keeps them: without
            // this flush, the recognizer's next partial re-emits the whole
            // utterance (task + command) and the just-sent task text lands
            // back in the draft — looking like nothing executed.
            dictation.reset()
        }
        // Leaving the composer (session switch, window close) must not leave
        // a live microphone behind.
        .onDisappear {
            if dictation.snapshot.isListening { dictation.stop() }
        }
    }

    /// One space between the user's own words and the dictated ones — but
    /// only when there are words to separate.
    private var separator: String {
        appendBase.isEmpty || appendBase.hasSuffix(" ") || appendBase.hasSuffix("\n") ? "" : " "
    }
}

/// Live dictation status above the composer row: the hypothesis as it grows,
/// or the error/denial that stopped the stream. Mirrors how `uploadError`
/// is surfaced — close to the input, dismissible by the next attempt.
struct DictationStatus: View {
    @Environment(DictationCenter.self) private var dictation

    var body: some View {
        let snap = dictation.snapshot
        if snap.isListening {
            HStack(spacing: 6) {
                Image(systemName: "waveform")
                    .font(.system(size: 10))
                    .foregroundStyle(Color.accentColor)
                Text(snap.currentText.isEmpty ? "Listening…" : snap.currentText)
                    .font(Typography.caption)
                    .foregroundStyle(snap.currentText.isEmpty ? Palette.secondaryText : .primary)
                    .lineLimit(2)
            }
        } else if let error = snap.error {
            Text(error)
                .font(Typography.caption)
                .foregroundStyle(snap.isDenied ? Palette.danger : Palette.secondaryText)
                .textSelection(.enabled)
        }
    }
}
