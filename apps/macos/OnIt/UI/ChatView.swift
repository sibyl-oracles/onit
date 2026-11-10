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
            .onChange(of: session.messages.count) {
                withAnimation(.onit) { proxy.scrollTo("bottom", anchor: .bottom) }
            }
            .onChange(of: session.messages.last?.activityFootprint) { _, _ in
                // Every transcript update pins the reader to the bottom: the
                // window is the agent's progress readout, and a token arriving
                // every few milliseconds must still land the newest line in
                // view. (Scrolled-up reading deliberately loses — that is the
                // requested policy.)
                proxy.scrollTo("bottom", anchor: .bottom)
            }
            // The answer replacing the working state can shrink the transcript
            // or land a large final bubble in one pass; a lazy stack has not
            // laid out its full height yet, so the first scrollTo can undershoot.
            // One deferred re-scroll lands the reader at the true bottom.
            .onChange(of: session.messages.last?.isStreaming) { _, streaming in
                guard streaming == false else { return }
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
        // this, the next dictation session would resurrect them. While the
        // mic is live this also restarts recognition, so the flushed
        // request's buffer cannot re-emit the sent words into the draft.
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
    /// uploads, draft clearing). Spoken "onit, run" goes through the same
    /// door as ⌘Return — never a side door.
    let onSend: () -> Void
    /// The draft as it was when listening began. Words recognized during
    /// this session append after it; the field itself is never rewritten
    /// wholesale, so keyboard edits mid-dictation survive.
    @State private var appendBase = ""

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
        // Spoken commands ride the same stream: "onit, run" sends the
        // composer, "onit, stop" barges in on the running task, "onit,
        // clear" wipes the draft — and after a pause, a bare "run" / "go" /
        // "stop" / "clear" utterance works without the anchor. The center
        // flags the command; the view consumes it exactly once and strips
        // the command words from the draft's source.
        .onChange(of: dictation.snapshot) { _, snap in
            guard snap.isListening, snap.pendingCommand != nil else { return }
            guard let command = dictation.takeCommand() else { return }
            // The command words leave the draft's source; what remains is
            // the task, in its raw casing and punctuation.
            draft = appendBase + separator + dictation.snapshot.currentText
            switch command {
            case .run:
                onSend()
            case .stop:
                session.stop()
                // Drop the recognizer's buffer so the consumed words cannot
                // resurface as the next partial — without clearing the task
                // text, which must survive for the next "onit, run".
                dictation.flush()
            case .clear:
                // Wipe everything: the dictated task *and* the words typed
                // before it. A clear that spared appendBase would leave the
                // user dictating into a draft they believe is empty.
                appendBase = ""
                draft = ""
                // The center's accumulator must go too, not just the
                // recognizer's buffer. takeCommand() strips only the command
                // words — the task text stays in the transcript — and the
                // next partial would ride it straight back into the draft.
                // reset() empties transcript and partial AND restarts
                // recognition with a fresh request, so the mic stays live
                // for the next task.
                dictation.reset()
            }
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
///
/// Read-aloud shares this row: "Summarizing…" while the spoken summary is
/// being prepared and "Speaking…" while it plays, because a listener who
/// just heard the answer land needs to know why the mic is quiet and when
/// it will be live again. Dictation wins the row the moment it is on.
struct DictationStatus: View {
    @Environment(DictationCenter.self) private var dictation
    @Environment(TtsCenter.self) private var tts

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
        } else if tts.phase == .summarizing {
            HStack(spacing: 6) {
                ProgressView()
                    .controlSize(.mini)
                Text("Summarizing…")
                    .font(Typography.caption)
                    .foregroundStyle(Palette.secondaryText)
            }
        } else if tts.isSpeaking {
            HStack(spacing: 6) {
                Image(systemName: "speaker.wave.2.fill")
                    .font(.system(size: 10))
                    .foregroundStyle(Color.accentColor)
                Text("Speaking…")
                    .font(Typography.caption)
                    .foregroundStyle(Palette.secondaryText)
            }
        }
    }
}
